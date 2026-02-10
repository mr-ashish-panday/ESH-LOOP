"""
Data Loader for ESH-Loop
========================
Mixed complexity dataset: TinyStories + WikiText-103 + GSM8K
(Same as ESH Paper 1 for direct comparison)
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import random


class MixedComplexityDataset(IterableDataset):
    """
    Streams from multiple datasets with configurable ratios.
    Default: 70% TinyStories, 20% WikiText-103, 10% GSM8K
    """

    def __init__(self, tokenizer, max_length=512,
                 ratios=None, split="train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ratios = ratios or {"tinystories": 0.7, "wikitext": 0.2, "gsm8k": 0.1}
        self.split = split

        # Load datasets (streaming to save RAM)
        print("Loading TinyStories (streaming)...")
        self.tinystories = iter(load_dataset(
            "roneneldan/TinyStories", split=split, streaming=True
        ))

        print("Loading WikiText-103 (streaming)...")
        self.wikitext = iter(load_dataset(
            "wikitext", "wikitext-103-raw-v1", split=split, streaming=True
        ))

        print("Loading GSM8K (streaming)...")
        self.gsm8k = iter(load_dataset(
            "openai/gsm8k", "main", split=split, streaming=True
        ))

        print("Mixed dataset ready!")

    def _get_text(self, source):
        """Get next text from a source, with wraparound."""
        try:
            if source == "tinystories":
                item = next(self.tinystories)
                return item.get("text", "")
            elif source == "wikitext":
                item = next(self.wikitext)
                return item.get("text", "")
            elif source == "gsm8k":
                item = next(self.gsm8k)
                q = item.get("question", "")
                a = item.get("answer", "")
                return f"Question: {q}\nAnswer: {a}"
        except StopIteration:
            # Restart the iterator
            if source == "tinystories":
                self.tinystories = iter(load_dataset(
                    "roneneldan/TinyStories", split=self.split, streaming=True
                ))
            elif source == "wikitext":
                self.wikitext = iter(load_dataset(
                    "wikitext", "wikitext-103-raw-v1", split=self.split, streaming=True
                ))
            elif source == "gsm8k":
                self.gsm8k = iter(load_dataset(
                    "openai/gsm8k", "main", split=self.split, streaming=True
                ))
            return self._get_text(source)

    def __iter__(self):
        sources = list(self.ratios.keys())
        weights = list(self.ratios.values())

        while True:
            # Sample source based on ratios
            source = random.choices(sources, weights=weights, k=1)[0]
            text = self._get_text(source)

            if not text or len(text.strip()) < 10:
                continue

            # Tokenize
            tokens = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt"
            )

            yield {
                "input_ids": tokens["input_ids"].squeeze(0),
                "source": source,
            }


def create_data_loaders(tokenizer, batch_size=4, max_length=512,
                        num_workers=0):
    """Create train and eval data loaders."""

    train_dataset = MixedComplexityDataset(
        tokenizer, max_length=max_length, split="train"
    )

    # Eval: load fixed samples
    print("Loading eval samples...")
    eval_texts = []
    sources = [
        ("roneneldan/TinyStories", "train", "text", 50),
        ("wikitext", "wikitext-103-raw-v1", "text", 30),
        ("openai/gsm8k", "main", "question", 20),
    ]

    for name_or_path, config_or_split, field, n in sources:
        try:
            if config_or_split in ["train"]:
                ds = load_dataset(name_or_path, split=config_or_split, streaming=True)
            else:
                ds = load_dataset(name_or_path, config_or_split, split="test", streaming=True)

            count = 0
            for item in ds:
                text = item.get(field, "")
                if text and len(text.strip()) > 10:
                    eval_texts.append(text)
                    count += 1
                    if count >= n:
                        break
        except Exception as e:
            print(f"  Warning: Could not load eval from {name_or_path}: {e}")

    print(f"Loaded {len(eval_texts)} eval examples")

    # Tokenize eval
    eval_tokens = tokenizer(
        eval_texts, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt"
    )

    eval_dataset = torch.utils.data.TensorDataset(eval_tokens["input_ids"])
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, eval_loader
