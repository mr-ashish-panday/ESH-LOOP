"""
ESH-Loop Generation & Visualization
=====================================
Generate text and visualize per-token pondering behavior.
Shows which tokens the model "thinks longer" about.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from esh_loop import ESHLoopModel, ESHLoopConfig


def create_model():
    config = ESHLoopConfig(
        d_model=768, n_layers=8, n_heads=12, n_experts=8,
        expert_dim=3072, max_seq_len=2048,
        max_ponder_steps=3, dropout=0.0, layer_scale_init=1e-5,
    )
    return ESHLoopModel(config), config


def generate_with_pondering(model, tokenizer, prompt, max_new_tokens=30,
                            temperature=0.8, device="cpu"):
    """Generate text and track per-token routing & pondering."""
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    generated_tokens = []
    routing_info = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(tokens, return_routing_stats=True)
            logits = outputs["logits"][:, -1, :] / temperature

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Track routing for the last generated token
            token_text = tokenizer.decode(next_token[0])
            avg_alpha = sum(s["alpha_mean"] for s in outputs["routing_stats"]) / len(outputs["routing_stats"])
            avg_ponder = sum(s["avg_ponder_steps"] for s in outputs["routing_stats"]) / len(outputs["routing_stats"])

            generated_tokens.append(token_text)
            routing_info.append({
                "token": token_text,
                "alpha": avg_alpha,
                "ponder_steps": avg_ponder,
            })

    generated_text = "".join(generated_tokens)
    return generated_text, routing_info


def print_pondering_summary(routing_info):
    """Print colored summary of pondering behavior."""
    print(f"\n{'═' * 60}")
    print(f"  Pondering Summary")
    print(f"{'═' * 60}")

    total = len(routing_info)
    avg_alpha = sum(r["alpha"] for r in routing_info) / total
    avg_ponder = sum(r["ponder_steps"] for r in routing_info) / total

    quick = sum(1 for r in routing_info if r["ponder_steps"] < 1.5)
    medium = sum(1 for r in routing_info if 1.5 <= r["ponder_steps"] < 2.5)
    deep = sum(1 for r in routing_info if r["ponder_steps"] >= 2.5)

    print(f"Total tokens: {total}")
    print(f"Average α: {avg_alpha:.3f}")
    print(f"Average ponder steps: {avg_ponder:.2f}")
    print(f"  Quick (1 pass):  {quick} ({quick/total*100:.1f}%)")
    print(f"  Medium (2 pass): {medium} ({medium/total*100:.1f}%)")
    print(f"  Deep (3 pass):   {deep} ({deep/total*100:.1f}%)")

    # Token-level detail
    print(f"\nPer-token breakdown:")
    print(f"{'Token':<15} | {'α':>6} | {'Ponder':>6} | {'Depth'}")
    print("-" * 50)
    for r in routing_info:
        depth_bar = "█" * int(r["ponder_steps"])
        print(f"{r['token']:<15} | {r['alpha']:>6.3f} | {r['ponder_steps']:>6.2f} | {depth_bar}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate with ESH-Loop")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device

    # Model
    print("Creating ESH-Loop model...")
    model, config = create_model()

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "unknown")
        print(f"Loaded from step {step}")
    model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        text, routing_info = generate_with_pondering(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens, device=device,
        )
        print(f"Generated: {text}")
        print_pondering_summary(routing_info)
    else:
        # Default test: Math vs Story
        prompts = [
            ("MATH", "Question: What is 5 + 7? Answer:"),
            ("STORY", "Once upon a time there was a little bunny"),
        ]
        for label, prompt in prompts:
            print(f"\n{'=' * 60}")
            print(f"  [{label}] Prompt: {prompt}")
            print(f"{'=' * 60}")
            text, routing_info = generate_with_pondering(
                model, tokenizer, prompt,
                max_new_tokens=30, device=device,
            )
            print(f"Generated: {text}")
            print_pondering_summary(routing_info)


if __name__ == "__main__":
    main()
