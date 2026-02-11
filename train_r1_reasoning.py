"""
ESH-Loop-Reason: GRPO Fine-tuning for Adaptive Reasoning
==========================================================
DeepSeek R1-Zero style RL training on GSM8K math problems.

The model learns to allocate more ponder steps to harder problems
through reinforcement learning — no explicit ponder cost needed.

Usage:
    nohup python train_r1_reasoning.py --device cuda > training_rl.log 2>&1 &
"""

import re
import sys
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from esh_loop import ESHLoopModel, ESHLoopConfig


# ============================================================
# Math Answer Extraction & Verification
# ============================================================

def extract_number(text):
    """Extract the last number from generated text."""
    # Try to find number after "Answer:" or "=" or "is"
    patterns = [
        r'[Aa]nswer[:\s]*[\$]?\s*([-]?\d+[\.,]?\d*)',
        r'=\s*([-]?\d+[\.,]?\d*)',
        r'is\s+([-]?\d+[\.,]?\d*)',
        r'([-]?\d+[\.,]?\d*)\s*$',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except ValueError:
                continue
    # Fallback: find any number
    numbers = re.findall(r'[-]?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def extract_gsm8k_answer(answer_text):
    """Extract the final numerical answer from GSM8K format (#### NUMBER)."""
    match = re.search(r'####\s*([-]?\d+[\.,]?\d*)', answer_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return extract_number(answer_text)


def check_answer(generated, ground_truth):
    """Check if generated answer matches ground truth."""
    gen_num = extract_number(generated)
    gt_num = extract_gsm8k_answer(ground_truth)
    if gen_num is None or gt_num is None:
        return False
    # Allow small floating point tolerance
    return abs(gen_num - gt_num) < 0.01


# ============================================================
# Generation with Ponder Tracking
# ============================================================

@torch.no_grad()
def generate_completion(model, input_ids, max_new_tokens=64,
                        temperature=0.8, device="cuda"):
    """Generate a single completion and track ponder depth per token."""
    model.eval()
    tokens = input_ids.clone()
    total_ponder = 0.0
    n_generated = 0

    for _ in range(max_new_tokens):
        if tokens.shape[1] >= 256:  # Max seq len
            break

        outputs = model(tokens, return_routing_stats=True)
        logits = outputs["logits"][:, -1, :] / temperature

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Track ponder depth
        avg_ponder = sum(
            s["avg_ponder_steps"] for s in outputs["routing_stats"]
        ) / len(outputs["routing_stats"])
        total_ponder += avg_ponder
        n_generated += 1

        # Stop at EOS or period followed by newline
        if next_token.item() == 50256:  # GPT-2 EOS
            break

    avg_ponder_depth = total_ponder / max(n_generated, 1)
    return tokens, avg_ponder_depth, n_generated


# ============================================================
# GRPO: Group Relative Policy Optimization
# ============================================================

def compute_log_probs(model, input_ids, response_start_idx, device="cuda"):
    """Compute log probabilities of the response tokens under current policy."""
    model.train()

    with autocast(enabled=True):
        outputs = model(input_ids, labels=input_ids, return_routing_stats=True)
        logits = outputs["logits"]

    # Only look at response tokens (after the prompt)
    response_logits = logits[:, response_start_idx - 1:-1, :]  # Shifted
    response_targets = input_ids[:, response_start_idx:]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(2, response_targets.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum(dim=-1), outputs  # Sum over tokens


def grpo_step(model, optimizer, scaler, question_ids, completions, rewards,
              ref_log_probs, prompt_len, device="cuda", kl_coeff=0.04):
    """
    One GRPO optimization step.

    completions: list of (full_ids, ponder_depth, n_tokens)
    rewards: list of float rewards
    ref_log_probs: list of reference log probs (from before update)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Compute advantages (group-normalized)
    rewards_tensor = torch.tensor(rewards, device=device)
    if rewards_tensor.std() > 1e-6:
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    else:
        advantages = torch.zeros_like(rewards_tensor)

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for i, (full_ids, ponder_depth, n_tokens) in enumerate(completions):
        if n_tokens == 0:
            continue

        full_ids = full_ids.to(device)
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)

        with autocast(enabled=True):
            cur_log_prob, outputs = compute_log_probs(
                model, full_ids, prompt_len, device
            )

        # Policy gradient with advantage
        # L = -advantage * log_prob + kl_penalty
        advantage = advantages[i]

        # KL penalty (prevent drift from reference)
        if ref_log_probs[i] is not None:
            kl_penalty = kl_coeff * (cur_log_prob - ref_log_probs[i].detach())
        else:
            kl_penalty = torch.tensor(0.0, device=device)

        step_loss = -(advantage * cur_log_prob) + kl_penalty
        total_loss = total_loss + step_loss.squeeze()
        n_valid += 1

    if n_valid > 0:
        total_loss = total_loss / n_valid
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    return total_loss.item() if n_valid > 0 else 0.0


# ============================================================
# Main Training Loop
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESH-Loop RL Reasoning")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/adaptive_final.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of completions per question (G)")
    parser.add_argument("--max_gen_tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--ponder_penalty", type=float, default=0.05,
                        help="Penalty per ponder step in reward")
    parser.add_argument("--kl_coeff", type=float, default=0.04)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    device = args.device

    # ── Model ──
    print("=" * 60)
    print("  ESH-Loop-Reason: RL Fine-tuning (GRPO)")
    print("=" * 60)

    config = ESHLoopConfig(
        d_model=768, n_layers=8, n_heads=12, n_experts=4,
        expert_dim=2048, max_seq_len=256,
        max_ponder_steps=3, dropout=0.0, layer_scale_init=1e-5,
    )
    model = ESHLoopModel(config)

    print(f"Loading base model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from step {checkpoint.get('step', '?')}")
    model.to(device)
    print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset: GSM8K ──
    print("\nLoading GSM8K dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="train")
    print(f"  Loaded {len(ds)} training problems")

    # Format questions as prompts
    questions = []
    for item in ds:
        q = item["question"].strip()
        a = item["answer"].strip()
        # Create a concise prompt
        prompt = f"Question: {q}\nAnswer:"
        questions.append({"prompt": prompt, "answer": a})

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler()

    # ── Training ──
    print(f"\nStarting GRPO training...")
    print(f"  Steps: {args.max_steps}")
    print(f"  Group size (G): {args.group_size}")
    print(f"  Max gen tokens: {args.max_gen_tokens}")
    print(f"  Ponder penalty: {args.ponder_penalty}")
    print(f"  KL coefficient: {args.kl_coeff}")
    print(f"  Learning rate: {args.lr}")
    print()

    checkpoint_dir = Path("./checkpoints_rl")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Metrics tracking
    total_correct = 0
    total_questions = 0
    total_ponder = 0.0
    total_reward = 0.0
    step_start = time.time()

    for step in range(1, args.max_steps + 1):
        # Sample a random question
        q_idx = random.randint(0, len(questions) - 1)
        q = questions[q_idx]

        # Tokenize prompt
        prompt_tokens = tokenizer(q["prompt"], return_tensors="pt",
                                   truncation=True, max_length=192)
        prompt_ids = prompt_tokens["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]

        # Skip if prompt is too long
        if prompt_len > 192:
            continue

        # ── Generate G completions ──
        completions = []
        rewards = []
        ref_log_probs = []
        step_correct = 0
        step_ponder = 0.0

        for g in range(args.group_size):
            # Generate
            full_ids, ponder_depth, n_tokens = generate_completion(
                model, prompt_ids, max_new_tokens=args.max_gen_tokens,
                temperature=args.temperature, device=device
            )

            completions.append((full_ids, ponder_depth, n_tokens))

            # Decode response
            response_ids = full_ids[0, prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Compute reward
            is_correct = check_answer(response_text, q["answer"])
            accuracy_reward = 1.0 if is_correct else -0.5
            ponder_reward = -args.ponder_penalty * ponder_depth
            reward = accuracy_reward + ponder_reward

            rewards.append(reward)
            if is_correct:
                step_correct += 1
            step_ponder += ponder_depth

            # Reference log probs (compute before update)
            with torch.no_grad():
                if n_tokens > 0:
                    ref_lp, _ = compute_log_probs(
                        model, full_ids, prompt_len, device
                    )
                    ref_log_probs.append(ref_lp)
                else:
                    ref_log_probs.append(None)

        # ── GRPO Update ──
        grpo_loss = grpo_step(
            model, optimizer, scaler,
            prompt_ids, completions, rewards, ref_log_probs,
            prompt_len, device, kl_coeff=args.kl_coeff
        )

        # Accumulate metrics
        total_correct += step_correct
        total_questions += args.group_size
        total_ponder += step_ponder / args.group_size
        total_reward += sum(rewards) / len(rewards)

        # ── Logging ──
        if step % args.log_interval == 0:
            elapsed = time.time() - step_start
            acc = total_correct / max(total_questions, 1) * 100
            avg_ponder = total_ponder / args.log_interval
            avg_reward = total_reward / args.log_interval

            print(
                f"Step {step:>5d} | "
                f"Acc {acc:>5.1f}% | "
                f"Reward {avg_reward:>+6.3f} | "
                f"Ponder {avg_ponder:>5.2f} | "
                f"Loss {grpo_loss:>7.4f} | "
                f"Q/s {args.log_interval / elapsed:.1f}"
            )

            # Reset
            total_correct = 0
            total_questions = 0
            total_ponder = 0.0
            total_reward = 0.0
            step_start = time.time()

        # ── Save ──
        if step % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"rl_step_{step}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "config": config,
            }, str(ckpt_path))
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = checkpoint_dir / "rl_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": args.max_steps,
        "config": config,
    }, str(final_path))
    print(f"\n{'=' * 60}")
    print(f"  GRPO Training Complete!")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
