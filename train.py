"""
ESH-Loop Training Script
=========================
Train the ESH-Loop model on mixed-complexity data.

Usage:
    python train.py                           # Default config
    python train.py --max_steps 50000         # Longer training
    python train.py --max_ponder_steps 5      # More pondering
"""

import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from esh_loop import ESHLoopModel, ESHLoopConfig
from esh_loop.training import ESHLoopTrainer
from data_loader import create_data_loaders


def main():
    parser = argparse.ArgumentParser(description="Train ESH-Loop")
    parser.add_argument("--max_steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_ponder_steps", type=int, default=3)
    parser.add_argument("--ponder_cost_weight", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--expert_dim", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("ESH-Loop: Adaptive Compute via Entropy-Gated Pondering")
    print("=" * 60)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Config
    config = ESHLoopConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        max_seq_len=args.max_seq_len,
        max_ponder_steps=args.max_ponder_steps,
        ponder_cost_weight=args.ponder_cost_weight,
        dropout=0.1,
        layer_scale_init=1e-5,
    )

    # Model
    print(f"\nCreating ESH-Loop model...")
    print(f"  Layers: {config.n_layers}")
    print(f"  Max Ponder Steps: {config.max_ponder_steps}")
    print(f"  Ponder Cost Weight: {config.ponder_cost_weight}")
    model = ESHLoopModel(config)
    total_params = model.count_parameters()
    print(f"\nTotal Parameters: {total_params / 1e6:.2f}M")

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Data
    print("\nCreating mixed complexity data loaders...")
    train_loader, eval_loader = create_data_loaders(
        tokenizer, batch_size=args.batch_size,
        max_length=args.max_seq_len,
    )

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=args.lr,
            weight_decay=0.01, betas=(0.9, 0.95),
        )
        print("Using 8-bit PagedAdamW (saves ~3GB VRAM)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=0.01, betas=(0.9, 0.95),
        )
        print("Using standard AdamW")

    # Scheduler (cosine with warmup)
    actual_opt_steps = args.max_steps // args.grad_accum
    warmup_steps = min(200, actual_opt_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=actual_opt_steps,
        pct_start=warmup_steps / actual_opt_steps,
        anneal_strategy='cos',
    )

    # Training config summary
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accumulation: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {warmup_steps}")

    # Trainer
    trainer = ESHLoopTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=True,
        grad_accum_steps=args.grad_accum,
        scheduler=scheduler,
        log_interval=10,
    )

    print(f"\n{'=' * 60}")
    print("Starting ESH-Loop Training")
    print(f"{'=' * 60}")
    print(f"\nKey metric to watch: 'Ponder' (avg ponder steps per token)")
    print(f"Expected: Easy tokens → ~1.0, Hard tokens → ~{args.max_ponder_steps}.0\n")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train(max_steps=args.max_steps, save_every=args.save_every)

    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print("Next: Run generate.py to test adaptive pondering!")


if __name__ == "__main__":
    main()
