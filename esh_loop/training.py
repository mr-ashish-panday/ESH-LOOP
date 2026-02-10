"""
ESH-Loop Training Pipeline
============================
Handles training with ponder cost regularization,
gradient accumulation, mixed precision, and logging.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path


class ESHLoopTrainer:
    """Trainer for ESH-Loop model with adaptive compute tracking."""

    def __init__(self, model, optimizer, train_loader, eval_loader=None,
                 config=None, device="cuda", checkpoint_dir="./checkpoints",
                 use_amp=True, grad_accum_steps=8, max_grad_norm=1.0,
                 scheduler=None, log_interval=10):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval

        self.scaler = GradScaler() if self.use_amp else None
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch):
        """Single training step with ponder cost tracking."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)

        with autocast(enabled=self.use_amp):
            outputs = self.model(input_ids, labels=input_ids,
                               return_routing_stats=True)
            loss = outputs["loss"]
            aux_loss = outputs["aux_loss"]
            total_loss = loss + aux_loss

            # Scale for gradient accumulation
            total_loss = total_loss / self.grad_accum_steps

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Collect routing + ponder stats
        routing_stats = {}
        if outputs["routing_stats"]:
            alpha_means = [s["alpha_mean"] for s in outputs["routing_stats"]]
            attn_ratios = [s["attention_ratio"] for s in outputs["routing_stats"]]
            ponder_steps = [s["avg_ponder_steps"] for s in outputs["routing_stats"]]

            routing_stats = {
                "alpha_mean": sum(alpha_means) / len(alpha_means),
                "attention_ratio": sum(attn_ratios) / len(attn_ratios),
                "avg_ponder_steps": sum(ponder_steps) / len(ponder_steps),
            }

        return {
            "loss": loss.item(),
            "aux_loss": aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
            "ppl": math.exp(min(loss.item(), 20)),
            "ponder_cost": outputs["ponder_cost"] if isinstance(outputs["ponder_cost"], float) else outputs["ponder_cost"].item() if torch.is_tensor(outputs["ponder_cost"]) else float(outputs["ponder_cost"]),
            "avg_ponder_steps": outputs["avg_ponder_steps"] if isinstance(outputs["avg_ponder_steps"], float) else float(outputs["avg_ponder_steps"]),
            **routing_stats,
        }

    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

    def train(self, max_steps=25000, save_every=5000):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters() / 1e6:.2f}M")
        print(f"Effective batch size: {self.grad_accum_steps * next(iter(self.train_loader))['input_ids'].shape[0]}")

        self.model.to(self.device)
        data_iter = iter(self.train_loader)
        accum_metrics = {}
        start_time = time.time()
        tokens_processed = 0

        while self.global_step < max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Train step
            metrics = self.train_step(batch)
            tokens_processed += batch["input_ids"].numel()

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in accum_metrics:
                    accum_metrics[k] = 0.0
                accum_metrics[k] += v

            # Optimizer step every grad_accum_steps
            if (self.global_step + 1) % self.grad_accum_steps == 0:
                self.optimizer_step()

            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / elapsed

                avg_metrics = {k: v / self.log_interval
                             for k, v in accum_metrics.items()}

                lr = self.scheduler.get_last_lr()[0] if self.scheduler else \
                     self.optimizer.param_groups[0]['lr']

                print(
                    f"Step {self.global_step:>6d} | "
                    f"Loss {avg_metrics.get('loss', 0):.4f} | "
                    f"PPL {avg_metrics.get('ppl', 0):.2f} | "
                    f"Aux {avg_metrics.get('aux_loss', 0):.4f} | "
                    f"Attn% {avg_metrics.get('attention_ratio', 0) * 100:.1f} | "
                    f"Ponder {avg_metrics.get('avg_ponder_steps', 0):.2f} | "
                    f"LR {lr:.2e} | "
                    f"Tok/s {tok_per_sec:.0f}"
                )

                accum_metrics = {}
                tokens_processed = 0
                start_time = time.time()

            # Save checkpoints
            if self.global_step % save_every == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        print(f"\nTraining complete!")

    def save_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config,
        }, str(path))
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("step", 0)
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
