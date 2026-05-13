"""
concept_lm/train.py

Training loop for the concept LM.

Joint training objective:
    L_total = L_NTP + w_aux * L_boundary + w_concept * L_concept

Where:
    L_NTP: standard next-token prediction cross-entropy
    L_boundary: Global Load Balancing loss for compression rate control
    L_concept: optional SAE-derived concept supervision (CoCoMix-style)

Training stability:
    Uses decoupled muP (DLCM Section 6.1) with separate learning rates
    for token-level and concept-level components.

References:
    DLCM (arXiv:2512.24617) - boundary loss, decoupled muP
    CoCoMix (arXiv:2502.08524) - concept supervision objective
    COCONUT (arXiv:2412.06769) - multi-stage curriculum training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import Optional, List
import math
import os

from .model import ConceptLM, ConceptLMConfig


@dataclass
class TrainConfig:
    # Optimization
    lr_token: float = 3e-4        # learning rate for encoder/decoder (token-level)
    lr_concept: float = 1e-4      # learning rate for concept transformer (backbone)
    lr_boundary: float = 3e-4     # learning rate for boundary detector
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    total_steps: int = 100000
    eval_every: int = 500
    save_every: int = 2000

    # Batch
    batch_size: int = 8
    grad_accum: int = 4
    max_seq_len: int = 2048

    # Loss weights
    aux_loss_weight: float = 0.01  # boundary load balancing
    concept_loss_weight: float = 0.1  # SAE concept supervision (set 0 to disable)

    # Output
    output_dir: str = "checkpoints"


def get_lr_scheduler(optimizer, cfg: TrainConfig):
    """Cosine decay with warmup."""
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer_decoupled_mup(model: ConceptLM, cfg: TrainConfig) -> optim.Optimizer:
    """
    Decoupled muP optimizer: separate learning rates for each component group.

    For heterogeneous architectures, each component requires independent
    learning rate scaling inversely proportional to its width:
        eta_token = lr_token / stoken
        eta_concept = lr_concept / sconcept

    Reference: DLCM Section 6.1, Equations 19-20 (arXiv:2512.24617)
    """
    token_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    concept_params = list(model.concept_transformer.parameters()) + list(model.smoother.parameters())
    boundary_params = list(model.boundary.parameters()) + list(model.pooler.parameters())

    param_groups = [
        {"params": token_params, "lr": cfg.lr_token, "name": "token_level"},
        {"params": concept_params, "lr": cfg.lr_concept, "name": "concept_level"},
        {"params": boundary_params, "lr": cfg.lr_boundary, "name": "boundary"},
    ]
    return optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


class TextDataset(Dataset):
    """Simple token-id dataset from pre-tokenized sequences."""
    def __init__(self, token_ids: List[List[int]], max_len: int):
        self.data = token_ids
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx][:self.max_len]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    max_len = max(x.shape[0] for x in batch)
    padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x
    return padded


def train(
    model: ConceptLM,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: str = "cuda"
):
    model = model.to(device)
    model.train()

    optimizer = build_optimizer_decoupled_mup(model, cfg)
    scheduler = get_lr_scheduler(optimizer, cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    global_step = 0
    accum_step = 0
    running_loss = 0.0
    running_aux = 0.0
    optimizer.zero_grad()

    for epoch in range(10000):  # breaks on total_steps
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.clone()

            logits, loss, aux_loss = model(batch, labels=labels)
            loss = loss / cfg.grad_accum
            loss.backward()

            running_loss += loss.item() * cfg.grad_accum
            running_aux += aux_loss.item()
            accum_step += 1

            if accum_step == cfg.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_step = 0
                global_step += 1

                if global_step % 100 == 0:
                    avg_loss = running_loss / 100
                    avg_aux = running_aux / 100
                    lr_token = optimizer.param_groups[0]['lr']
                    lr_concept = optimizer.param_groups[1]['lr']
                    print(f"step {global_step} | loss {avg_loss:.4f} | aux {avg_aux:.4f} "
                          f"| lr_tok {lr_token:.2e} | lr_con {lr_concept:.2e}")
                    running_loss = 0.0
                    running_aux = 0.0

                if global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")

                if global_step >= cfg.total_steps:
                    print("Training complete.")
                    return
