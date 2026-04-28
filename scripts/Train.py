"""
Train.py — MM-MARAS training loop

Features:
    - Multi-GPU training via DDP (torchrun) — auto-detected, no flag needed
    - Single-GPU fallback when launched with plain python
    - Mixed precision training (torch.cuda.amp) — halves VRAM usage
    - Train / val loop with per-epoch metrics
    - Curriculum scheduling (forecast + ERI ramp over first 20% of steps)
    - AdamW optimiser + cosine LR schedule with linear warmup
    - Gradient clipping (max norm 1.0)
    - Checkpoint saving: best val loss + periodic every N epochs
    - TensorBoard logging on rank 0 only
    - Resume from checkpoint
    - DistributedSampler with per-epoch shuffle for DDP correctness
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "model" / "encoders"))
sys.path.insert(0, str(_REPO_ROOT / "model"))
sys.path.insert(0, str(_REPO_ROOT / "data-preprocessing-pipeline"))

from augment import augment_batch
from dataset import build_dataloaders, MARASSDataset
from loss import MARASSLoss, LossWeights
from model import MARASSModel, ModelConfig

log = logging.getLogger(__name__)


# ======================================================================
# DDP helpers
# ======================================================================

def is_ddp_run() -> bool:
    return "LOCAL_RANK" in os.environ

def ddp_setup() -> tuple[int, int, torch.device]:
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, world_size, device

def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_sum_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def reduce_max_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor

def unwrap_model(model: nn.Module) -> MARASSModel:
    return model.module if isinstance(model, DDP) else model


# ======================================================================
# [v3.5] Exponential moving average of model weights
# ======================================================================

class ModelEMA:
    """
    Maintains an exponentially-decayed shadow copy of model parameters.

    Update rule (per optimizer step):
        shadow = decay * shadow + (1 - decay) * param

    `apply_to(model)` is a context manager that temporarily swaps the model's
    parameters with the shadow copy, for running validation on EMA weights.
    Only `parameters()` are shadowed; buffers (BN running stats, etc.) are
    left on the model — the ConvLSTM / Swin stack here has no running stats
    that would drift, so buffer-shadowing would be overhead with no gain.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        src = unwrap_model(model)
        for name, p in src.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = unwrap_model(model)
        d = self.decay
        for name, p in src.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow[name]
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module):
        src = unwrap_model(model)
        backup: dict[str, torch.Tensor] = {}
        try:
            for name, p in src.named_parameters():
                if name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.data.copy_(self.shadow[name])
            yield
        finally:
            for name, p in src.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        for k, v in sd["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: MARASSDataset, rank: int, world_size: int) -> None:
        self.start = (len(dataset) * rank) // world_size
        self.end = (len(dataset) * (rank + 1)) // world_size

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start


# ======================================================================
# DataLoader builders
# ======================================================================

def _build_ddp_loaders(
    patch_dir: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    bloom_oversample: int = 1,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MARASSDataset(patch_dir=patch_dir, split=split,
                                bloom_oversample=bloom_oversample)
        if split == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else eval_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    return loaders


# ======================================================================
# Args
# ======================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MM-MARAS")
    p.add_argument("--patch-dir",    default="data/patches",  help="Root patches directory")
    p.add_argument("--ckpt-dir",     default="checkpoints",   help="Checkpoint output directory")
    p.add_argument("--log-dir",      default="runs",          help="TensorBoard log directory")
    p.add_argument("--resume",       default=None,            help="Path to checkpoint to resume from")
    p.add_argument("--epochs",        type=int,   default=60)  # [v3.5] 50→60: post-curriculum refinement room
    p.add_argument("--ema-decay",     type=float, default=0.999,
                   help="[v3.5] EMA decay; set <=0 to disable")
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=5e-5)   # [v3.4] 3e-5→5e-5: v3.3 too slow for secondary tasks
    p.add_argument("--weight-decay",  type=float, default=2e-2)  # [v3.4] 5e-2→2e-2: v3.3 over-regularized with dropout
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int,   default=5)
    p.add_argument("--save-every",    type=int,   default=10)
    p.add_argument("--no-amp",        action="store_true")
    p.add_argument("--w-recon",    type=float, default=1.0)
    p.add_argument("--w-forecast", type=float, default=0.5)
    p.add_argument("--w-eri",      type=float, default=0.3)
    p.add_argument("--w-aux",      type=float, default=0.05)  # [v3.5] 0.01→0.05: force expert specialization
    # [v3.6] MoE aux annealing — push experts apart early, then back off so
    # they can specialise. v3.5 ended at uniform 0.2500 routing weights;
    # decaying w_aux after the load-balancing job is done is the cleanest fix.
    p.add_argument("--w-aux-final", type=float, default=0.005,
                   help="[v3.6] Anneal target for MoE aux weight (default 0.005). "
                        "Set equal to --w-aux to disable annealing.")
    p.add_argument("--w-aux-anneal-start-epoch", type=int, default=5,
                   help="[v3.6] Epoch at which w_aux annealing starts (default 5, "
                        "matches --warmup-epochs).")
    p.add_argument("--w-aux-anneal-end-epoch", type=int, default=18,
                   help="[v3.6] Epoch at which w_aux reaches --w-aux-final (default 18).")
    p.add_argument("--w-holdout",  type=float, default=0.8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--bloom-oversample", type=int, default=5,
                   help="[v3] Duplicate bloom patches N× in training set "
                        "(v3.5 default 3, [v3.6] 5 for stronger class-1 supervision).")
    p.add_argument("--eri-class1-weight", type=float, default=12.0,
                   help="[v3.6] ERI class-1 (low-bloom) weight in ordinal CE. "
                        "v3.5 used 10.0; default 12.0 to push class-1 F1 (was 0.634). "
                        "Other classes stay at [0.1, ?, 12, 6, 6].")
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def w_aux_for_epoch(
    epoch: int,
    w_aux_start: float,
    w_aux_final: float,
    anneal_start_epoch: int,
    anneal_end_epoch: int,
) -> float:
    """
    [v3.6] Linearly anneal MoE aux loss weight from `w_aux_start` to
    `w_aux_final` between `anneal_start_epoch` (inclusive) and
    `anneal_end_epoch` (inclusive).

    Held flat at `w_aux_start` before the start epoch, and at
    `w_aux_final` from the end epoch onward. If the two endpoints are
    equal (or the window is degenerate), annealing is a no-op.
    """
    if w_aux_start == w_aux_final:
        return w_aux_start
    if anneal_end_epoch <= anneal_start_epoch:
        # Degenerate window — snap to final once past start.
        return w_aux_final if epoch >= anneal_end_epoch else w_aux_start
    if epoch <= anneal_start_epoch:
        return w_aux_start
    if epoch >= anneal_end_epoch:
        return w_aux_final
    span = anneal_end_epoch - anneal_start_epoch
    progress = (epoch - anneal_start_epoch) / span
    return w_aux_start + (w_aux_final - w_aux_start) * progress


def routing_entropy(routing_weights: torch.Tensor) -> float:
    mean_w = routing_weights if routing_weights.ndim == 1 else routing_weights.mean(dim=0)
    return -(mean_w * (mean_w + 1e-8).log()).sum().item()

def stable_holdout_mask(
    obs_mask: torch.Tensor,
    land_mask: torch.Tensor,
    holdout_frac: float,
) -> torch.Tensor:
    if holdout_frac <= 0:
        return torch.zeros_like(obs_mask)

    obs_cpu = obs_mask.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha1(obs_cpu.numpy().tobytes()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rand = torch.rand(obs_mask.shape, generator=generator).to(obs_mask.device)
    ocean = 1.0 - land_mask
    return ((obs_mask > 0.5) & (ocean > 0.5) & (rand < holdout_frac)).float()

def compute_masked_rmse_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    valid = mask.bool()
    if not valid.any():
        return 0.0, 0.0
    diff = pred[valid].float() - target[valid].float()
    return diff.pow(2).sum().item(), float(valid.sum().item())

def build_gap_eval_batch(
    batch: dict[str, torch.Tensor],
    holdout_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    eval_batch = dict(batch)
    eval_batch["chl_obs"] = batch["chl_obs"].clone()
    eval_batch["obs_mask"] = batch["obs_mask"].clone()
    eval_batch["chl_obs"][:, -1] *= (1.0 - holdout_mask)
    eval_batch["obs_mask"][:, -1] *= (1.0 - holdout_mask)
    return eval_batch


# ======================================================================
# One epoch
# ======================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MARASSLoss,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    scaler: GradScaler | None,
    device: torch.device,
    global_step: int,
    total_steps: int,
    grad_clip: float,
    writer: SummaryWriter | None,
    phase: str,
    use_amp: bool,
    world_size: int = 1,
    is_main: bool = True,
    ema: ModelEMA | None = None,
) -> tuple[dict[str, float], int]:
    is_train = (phase == "train")
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = use_amp and (device.type == "cuda")

    model.train(is_train)

    metric_keys = ("aux", "bloom_fcast", "curriculum_scale", "eri", "forecast", "holdout", "recon", "total")
    totals = {k: 0.0 for k in metric_keys}
    model_cfg = unwrap_model(model).cfg
    n_examples = 0.0
    gap_sse = 0.0
    gap_count = 0.0
    routing_weight_sum = torch.zeros(model_cfg.n_experts, dtype=torch.float64, device=device)
    routing_count = 0.0
    n_skipped = 0.0
    t0 = time.time()

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch_size = batch["chl_obs"].shape[0]

            if is_train:
                batch = augment_batch(batch)
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=amp_device, enabled=amp_enabled):
                outputs = model(batch)
                loss, breakdown = criterion(
                    outputs, batch,
                    step=global_step if is_train else None,
                    total_steps=total_steps,
                )

            if is_train:
                is_valid_loss = torch.tensor(1.0 if torch.isfinite(loss) else 0.0, device=device)
                is_valid_loss = reduce_sum_tensor(is_valid_loss, world_size)
                
                if is_valid_loss.item() < world_size:
                    optimizer.zero_grad(set_to_none=True)
                    n_skipped += 1.0
                    del outputs, loss, breakdown
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                stepped = False
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    ).item()
                    
                    is_valid_grad = torch.tensor(1.0 if math.isfinite(grad_norm) else 0.0, device=device)
                    is_valid_grad = reduce_sum_tensor(is_valid_grad, world_size)
                    
                    if is_valid_grad.item() < world_size:
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        n_skipped += 1.0
                        del outputs, loss, breakdown
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                        stepped = True
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    ).item()
                    
                    is_valid_grad = torch.tensor(1.0 if math.isfinite(grad_norm) else 0.0, device=device)
                    is_valid_grad = reduce_sum_tensor(is_valid_grad, world_size)
                    
                    if is_valid_grad.item() < world_size:
                        optimizer.zero_grad(set_to_none=True)
                        n_skipped += 1.0
                        del outputs, loss, breakdown
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    optimizer.step()
                    stepped = True

                if stepped:
                    scheduler.step()
                    global_step += 1
                    if ema is not None:
                        ema.update(model)

                if writer and is_main:
                    writer.add_scalar("train/loss",             breakdown["total"],    global_step)
                    writer.add_scalar("train/recon",            breakdown["recon"],    global_step)
                    writer.add_scalar("train/forecast",         breakdown["forecast"], global_step)
                    writer.add_scalar("train/eri",              breakdown["eri"],      global_step)
                    writer.add_scalar("train/aux",              breakdown["aux"],      global_step)
                    writer.add_scalar("train/grad_norm",        grad_norm,             global_step)
                    writer.add_scalar("train/lr",               scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/curriculum_scale", breakdown["curriculum_scale"], global_step)
                    writer.add_scalar("train/holdout",           breakdown.get("holdout", 0.0),  global_step)
                    writer.add_scalar("train/bloom_fcast",       breakdown.get("bloom_fcast", 0.0), global_step)
                    writer.add_scalar("train/skipped_batches",   n_skipped,             global_step)
                    if amp_enabled and scaler is not None:
                        writer.add_scalar("train/amp_scale", scaler.get_scale(), global_step)

            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + (v * batch_size)
            n_examples += batch_size

            with torch.no_grad():
                last_chl = batch["chl_obs"][:, -1]
                pred_recon = outputs["recon"].squeeze(1).float()

                if is_train and "holdout_mask" in outputs:
                    sse, count = compute_masked_rmse_stats(
                        pred_recon, last_chl, outputs["holdout_mask"]
                    )
                    gap_sse += sse
                    gap_count += count
                elif not is_train:
                    holdout_mask = stable_holdout_mask(
                        batch["obs_mask"][:, -1],
                        batch["land_mask"],
                        model_cfg.holdout_frac,
                    )
                    if holdout_mask.any():
                        eval_batch = build_gap_eval_batch(batch, holdout_mask)
                        try:
                            with autocast(device_type=amp_device, enabled=amp_enabled):
                                gap_outputs = model(eval_batch)
                            gap_pred = gap_outputs["recon"].squeeze(1).float()
                            sse, count = compute_masked_rmse_stats(gap_pred, last_chl, holdout_mask)
                            gap_sse += sse
                            gap_count += count
                            del gap_outputs
                        except RuntimeError as e:
                            if "out of memory" not in str(e):
                                raise
                            # OOM during gap eval — skip this batch's gap metric
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        del eval_batch

            if "routing_weights" in outputs:
                batch_routing_sum = outputs["routing_weights"].detach().sum(dim=0, dtype=torch.float64)
                routing_weight_sum += batch_routing_sum
                routing_count += batch_size

    elapsed = time.time() - t0
    keys = list(metric_keys)
    totals_tensor = torch.tensor([totals[k] for k in keys], dtype=torch.float64, device=device)
    totals_tensor = reduce_sum_tensor(totals_tensor, world_size)

    counts_tensor = torch.tensor(
        [n_examples, gap_sse, gap_count, routing_count, n_skipped],
        dtype=torch.float64,
        device=device,
    )
    counts_tensor = reduce_sum_tensor(counts_tensor, world_size)
    n_examples, gap_sse, gap_count, routing_count, n_skipped = counts_tensor.tolist()

    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
    elapsed_tensor = reduce_max_tensor(elapsed_tensor, world_size)

    metrics = {k: totals_tensor[i].item() / max(n_examples, 1.0) for i, k in enumerate(keys)}
    metrics["epoch_time_s"] = elapsed_tensor.item()
    metrics["gap_rmse"] = math.sqrt(gap_sse / gap_count) if gap_count > 0 else float("nan")
    metrics["skipped_batches"] = n_skipped

    routing_weight_sum = reduce_sum_tensor(routing_weight_sum, world_size)
    if routing_count > 0:
        metrics["routing_entropy"] = routing_entropy((routing_weight_sum / routing_count).float())

    return metrics, global_step


# ======================================================================
# Checkpoint helpers
# ======================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    epoch: int,
    global_step: int,
    val_loss: float,
    ema: ModelEMA | None = None,
) -> None:
    ckpt = {
        "epoch":       epoch,
        "global_step": global_step,
        "val_loss":    val_loss,
        "model":       unwrap_model(model).state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)
    log.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    device: torch.device,
    ema: ModelEMA | None = None,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device)
    result = unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
    if result.missing_keys:
        log.warning(f"Missing keys in checkpoint (random init): {result.missing_keys}")
    if result.unexpected_keys:
        log.warning(f"Unexpected keys in checkpoint (ignored): {result.unexpected_keys}")
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    elif ema is not None:
        log.warning("Resume checkpoint has no EMA state; EMA shadow initialized from model weights.")
    log.info(
        f"Resumed from {path} "
        f"(epoch {ckpt['epoch']}, step {ckpt['global_step']}, "
        f"val_loss {ckpt['val_loss']:.4f})"
    )
    return ckpt["epoch"] + 1, ckpt["global_step"], ckpt["val_loss"]


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = get_args()

    using_ddp = is_ddp_run()
    eval_batch_size = args.eval_batch_size or (max(1, args.batch_size // 2) if using_ddp else args.batch_size)

    if using_ddp:
        local_rank, world_size, device = ddp_setup()
        is_main = (local_rank == 0)
    else:
        local_rank  = 0
        world_size  = 1
        is_main     = True
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    use_amp = not args.no_amp and device.type == "cuda"

    if is_main:
        mode_str = f"DDP world_size={world_size}" if using_ddp else "single-GPU"
        log.info(f"Device: {device}  |  Mode: {mode_str}  |  AMP: {'enabled' if use_amp else 'disabled'}")

    ckpt_dir = Path(args.ckpt_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if using_ddp:
        dist.barrier()

    if using_ddp:
        loaders = _build_ddp_loaders(
            patch_dir=args.patch_dir,
            batch_size=args.batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=args.num_workers,
            rank=local_rank,
            world_size=world_size,
            bloom_oversample=args.bloom_oversample,
        )
        steps_per_epoch = len(loaders["train"])
        total_steps  = steps_per_epoch * args.epochs
        warmup_steps = steps_per_epoch * args.warmup_epochs
    else:
        loaders = build_dataloaders(
            patch_dir=args.patch_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            bloom_oversample=args.bloom_oversample,
        )
        steps_per_epoch = len(loaders["train"])
        total_steps     = steps_per_epoch * args.epochs
        warmup_steps    = steps_per_epoch * args.warmup_epochs

    if is_main:
        log.info(
            f"Data: train batch={args.batch_size}, eval batch={eval_batch_size}; "
            f"{steps_per_epoch} train batches/rank/epoch x {world_size} rank(s); "
            f"scheduler sees {steps_per_epoch} steps/epoch x {args.epochs} epochs "
            f"= {total_steps} steps  ({warmup_steps} warmup)"
        )

    cfg   = ModelConfig()
    model = MARASSModel(cfg).to(device)

    if using_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "bias" not in n and "norm" not in n and "bn" not in n
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ("bias" in n or "norm" in n or "bn" in n)
    ]
    optimizer = AdamW([
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    # [v3.5] EMA of model weights. Updated once per optimizer step inside
    # run_epoch; val runs a second forward pass with EMA weights swapped in.
    # EMA typically damps single-epoch noise that was causing the selector to
    # lock onto early checkpoints (v3.4 best ckpt at epoch 9/50).
    ema: ModelEMA | None = None
    if args.ema_decay > 0.0:
        ema = ModelEMA(model, decay=args.ema_decay)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    # [v3.1] Conservative scaler: growth_interval=100000 exceeds total training
    # steps (14100), so scale stays at 2^13=8192 for the entire run.
    # With log_var floor=-3 (20x gradient amplifier) and scale=8192,
    # max FP16 gradient = 8192*20.1*residual = 164k*residual.
    # Overflow requires residual > 0.40 (vs 0.073 with old floor=-4, scale=16k).
    scaler = GradScaler(
        device="cuda", init_scale=2**13, growth_interval=100_000,
    ) if use_amp else None

    # [v3.6] ERI class-1 weight bumped via CLI (default 12.0). Other entries
    # follow the v3.5 ratio [0.1, _, 12, 6, 6]; we just splice in arg value.
    eri_class_weights = (0.1, float(args.eri_class1_weight), 12.0, 6.0, 6.0)

    criterion = MARASSLoss(
        weights=LossWeights(
            recon=args.w_recon,
            forecast=args.w_forecast,
            eri=args.w_eri,
            bloom_fcast=0.3,
            aux=args.w_aux,
            holdout=args.w_holdout,
        ),
        bloom_threshold=2.5,
        eri_class_weights=eri_class_weights,
    ).to(device)

    if is_main:
        log.info(
            f"[v3.6] ERI class weights: {eri_class_weights}  "
            f"(class-1 from --eri-class1-weight={args.eri_class1_weight})"
        )
        if args.w_aux_final != args.w_aux:
            log.info(
                f"[v3.6] MoE aux annealing: w_aux {args.w_aux} → {args.w_aux_final} "
                f"linearly across epochs "
                f"[{args.w_aux_anneal_start_epoch}, {args.w_aux_anneal_end_epoch}]"
            )
        else:
            log.info(
                f"[v3.6] MoE aux annealing disabled (--w-aux == --w-aux-final = "
                f"{args.w_aux})"
            )

    start_epoch       = 0
    global_step       = 0
    best_val_loss     = float("inf")
    best_ema_val_loss = float("inf")
    nan_val_streak    = 0
    MAX_NAN_EPOCHS    = 3   # [v3.1] stop training after 3 consecutive NaN val epochs
    # [v3.3] Early stopping — model peaked at epoch 5/50 last run (severe overfitting)
    no_improve_count  = 0
    EARLY_STOP_PATIENCE = 12  # [v3.4] 10→12: give slower curriculum more room

    if args.resume:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler, device, ema=ema,
        )

    writer = SummaryWriter(log_dir=args.log_dir) if is_main else None

    if is_main and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved  = torch.cuda.memory_reserved(device)  / 1024**3
        log.info(f"VRAM at start: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    if is_main:
        log.info(f"Starting training: epochs {start_epoch}-{args.epochs - 1}")

    for epoch in range(start_epoch, args.epochs):
        if using_ddp:
            loaders["train"].sampler.set_epoch(epoch)

        # [v3.6] Apply MoE aux weight anneal for this epoch.
        # criterion is plain nn.Module (not DDP-wrapped), so .w is accessible.
        current_w_aux = w_aux_for_epoch(
            epoch=epoch,
            w_aux_start=args.w_aux,
            w_aux_final=args.w_aux_final,
            anneal_start_epoch=args.w_aux_anneal_start_epoch,
            anneal_end_epoch=args.w_aux_anneal_end_epoch,
        )
        criterion.w.aux = current_w_aux
        if is_main:
            if writer is not None:
                writer.add_scalar("train_epoch/w_aux", current_w_aux, epoch)
            log.info(f"Epoch {epoch:03d} | w_aux = {current_w_aux:.5f}")

        train_metrics, global_step = run_epoch(
            model=model, loader=loaders["train"],
            criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            device=device, global_step=global_step,
            total_steps=total_steps, grad_clip=args.grad_clip,
            writer=writer, phase="train", use_amp=use_amp,
            world_size=world_size, is_main=is_main,
            ema=ema,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        val_metrics, _ = run_epoch(
            model=model, loader=loaders["val"],
            criterion=criterion, optimizer=None,
            scheduler=None, scaler=None,
            device=device, global_step=global_step,
            total_steps=total_steps, grad_clip=args.grad_clip,
            writer=None, phase="val", use_amp=use_amp,
            world_size=world_size, is_main=is_main,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # [v3.5] Second val pass with EMA weights swapped in. Cheap compared
        # to training (val loader is ~20% of train), and gives a smoother
        # selection signal than the raw weights at the current step.
        ema_val_metrics: dict[str, float] | None = None
        if ema is not None:
            with ema.apply_to(model):
                ema_val_metrics, _ = run_epoch(
                    model=model, loader=loaders["val"],
                    criterion=criterion, optimizer=None,
                    scheduler=None, scaler=None,
                    device=device, global_step=global_step,
                    total_steps=total_steps, grad_clip=args.grad_clip,
                    writer=None, phase="val", use_amp=use_amp,
                    world_size=world_size, is_main=is_main,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        val_loss = val_metrics["total"]

        if is_main:
            if writer:
                # <--- FIXED: using .get() prevents KeyErrors if batches were skipped
                writer.add_scalar("val/loss",       val_metrics["total"],      epoch)
                writer.add_scalar("val/recon",      val_metrics["recon"],      epoch)
                writer.add_scalar("val/forecast",   val_metrics["forecast"],   epoch)
                writer.add_scalar("val/eri",        val_metrics["eri"],        epoch)
                writer.add_scalar("val/bloom_fcast", val_metrics.get("bloom_fcast", 0.0), epoch)
                writer.add_scalar("train_epoch/routing_entropy", train_metrics.get("routing_entropy", 0.0), epoch)
                writer.add_scalar("train_epoch/gap_rmse",        train_metrics.get("gap_rmse", 0.0),        epoch)
                writer.add_scalar("val/gap_rmse",                val_metrics.get("gap_rmse", 0.0),          epoch)
                if ema_val_metrics is not None:
                    writer.add_scalar("val_ema/loss",     ema_val_metrics["total"],    epoch)
                    writer.add_scalar("val_ema/recon",    ema_val_metrics["recon"],    epoch)
                    writer.add_scalar("val_ema/forecast", ema_val_metrics["forecast"], epoch)
                    writer.add_scalar("val_ema/eri",      ema_val_metrics["eri"],      epoch)
                    writer.add_scalar("val_ema/gap_rmse", ema_val_metrics.get("gap_rmse", 0.0), epoch)

            vram_str = ""
            if device.type == "cuda":
                vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                torch.cuda.reset_peak_memory_stats(device)
                vram_str = f"  VRAM {vram_gb:.1f}GB"

            ema_str = (
                f" | ema {ema_val_metrics['total']:.4f} gap {ema_val_metrics.get('gap_rmse', float('nan')):.4f}"
                if ema_val_metrics is not None else ""
            )
            log.info(
                f"Epoch {epoch:03d} | "
                f"train {train_metrics['total']:.4f} "
                f"(R {train_metrics['recon']:.4f} "
                f"F {train_metrics['forecast']:.4f} "
                f"E {train_metrics['eri']:.4f} "
                f"B {train_metrics.get('bloom_fcast', 0.0):.4f}) | "
                f"val {val_loss:.4f} gap_rmse {val_metrics.get('gap_rmse', float('nan')):.4f}"
                f"{ema_str} "
                f"skipped {int(train_metrics.get('skipped_batches', 0.0))} | "
                f"{train_metrics['epoch_time_s']:.0f}s{vram_str}"
            )

            if not math.isfinite(val_loss):
                log.warning("Validation loss is non-finite; skipping checkpoint save for this epoch.")
                nan_val_streak += 1
                if nan_val_streak >= MAX_NAN_EPOCHS:
                    log.error(
                        f"Stopping: {nan_val_streak} consecutive NaN val epochs. "
                        f"Best checkpoint: {ckpt_dir / 'best.pt'} (val {best_val_loss:.4f})"
                    )
            else:
                nan_val_streak = 0
                raw_improved = val_loss < best_val_loss
                ema_improved = (
                    ema_val_metrics is not None
                    and math.isfinite(ema_val_metrics["total"])
                    and ema_val_metrics["total"] < best_ema_val_loss
                )

                if raw_improved:
                    best_val_loss = val_loss
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, optimizer,
                        scheduler, scaler, epoch, global_step, best_val_loss,
                    )
                    log.info(f"  -> New best val loss: {best_val_loss:.4f}")

                if ema_improved:
                    best_ema_val_loss = ema_val_metrics["total"]
                    # [v3.5] Save with EMA weights swapped in — the checkpoint
                    # file contains the EMA shadow as the model state_dict.
                    with ema.apply_to(model):
                        save_checkpoint(
                            ckpt_dir / "best_ema.pt", model, optimizer,
                            scheduler, scaler, epoch, global_step, best_ema_val_loss,
                        )
                    log.info(f"  -> New best EMA val loss: {best_ema_val_loss:.4f}")

                # [v3.5] Early stopping resets if EITHER the raw or EMA val
                # improves. This prevents stopping while EMA is still winning.
                if raw_improved or ema_improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= EARLY_STOP_PATIENCE:
                        log.info(
                            f"  Early stopping: no improvement for {EARLY_STOP_PATIENCE} epochs. "
                            f"Best raw: {best_val_loss:.4f}  Best EMA: {best_ema_val_loss:.4f}"
                        )

            if math.isfinite(val_loss):
                save_checkpoint(
                    ckpt_dir / "last.pt", model, optimizer,
                    scheduler, scaler, epoch, global_step, val_loss, ema=ema,
                )

            if math.isfinite(val_loss) and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                    scheduler, scaler, epoch, global_step, val_loss, ema=ema,
                )

        # [v3.1] Broadcast early-stop decision to all ranks
        # [v3.3] Also triggers on patience-based early stopping
        should_stop_flag = (
            nan_val_streak >= MAX_NAN_EPOCHS
            or no_improve_count >= EARLY_STOP_PATIENCE
        )
        if using_ddp:
            should_stop = torch.tensor(
                1.0 if (is_main and should_stop_flag) else 0.0,
                device=device,
            )
            dist.broadcast(should_stop, src=0)
            dist.barrier()
            if should_stop.item() > 0.5:
                break
        elif should_stop_flag:
            break

    if writer:
        writer.close()

    ddp_cleanup()

    if is_main:
        log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        log.info(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
        if ema is not None:
            log.info(
                f"Best EMA val loss: {best_ema_val_loss:.4f}  "
                f"checkpoint: {ckpt_dir / 'best_ema.pt'}"
            )


if __name__ == "__main__":
    main()