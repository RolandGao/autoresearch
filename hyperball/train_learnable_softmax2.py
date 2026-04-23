"""Run the SGDH ablation study for a fixed-scale learnable softmax.

The model is a single bias-free linear layer. Inputs, target weights, and
learned weights are row-normalized. The softmax scalar is fixed to 10, so the
only trainable parameter is the linear weight.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_SAMPLES = 30_000
BATCH_SIZES = (32, 128)
RUNS_PER_SETTING = 100
MOMENTUMS = (0.0, 0.5, 0.7, 0.8, 0.9)
SGDH_LR_SLOPE = 0.001409420528
SGDH_LR_INTERCEPT = 0.002388538963
LR_DECAY_RANGE = (3.5, 6.0)


@dataclass
class Config:
    seed: int = 0
    input_dim: int = 128
    num_classes: int = 4000
    train_size: int = NUM_SAMPLES
    softmax_scale: float = 10.0
    ground_truth_softmax_scale: float = 10.0
    prob_noise_std: float = 1e-4
    eval_batch_size: int = 512
    target_rmse: float = 1.19e-5
    log_every: int = 0
    device: str = "auto"


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--input-dim", type=int, default=Config.input_dim)
    parser.add_argument("--num-classes", type=int, default=Config.num_classes)
    parser.add_argument("--train-size", type=int, default=Config.train_size)
    parser.add_argument("--softmax-scale", type=float, default=Config.softmax_scale)
    parser.add_argument(
        "--ground-truth-softmax-scale",
        type=float,
        default=Config.ground_truth_softmax_scale,
    )
    parser.add_argument("--prob-noise-std", type=float, default=Config.prob_noise_std)
    parser.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--target-rmse", type=float, default=Config.target_rmse)
    parser.add_argument("--log-every", type=int, default=Config.log_every)
    parser.add_argument("--device", type=str, default=Config.device)
    return Config(**vars(parser.parse_args()))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def row_normalize_update(update: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    update_norm = update.norm(dim=1, keepdim=True)
    normalized = update / update_norm.clamp_min(eps)
    return torch.where(update_norm > 0, normalized, torch.zeros_like(normalized))


def project_rows(update: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return update - (update * weight).sum(dim=1, keepdim=True) * weight / denom


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def predicted_lr(batch_size: int) -> float:
    return SGDH_LR_SLOPE * batch_size + SGDH_LR_INTERCEPT


def rounded_steps_for_num_samples(num_samples: int, batch_size: int) -> int:
    return max(1, int(round(num_samples / batch_size)))


@torch.no_grad()
def build_problem(
    config: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = normalize_rows(torch.randn(config.train_size, config.input_dim, device=device))
    ground_truth_weight = normalize_rows(
        torch.randn(config.num_classes, config.input_dim, device=device)
    )

    clean_target_probs = torch.empty(
        config.train_size, config.num_classes, device=device
    )
    noisy_target_probs = torch.empty(
        config.train_size, config.num_classes, device=device
    )
    for start in range(0, config.train_size, config.eval_batch_size):
        end = min(start + config.eval_batch_size, config.train_size)
        clean_logits = config.ground_truth_softmax_scale * F.linear(
            x[start:end], ground_truth_weight
        )
        clean_probs = clean_logits.softmax(dim=1)
        clean_target_probs[start:end] = clean_probs
        noisy_target_probs[start:end] = make_noisy_probs(
            clean_probs, config.prob_noise_std
        )

    return x, clean_target_probs, noisy_target_probs


def make_noisy_probs(clean_probs: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0:
        return clean_probs
    noisy_probs = clean_probs + noise_std * torch.randn_like(clean_probs)
    noisy_probs = noisy_probs.clamp_min(0)
    row_sums = noisy_probs.sum(dim=1, keepdim=True)

    empty_rows = row_sums.squeeze(1) <= 0
    if empty_rows.any():
        noisy_probs[empty_rows] = clean_probs[empty_rows]
        row_sums = noisy_probs.sum(dim=1, keepdim=True)

    return noisy_probs / row_sums.clamp_min(1e-12)


def probability_rmse_loss(
    logits: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().mean().sqrt()


class FixedScaleSoftmax(nn.Module):
    def __init__(self, config: Config, initial_weight: torch.Tensor):
        super().__init__()
        expected_shape = (config.num_classes, config.input_dim)
        if tuple(initial_weight.shape) != expected_shape:
            raise ValueError(
                f"initial_weight has shape {tuple(initial_weight.shape)}, "
                f"expected {expected_shape}"
            )
        self.softmax_scale = float(config.softmax_scale)
        self.weight = nn.Parameter(initial_weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_scale * F.linear(x, self.weight).float()


class SGDH:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        momentum: float,
        g_projection: bool,
        g_norm: bool,
        nesterov: bool,
        m_projection: bool,
    ):
        self.param = param
        self.lr = lr
        self.momentum = momentum
        self.g_projection = g_projection
        self.g_norm = g_norm
        self.nesterov = nesterov
        self.m_projection = m_projection
        self.momentum_buffer = torch.zeros_like(param)

    def zero_grad(self) -> None:
        self.param.grad = None

    @torch.no_grad()
    def step(self, lr_factor: float) -> None:
        if self.param.grad is None:
            return

        grad = self.param.grad
        if self.g_projection:
            grad = project_rows(grad, self.param)
        if self.g_norm:
            grad = row_normalize_update(grad)

        self.momentum_buffer.mul_(self.momentum).add_(grad)
        if self.m_projection:
            self.momentum_buffer.copy_(project_rows(self.momentum_buffer, self.param))

        update = (
            grad + self.momentum * self.momentum_buffer
            if self.nesterov
            else self.momentum_buffer
        )
        self.param.add_(row_normalize_update(update), alpha=-self.lr * lr_factor)
        self.param.copy_(normalize_rows(self.param))


def lr_schedule_factor(step: int, steps: int, hparams: dict[str, Any]) -> float:
    progress = step / max(1, steps)
    return math.exp(-float(hparams["lr_decay"]) * progress)


def next_fixed_cycle_indices(
    x_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    sample_state: dict[str, Any],
) -> torch.Tensor:
    parts = []
    remaining = batch_size
    while remaining > 0:
        cursor = int(sample_state.get("cursor", 0))
        permutation = sample_state.get("permutation")
        if permutation is None:
            permutation = torch.randperm(x_size, device=device, generator=generator)
            sample_state["permutation"] = permutation
            cursor = 0
        elif cursor >= x_size:
            cursor = 0

        take = min(remaining, x_size - cursor)
        parts.append(permutation[cursor : cursor + take])
        cursor += take
        remaining -= take
        sample_state["cursor"] = cursor

    return torch.cat(parts) if len(parts) > 1 else parts[0]


@torch.no_grad()
def evaluate_clean_rmse(
    model: FixedScaleSoftmax,
    x: torch.Tensor,
    clean_target_probs: torch.Tensor,
    config: Config,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    num_values = 0

    for start in range(0, x.size(0), config.eval_batch_size):
        end = min(start + config.eval_batch_size, x.size(0))
        output_probs = model(x[start:end]).softmax(dim=1)
        diff = output_probs - clean_target_probs[start:end]
        squared_error_sum += diff.square().sum().item()
        num_values += diff.numel()

    model.train()
    return math.sqrt(squared_error_sum / num_values)


def setting_key(hparams: dict[str, Any]) -> str:
    return (
        f"g_projection={hparams['g_projection']},"
        f"g_norm={hparams['g_norm']},"
        f"nesterov={hparams['nesterov']},"
        f"m_projection={hparams['m_projection']},"
        f"batch_size={hparams['batch_size']}"
    )


def build_ablation_hparams(config: Config) -> list[dict[str, Any]]:
    rng = random.Random(config.seed + 1_000_003)
    candidate_specs: list[dict[str, Any]] = []

    for g_projection in (False, True):
        for g_norm in (False, True):
            for nesterov in (False, True):
                for m_projection in (False, True):
                    for batch_size in BATCH_SIZES:
                        steps = rounded_steps_for_num_samples(NUM_SAMPLES, batch_size)
                        lr_center = predicted_lr(batch_size)
                        for sample_idx in range(RUNS_PER_SETTING):
                            candidate_specs.append(
                                {
                                    "variant": "SGDH",
                                    "g_projection": g_projection,
                                    "g_norm": g_norm,
                                    "nesterov": nesterov,
                                    "m_projection": m_projection,
                                    "batch_size": batch_size,
                                    "steps": steps,
                                    "num_samples": NUM_SAMPLES,
                                    "sample_mode": "fixed_cycle",
                                    "lr_schedule": "exp_power",
                                    "lr_power": 1.0,
                                    "lr_decay": rng.uniform(*LR_DECAY_RANGE),
                                    "momentum": rng.choice(MOMENTUMS),
                                    "predicted_lr": lr_center,
                                    "lr": log_uniform(
                                        rng, lr_center / 10, lr_center * 10
                                    ),
                                    "sample_idx": sample_idx,
                                }
                            )

    return candidate_specs


def train_one_run(
    hparams: dict[str, Any],
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    clean_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    config: Config,
    device: torch.device,
    candidate_idx: int,
    num_candidates: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    model = FixedScaleSoftmax(config, initial_weight).to(device)
    optimizer = SGDH(
        model.weight,
        lr=float(hparams["lr"]),
        momentum=float(hparams["momentum"]),
        g_projection=bool(hparams["g_projection"]),
        g_norm=bool(hparams["g_norm"]),
        nesterov=bool(hparams["nesterov"]),
        m_projection=bool(hparams["m_projection"]),
    )
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(config.seed + 10_000)

    last_loss = float("nan")
    base_lr = float(hparams["lr"])
    steps = int(hparams["steps"])
    batch_size = int(hparams["batch_size"])
    sample_state: dict[str, Any] = {}
    training_start_time = time.time()
    for step in range(steps):
        lr_factor = lr_schedule_factor(step, steps, hparams)
        batch_idx = next_fixed_cycle_indices(
            x.size(0),
            batch_size,
            device=device,
            generator=batch_generator,
            sample_state=sample_state,
        )
        loss = probability_rmse_loss(model(x[batch_idx]), noisy_target_probs[batch_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lr_factor)

        last_loss = float(loss.detach().item())
        should_log = config.log_every > 0 and (
            (step + 1) % config.log_every == 0 or step + 1 == steps
        )
        if should_log:
            print(
                f"candidate {candidate_idx + 1:04d}/{num_candidates:04d} "
                f"SGDH step {step + 1:05d}/{steps:05d} "
                f"loss_rmse={last_loss:.8g} lr={base_lr * lr_factor:.6g}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed_sec = time.time() - training_start_time
    clean_rmse = evaluate_clean_rmse(model, x, clean_target_probs, config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time
    peak_allocated_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved_bytes = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    weight_norms = model.weight.detach().norm(dim=1)
    summary = {
        **hparams,
        "candidate_idx": candidate_idx,
        "actual_samples": steps * batch_size,
        "final_train_loss": last_loss,
        "clean_rmse": clean_rmse,
        "clean_train_rmse": clean_rmse,
        "target_met": clean_rmse <= config.target_rmse,
        "setting_key": setting_key(hparams),
        "weight_row_norm_mean": float(weight_norms.mean().item()),
        "weight_row_norm_std": float(weight_norms.std().item()),
        "duration_sec": elapsed_sec,
        "training_elapsed_sec": training_elapsed_sec,
        "elapsed_sec": elapsed_sec,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": peak_allocated_bytes / 1024**2,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_mib": peak_reserved_bytes / 1024**2,
    }
    print(f"RUN_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def main() -> None:
    config = parse_args()
    if config.softmax_scale != 10.0:
        raise ValueError("this experiment fixes --softmax-scale to 10.0")
    if config.ground_truth_softmax_scale != 10.0:
        raise ValueError("this experiment fixes --ground-truth-softmax-scale to 10.0")

    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)
    device = resolve_device(config.device)

    candidate_specs = build_ablation_hparams(config)
    print(
        "CONFIG "
        + json.dumps({**asdict(config), "device": str(device)}, sort_keys=True),
        flush=True,
    )
    print(
        "SEARCH_PLAN "
        + json.dumps(
            {
                "optimizer": "SGDH",
                "num_candidates": len(candidate_specs),
                "batch_sizes": BATCH_SIZES,
                "runs_per_setting": RUNS_PER_SETTING,
                "num_samples": NUM_SAMPLES,
                "momentums": MOMENTUMS,
                "lr_decay_range": LR_DECAY_RANGE,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    x, clean_target_probs, noisy_target_probs = build_problem(config, device)
    initial_weight = normalize_rows(
        torch.randn(config.num_classes, config.input_dim, device=device)
    )

    summaries: list[dict[str, Any]] = []
    best_by_setting: dict[str, dict[str, Any]] = {}
    for hparams in candidate_specs:
        candidate_idx = len(summaries)
        print(
            "RUN_START "
            + json.dumps(
                {
                    **hparams,
                    "candidate_idx": candidate_idx,
                    "num_candidates": len(candidate_specs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        summary = train_one_run(
            hparams,
            x,
            noisy_target_probs,
            clean_target_probs,
            initial_weight,
            config,
            device,
            candidate_idx,
            len(candidate_specs),
        )
        summaries.append(summary)
        key = str(summary["setting_key"])
        best = best_by_setting.get(key)
        if best is None or float(summary["clean_train_rmse"]) < float(
            best["clean_train_rmse"]
        ):
            best_by_setting[key] = summary

    summaries_by_rmse = sorted(
        summaries, key=lambda item: float(item["clean_train_rmse"])
    )
    print(f"BEST_BY_SETTING {json.dumps(best_by_setting, sort_keys=True)}", flush=True)
    print(f"FINAL_SUMMARY {json.dumps(summaries_by_rmse, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
