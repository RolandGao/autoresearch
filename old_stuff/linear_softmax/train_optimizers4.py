"""Run optimizer hyperparameter search for the fixed-scale softmax problem."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch._dynamo
import torch.nn.functional as F


torch._dynamo.config.recompile_limit = 128

DATASET_SIZE = 30_000
NUM_SAMPLE_OPTIONS = (512, 2048, 8192, 32768)
OPTIMIZERS = ("AdamW", "AdamH", "SGD", "SGD2")
H_OPTIMIZERS = ("AdamH", "SGD2")
H_NORMS = ("matrix", "row")
OPTIMIZER_VARIANTS = tuple(
    optimizer if optimizer not in H_OPTIMIZERS else f"{optimizer}_{h_norm}"
    for optimizer in OPTIMIZERS
    for h_norm in (H_NORMS if optimizer in H_OPTIMIZERS else (None,))
)
BATCH_SIZES = (8, 64, 512)
RUNS_PER_SETTING = 100
BETA1_MOMENTUM_VALUES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)
ADAM_BETA2S = (0.9, 0.95, 0.99, 0.999)
ADAM_EPS = 1e-10
SGD2_BETA2S = (0.0, 0.5, 0.8, 0.9, 0.99)
LR_POWER = 1.0
SCALAR_ADAMW_BETA1 = 0.9
SCALAR_ADAMW_BETA2 = 0.999
SCALAR_ADAMW_EPS = 1e-10
SCALAR_ADAMW_WD = 0.0
_LR_MEDIANS = {
    ("AdamW", None, 512, 8): 1.7,
    ("AdamW", None, 512, 64): 2.2,
    ("AdamW", None, 512, 512): 59.0,
    ("AdamW", None, 2048, 8): 1.8,
    ("AdamW", None, 2048, 64): 19.0,
    ("AdamW", None, 2048, 512): 70.0,
    ("AdamW", None, 8192, 8): 0.15,
    ("AdamW", None, 8192, 64): 5.4,
    ("AdamW", None, 8192, 512): 39.0,
    ("AdamW", None, 32768, 8): 0.7,
    ("AdamW", None, 32768, 64): 0.75,
    ("AdamW", None, 32768, 512): 230.0,
    ("AdamH", "matrix", 512, 8): 0.85,
    ("AdamH", "matrix", 512, 64): 3.1,
    ("AdamH", "matrix", 512, 512): 190.0,
    ("AdamH", "matrix", 2048, 8): 0.99,
    ("AdamH", "matrix", 2048, 64): 1.0,
    ("AdamH", "matrix", 2048, 512): 30.0,
    ("AdamH", "matrix", 8192, 8): 0.023,
    ("AdamH", "matrix", 8192, 64): 0.16,
    ("AdamH", "matrix", 8192, 512): 1.7,
    ("AdamH", "matrix", 32768, 8): 0.02,
    ("AdamH", "matrix", 32768, 64): 0.13,
    ("AdamH", "matrix", 32768, 512): 0.28,
    ("AdamH", "row", 512, 8): 0.49,
    ("AdamH", "row", 512, 64): 49.0,
    ("AdamH", "row", 512, 512): 510.0,
    ("AdamH", "row", 2048, 8): 0.2,
    ("AdamH", "row", 2048, 64): 0.48,
    ("AdamH", "row", 2048, 512): 40.0,
    ("AdamH", "row", 8192, 8): 0.05,
    ("AdamH", "row", 8192, 64): 0.39,
    ("AdamH", "row", 8192, 512): 1.2,
    ("AdamH", "row", 32768, 8): 0.014,
    ("AdamH", "row", 32768, 64): 0.04,
    ("AdamH", "row", 32768, 512): 0.49,
    ("SGD", None, 512, 8): 5_400_000.0,
    ("SGD", None, 512, 64): 79_000_000.0,
    ("SGD", None, 512, 512): 590_000_000.0,
    ("SGD", None, 2048, 8): 9_400_000.0,
    ("SGD", None, 2048, 64): 55_000_000.0,
    ("SGD", None, 2048, 512): 460_000_000.0,
    ("SGD", None, 8192, 8): 8_200_000.0,
    ("SGD", None, 8192, 64): 30_000_000.0,
    ("SGD", None, 8192, 512): 390_000_000.0,
    ("SGD", None, 32768, 8): 1_900_000.0,
    ("SGD", None, 32768, 64): 9_100_000.0,
    ("SGD", None, 32768, 512): 69_000_000.0,
    ("SGD2", "matrix", 512, 8): 1.2,
    ("SGD2", "matrix", 512, 64): 10.0,
    ("SGD2", "matrix", 512, 512): 520.0,
    ("SGD2", "matrix", 2048, 8): 0.26,
    ("SGD2", "matrix", 2048, 64): 2.0,
    ("SGD2", "matrix", 2048, 512): 120.0,
    ("SGD2", "matrix", 8192, 8): 0.16,
    ("SGD2", "matrix", 8192, 64): 0.54,
    ("SGD2", "matrix", 8192, 512): 1.2,
    ("SGD2", "matrix", 32768, 8): 0.068,
    ("SGD2", "matrix", 32768, 64): 0.11,
    ("SGD2", "matrix", 32768, 512): 0.23,
    ("SGD2", "row", 512, 8): 0.81,
    ("SGD2", "row", 512, 64): 34.0,
    ("SGD2", "row", 512, 512): 180.0,
    ("SGD2", "row", 2048, 8): 0.29,
    ("SGD2", "row", 2048, 64): 2.8,
    ("SGD2", "row", 2048, 512): 85.0,
    ("SGD2", "row", 8192, 8): 0.23,
    ("SGD2", "row", 8192, 64): 0.49,
    ("SGD2", "row", 8192, 512): 0.95,
    ("SGD2", "row", 32768, 8): 0.039,
    ("SGD2", "row", 32768, 64): 0.1,
    ("SGD2", "row", 32768, 512): 0.59,
}
LR_RANGES = {
    key: (lr_median * 1e-2, lr_median * 1e2)
    for key, lr_median in _LR_MEDIANS.items()
    if key[0] in OPTIMIZERS
}
LR_DECAY = 4.0


@dataclass
class Config:
    seed: int = 0
    input_dim: int = 128
    num_classes: int = 4000
    train_size: int = DATASET_SIZE
    prob_noise_std: float = 1e-4
    eval_batch_size: int = 512
    target_sse: float = 5.6644e-7
    log_every: int = 0
    compile: bool = True
    cuda_graph: bool = True
    device: str = "cuda"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU execution is disabled")
        return torch.device("cuda")
    if device == "cpu":
        raise RuntimeError("CUDA is required; CPU execution is disabled")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def normalize_matrix(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm().clamp_min(eps)


def normalize_by_h_norm(
    x: torch.Tensor, h_norm: str, eps: float = 1e-12
) -> torch.Tensor:
    if h_norm == "row":
        return normalize_rows(x, eps)
    if h_norm == "matrix":
        return normalize_matrix(x, eps)
    raise ValueError(f"unknown h_norm: {h_norm}")


def normalize_update(
    update: torch.Tensor, h_norm: str, eps: float = 1e-12
) -> torch.Tensor:
    if h_norm == "row":
        update_norm = update.norm(dim=1, keepdim=True)
    elif h_norm == "matrix":
        update_norm = update.norm()
    else:
        raise ValueError(f"unknown h_norm: {h_norm}")
    normalized = update / update_norm.clamp_min(eps)
    return torch.where(update_norm > 0, normalized, torch.zeros_like(normalized))


@torch.no_grad()
def normalize_weight_(weight: torch.Tensor, h_norm: str) -> None:
    weight.copy_(normalize_by_h_norm(weight, h_norm))


def effective_train_weight(
    weight: torch.Tensor,
    log_scale: torch.Tensor,
    h_norm: str | None,
) -> torch.Tensor:
    scale = log_scale.exp()
    if h_norm is None:
        return weight * scale
    if h_norm in ("row", "matrix"):
        return normalize_by_h_norm(weight, h_norm) * scale
    raise ValueError(f"unknown h_norm: {h_norm}")


@torch.no_grad()
def initial_log_scale(
    h_norm: str | None,
    weight: torch.Tensor,
    ground_truth_matrix_norm: torch.Tensor,
) -> torch.Tensor:
    if h_norm == "row":
        row_rms = ground_truth_matrix_norm / math.sqrt(weight.size(0))
        return row_rms.log().expand(weight.size(0), 1).clone()
    if h_norm == "matrix":
        return ground_truth_matrix_norm.log().reshape(())
    if h_norm is None:
        return torch.zeros((), device=weight.device, dtype=weight.dtype)
    raise ValueError(f"unknown h_norm: {h_norm}")


def sample_ground_truth_row_norms(
    num_rows: int,
    device: torch.device,
    mean: float = 3.0,
    std: float = 0.5,
    low: float = 1.0,
    high: float = 5.0,
) -> torch.Tensor:
    row_norms = torch.empty(num_rows, 1, device=device)
    filled = 0
    while filled < num_rows:
        sample_count = max(num_rows - filled, 1024)
        candidates = mean + std * torch.randn(sample_count, device=device)
        candidates = candidates[(low <= candidates) & (candidates <= high)]
        if candidates.numel() == 0:
            continue
        take = min(num_rows - filled, candidates.numel())
        row_norms[filled : filled + take, 0] = candidates[:take]
        filled += take
    return row_norms


def deterministic_log_lr(low: float, high: float, idx: int, count: int) -> float:
    if count <= 1:
        return math.sqrt(low * high)
    progress = (idx + 0.5) / count
    return math.exp(math.log(low) + progress * (math.log(high) - math.log(low)))


def deterministic_choice(values: tuple[Any, ...], idx: int, stride: int = 1) -> Any:
    return values[(idx // stride) % len(values)]


def nearest_power_of_two(value: float) -> float:
    if value <= 0:
        raise ValueError(f"expected a positive value, got {value}")
    return 2.0 ** round(math.log2(value))


def scalar_adamw_lr(batch_size: int, num_samples: int) -> tuple[float, float]:
    raw_lr = (
        0.15
        * (batch_size / 128) ** 0.75
        * (num_samples / 8192) ** -0.75
    )
    return nearest_power_of_two(raw_lr), raw_lr


def integer_step_size(num_samples: int, batch_size: int) -> int:
    return max(1, int(round(num_samples / batch_size)))


def lr_schedule_factor(step: int, steps: int, hparams: dict[str, Any]) -> float:
    progress = step / max(1, steps)
    return math.exp(
        -float(hparams["lr_decay"]) * progress ** float(hparams["lr_power"])
    )


@torch.no_grad()
def build_problem(
    config: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = normalize_rows(torch.randn(config.train_size, config.input_dim, device=device))
    ground_truth_row_norms = sample_ground_truth_row_norms(
        config.num_classes, device=device
    )
    ground_truth_weight = (
        normalize_rows(torch.randn(config.num_classes, config.input_dim, device=device))
        * ground_truth_row_norms
    )
    ground_truth_matrix_norm = ground_truth_weight.norm()

    clean_target_probs = torch.empty(
        config.train_size, config.num_classes, device=device
    )
    noisy_target_probs = torch.empty(
        config.train_size, config.num_classes, device=device
    )
    for start in range(0, config.train_size, config.eval_batch_size):
        end = min(start + config.eval_batch_size, config.train_size)
        clean_logits = F.linear(x[start:end], ground_truth_weight)
        clean_probs = clean_logits.softmax(dim=1)
        clean_target_probs[start:end] = clean_probs
        noisy_target_probs[start:end] = make_noisy_probs(
            clean_probs, config.prob_noise_std
        )

    return (
        x,
        clean_target_probs,
        noisy_target_probs,
        ground_truth_row_norms,
        ground_truth_matrix_norm,
    )


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


def probability_batch_sse_loss(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(x_batch, weight).float()
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().sum(dim=1).mean()


def probability_batch_sse_loss_and_grads(
    weight: torch.Tensor,
    log_scale: torch.Tensor,
    h_norm: str | None,
    x_batch: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = log_scale.exp()
    if h_norm is None:
        base_weight = weight
        effective_weight = weight * scale
    elif h_norm == "matrix":
        weight_norm = weight.norm().clamp_min(1e-12)
        base_weight = weight / weight_norm
        effective_weight = base_weight * scale
    elif h_norm == "row":
        weight_norm = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        base_weight = weight / weight_norm
        effective_weight = base_weight * scale
    else:
        raise ValueError(f"unknown h_norm: {h_norm}")

    logits = F.linear(x_batch, effective_weight).float()
    output_probs = logits.softmax(dim=1)
    diff = output_probs - target_probs
    loss = diff.square().sum(dim=1).mean()

    centered_diff = diff - (diff * output_probs).sum(dim=1, keepdim=True)
    grad_logits = output_probs * centered_diff * (2.0 / x_batch.size(0))
    grad_effective_weight = grad_logits.mT @ x_batch

    if h_norm is None:
        grad_weight = grad_effective_weight * scale
        grad_log_scale = (grad_effective_weight * effective_weight).sum()
    elif h_norm == "matrix":
        grad_base_weight = grad_effective_weight * scale
        tangent_component = (grad_base_weight * base_weight).sum()
        grad_weight = (grad_base_weight - base_weight * tangent_component) / weight_norm
        grad_log_scale = (grad_effective_weight * effective_weight).sum()
    else:
        grad_base_weight = grad_effective_weight * scale
        tangent_component = (grad_base_weight * base_weight).sum(dim=1, keepdim=True)
        grad_weight = (
            grad_base_weight - base_weight * tangent_component
        ) / weight_norm
        grad_log_scale = (grad_effective_weight * effective_weight).sum(
            dim=1, keepdim=True
        )

    return loss, grad_weight.to(dtype=weight.dtype), grad_log_scale.to(
        dtype=log_scale.dtype
    )


@torch.compile(dynamic=True, fullgraph=True)
def probability_batch_sse_loss_compiled(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(x_batch, weight).float()
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().sum(dim=1).mean()


@torch.compile(dynamic=True, fullgraph=True)
def softmax_probs_compiled(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(x_batch, weight).float()
    return logits.softmax(dim=1)


def build_fixed_cycle_orders(
    train_size: int, device: torch.device, seed: int
) -> dict[tuple[int, int], torch.Tensor]:
    orders: dict[tuple[int, int], torch.Tensor] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 10_000)
    base_permutation = torch.randperm(train_size, device=device, generator=generator)

    for num_samples in NUM_SAMPLE_OPTIONS:
        for batch_size in BATCH_SIZES:
            steps = integer_step_size(num_samples, batch_size)
            needed = steps * batch_size
            repeats = math.ceil(needed / train_size)
            orders[(num_samples, batch_size)] = base_permutation.repeat(repeats)[
                :needed
            ]

    return orders


@torch.no_grad()
def adam_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_count: int,
    lr: float,
    lr_factor: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    h_norm: str | None,
) -> None:
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1**step_count
    bias2 = 1 - beta2**step_count
    update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
    step_lr = lr * lr_factor / max(1e-12, 1 - beta1)

    if h_norm is not None:
        weight.add_(normalize_update(update, h_norm) * (-step_lr))
        normalize_weight_(weight, h_norm)
    else:
        if wd:
            weight.mul_(1 - step_lr * wd)
        weight.add_(update * (-step_lr))


@torch.no_grad()
def adamw_log_scale_step(
    log_scale: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_count: int | torch.Tensor,
    lr: float,
    lr_factor: float | torch.Tensor,
    beta1: float = SCALAR_ADAMW_BETA1,
    beta2: float = SCALAR_ADAMW_BETA2,
    eps: float = SCALAR_ADAMW_EPS,
    wd: float = SCALAR_ADAMW_WD,
) -> None:
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1**step_count
    bias2 = 1 - beta2**step_count
    update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
    step_lr = lr * lr_factor / max(1e-12, 1 - beta1)
    if wd:
        log_scale.mul_(1 - step_lr * wd)
    log_scale.add_(update * (-step_lr))

@torch.no_grad()
def sgd_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    lr_factor: float | torch.Tensor,
    momentum: float,
    wd: float,
) -> None:
    step_lr = lr * lr_factor
    if wd:
        weight.mul_(1 - step_lr * wd)
    momentum_buffer.mul_(momentum).add_(grad)
    weight.add_(momentum_buffer * (-step_lr))


@torch.no_grad()
def sgd2_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    lr: float,
    lr_factor: float | torch.Tensor,
    beta1: float,
    beta2: float,
    step_count: int | torch.Tensor,
    nesterov: bool,
    h_norm: str,
) -> None:
    momentum_buffer.mul_(beta1).add_(grad, alpha=1 - beta1)
    if h_norm == "row":
        grad_norm = grad.norm(dim=1, keepdim=True)
    elif h_norm == "matrix":
        grad_norm = grad.norm()
    else:
        raise ValueError(f"unknown h_norm: {h_norm}")
    v_buffer.mul_(beta2).add_(grad_norm, alpha=1 - beta2)
    beta2_bias = 1 - beta2**step_count
    if isinstance(beta2_bias, torch.Tensor):
        beta2_bias = beta2_bias.clamp_min(1e-12)
    else:
        beta2_bias = max(1e-12, beta2_bias)
    v_hat = v_buffer / beta2_bias

    if nesterov:
        update = grad * (1 - beta1) + momentum_buffer * beta1
    else:
        update = momentum_buffer

    weight.add_((update / v_hat.clamp_min(1e-12)) * (-(lr * lr_factor)))
    normalize_weight_(weight, h_norm)


def build_search_hparams(config: Config) -> list[dict[str, Any]]:
    candidate_specs: list[dict[str, Any]] = []

    for optimizer in OPTIMIZERS:
        h_norms: tuple[str | None, ...]
        if optimizer in H_OPTIMIZERS:
            h_norms = H_NORMS
        else:
            h_norms = (None,)
        for num_samples in NUM_SAMPLE_OPTIONS:
            for batch_size in BATCH_SIZES:
                steps = integer_step_size(num_samples, batch_size)
                for h_norm in h_norms:
                    lr_range = LR_RANGES[(optimizer, h_norm, num_samples, batch_size)]
                    lr_center = math.sqrt(lr_range[0] * lr_range[1])
                    scalar_lr, raw_scalar_lr = scalar_adamw_lr(batch_size, num_samples)
                    for sample_idx in range(RUNS_PER_SETTING):
                        hparams: dict[str, Any] = {
                            "optimizer": optimizer,
                            "variant": optimizer
                            if h_norm is None
                            else f"{optimizer}_{h_norm}",
                            "batch_size": batch_size,
                            "steps": steps,
                            "step_size": steps,
                            "num_samples": num_samples,
                            "sample_mode": "fixed_cycle",
                            "lr_schedule": "exp_power",
                            "lr_power": LR_POWER,
                            "lr_decay": LR_DECAY,
                            "predicted_lr": lr_center,
                            "lr": deterministic_log_lr(
                                lr_range[0], lr_range[1], sample_idx, RUNS_PER_SETTING
                            ),
                            "scalar_optimizer": "AdamW",
                            "scalar_lr": scalar_lr,
                            "raw_scalar_lr": raw_scalar_lr,
                            "scalar_parameterization": "exp_log_scale",
                            "sample_idx": sample_idx,
                        }
                        if h_norm is not None:
                            hparams["h_norm"] = h_norm
                        if optimizer in ("AdamW", "AdamH"):
                            hparams["beta1"] = deterministic_choice(
                                BETA1_MOMENTUM_VALUES, sample_idx
                            )
                            hparams["beta2"] = deterministic_choice(
                                ADAM_BETA2S,
                                sample_idx,
                                len(BETA1_MOMENTUM_VALUES),
                            )
                            hparams["eps"] = ADAM_EPS
                        if optimizer == "SGD":
                            hparams["momentum"] = deterministic_choice(
                                BETA1_MOMENTUM_VALUES, sample_idx
                            )
                        if optimizer == "SGD2":
                            hparams["beta1"] = deterministic_choice(
                                BETA1_MOMENTUM_VALUES, sample_idx
                            )
                            hparams["beta2"] = deterministic_choice(
                                SGD2_BETA2S,
                                sample_idx,
                                len(BETA1_MOMENTUM_VALUES),
                            )
                            hparams["nesterov"] = False
                        if optimizer in ("AdamW", "SGD"):
                            hparams["wd"] = 0.0
                        candidate_specs.append(hparams)

    return candidate_specs


def setting_key(hparams: dict[str, Any]) -> str:
    key = (
        f"optimizer={hparams['optimizer']},num_samples={hparams['num_samples']},"
        f"batch_size={hparams['batch_size']}"
    )
    if "h_norm" in hparams:
        key += f",h_norm={hparams['h_norm']}"
    return key


def warmup_compiled_training(
    config: Config,
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    batch_orders: dict[tuple[int, int], torch.Tensor],
    device: torch.device,
) -> None:
    if not config.compile:
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()
    print(
        "COMPILE_WARMUP_START "
        + json.dumps(
            {
                "batch_sizes": BATCH_SIZES,
                "optimizer_count": len(OPTIMIZER_VARIANTS),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    warmup_num_samples = max(NUM_SAMPLE_OPTIONS)
    for batch_size in BATCH_SIZES:
        batch_idx = batch_orders[(warmup_num_samples, batch_size)][:batch_size]
        weight = initial_weight.detach().clone().to(device).requires_grad_()
        loss = probability_batch_sse_loss_compiled(
            weight, x[batch_idx], noisy_target_probs[batch_idx]
        )
        loss.backward()

    eval_shapes = {min(config.eval_batch_size, x.size(0))}
    final_eval_batch = x.size(0) % config.eval_batch_size
    if final_eval_batch:
        eval_shapes.add(final_eval_batch)
    for eval_batch_size in sorted(eval_shapes):
        _ = softmax_probs_compiled(initial_weight.detach(), x[:eval_batch_size])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        "COMPILE_WARMUP_DONE "
        + json.dumps({"elapsed_sec": time.time() - start_time}, sort_keys=True),
        flush=True,
    )


@torch.no_grad()
def evaluate_clean_sse_tensor(
    weight: torch.Tensor,
    x: torch.Tensor,
    clean_target_probs: torch.Tensor,
    config: Config,
    device: torch.device,
) -> float:
    squared_error_sum = 0.0
    num_examples = 0

    for start in range(0, x.size(0), config.eval_batch_size):
        end = min(start + config.eval_batch_size, x.size(0))
        if config.compile:
            output_probs = softmax_probs_compiled(weight, x[start:end])
        else:
            logits = F.linear(x[start:end], weight).float()
            output_probs = logits.softmax(dim=1)
        diff = output_probs - clean_target_probs[start:end]
        squared_error_sum += diff.square().sum().item()
        num_examples += diff.size(0)

    return squared_error_sum / num_examples


def apply_matrix_optimizer_step(
    optimizer: str,
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_count: int | torch.Tensor,
    base_lr: float,
    lr_factor: float | torch.Tensor,
    hparams: dict[str, Any],
    h_norm: str | None,
) -> None:
    if optimizer in ("AdamW", "AdamH"):
        adam_step(
            weight,
            grad,
            exp_avg,
            exp_avg_sq,
            step_count,
            base_lr,
            lr_factor,
            float(hparams["beta1"]),
            float(hparams["beta2"]),
            float(hparams["eps"]),
            float(hparams.get("wd", 0.0)),
            h_norm,
        )
    elif optimizer == "SGD":
        sgd_step(
            weight,
            grad,
            momentum_buffer,
            base_lr,
            lr_factor,
            float(hparams["momentum"]),
            float(hparams["wd"]),
        )
    elif optimizer == "SGD2":
        sgd2_step(
            weight,
            grad,
            momentum_buffer,
            v_buffer,
            base_lr,
            lr_factor,
            float(hparams["beta1"]),
            float(hparams["beta2"]),
            step_count,
            bool(hparams["nesterov"]),
            str(h_norm),
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")


def train_one_run(
    hparams: dict[str, Any],
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    clean_target_probs: torch.Tensor,
    ground_truth_row_norms: torch.Tensor,
    ground_truth_matrix_norm: torch.Tensor,
    initial_weight: torch.Tensor,
    batch_orders: dict[tuple[int, int], torch.Tensor],
    config: Config,
    device: torch.device,
    candidate_idx: int,
    num_candidates: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    h_norm = hparams.get("h_norm")
    if h_norm is not None:
        h_norm = str(h_norm)

    weight = initial_weight.detach().clone().to(device)
    if h_norm is not None:
        weight = normalize_by_h_norm(weight, h_norm).detach()
    weight.requires_grad_(True)
    log_scale = initial_log_scale(h_norm, weight, ground_truth_matrix_norm).to(device)
    log_scale = log_scale.detach().clone().requires_grad_(True)
    momentum_buffer = torch.zeros_like(weight)
    if h_norm == "matrix":
        v_buffer = torch.zeros((), device=device, dtype=weight.dtype)
    else:
        v_buffer = torch.zeros(weight.size(0), 1, device=device, dtype=weight.dtype)
    exp_avg = torch.zeros_like(weight)
    exp_avg_sq = torch.zeros_like(weight)
    scale_exp_avg = torch.zeros_like(log_scale)
    scale_exp_avg_sq = torch.zeros_like(log_scale)

    last_loss = float("nan")
    optimizer = str(hparams["optimizer"])
    if optimizer in H_OPTIMIZERS and h_norm is None:
        raise ValueError(f"{optimizer} requires h_norm")
    if optimizer not in H_OPTIMIZERS and h_norm is not None:
        raise ValueError(f"{optimizer} does not support h_norm")
    base_lr = float(hparams["lr"])
    scalar_lr = float(hparams["scalar_lr"])
    steps = int(hparams["steps"])
    num_samples = int(hparams["num_samples"])
    batch_size = int(hparams["batch_size"])
    batch_order = batch_orders[(num_samples, batch_size)]

    initial_weight_state = weight.detach().clone()
    initial_log_scale_state = log_scale.detach().clone()

    @torch.no_grad()
    def reset_training_state() -> None:
        weight.copy_(initial_weight_state)
        log_scale.copy_(initial_log_scale_state)
        momentum_buffer.zero_()
        v_buffer.zero_()
        exp_avg.zero_()
        exp_avg_sq.zero_()
        scale_exp_avg.zero_()
        scale_exp_avg_sq.zero_()
        if weight.grad is not None:
            weight.grad.zero_()
        if log_scale.grad is not None:
            log_scale.grad.zero_()

    def run_eager_training() -> float:
        run_last_loss = float("nan")
        for step in range(steps):
            lr_factor = lr_schedule_factor(step, steps, hparams)
            batch_start = step * batch_size
            batch_idx = batch_order[batch_start : batch_start + batch_size]
            weight.grad = None
            log_scale.grad = None
            loss_weight = effective_train_weight(weight, log_scale, h_norm)
            if config.compile:
                loss = probability_batch_sse_loss_compiled(
                    loss_weight,
                    x[batch_idx],
                    noisy_target_probs[batch_idx],
                )
            else:
                loss = probability_batch_sse_loss(
                    loss_weight,
                    x[batch_idx],
                    noisy_target_probs[batch_idx],
                )

            loss.backward()
            grad = weight.grad
            scale_grad = log_scale.grad
            if grad is None:
                raise RuntimeError("loss did not produce a weight gradient")
            if scale_grad is None:
                raise RuntimeError("loss did not produce a log-scale gradient")

            apply_matrix_optimizer_step(
                optimizer,
                weight,
                grad,
                momentum_buffer,
                v_buffer,
                exp_avg,
                exp_avg_sq,
                step + 1,
                base_lr,
                lr_factor,
                hparams,
                h_norm,
            )
            adamw_log_scale_step(
                log_scale,
                scale_grad,
                scale_exp_avg,
                scale_exp_avg_sq,
                step + 1,
                scalar_lr,
                lr_factor,
            )

            run_last_loss = float(loss.detach().item())
            should_log = config.log_every > 0 and (
                (step + 1) % config.log_every == 0 or step + 1 == steps
            )
            if should_log:
                print(
                    f"candidate {candidate_idx + 1:05d}/{num_candidates:05d} "
                    f"{optimizer} step {step + 1:05d}/{steps:05d} "
                    f"loss={run_last_loss:.8g} "
                    f"lr={base_lr * lr_factor:.6g} "
                    f"scalar_lr={scalar_lr * lr_factor:.6g}",
                    flush=True,
                )
        return run_last_loss

    def run_cuda_graph_training() -> float:
        static_x = torch.empty(
            batch_size, x.size(1), device=device, dtype=x.dtype
        )
        static_target = torch.empty(
            batch_size,
            noisy_target_probs.size(1),
            device=device,
            dtype=noisy_target_probs.dtype,
        )
        static_lr_factor = torch.empty((), device=device, dtype=torch.float32)
        static_step_count = torch.empty((), device=device, dtype=torch.float32)

        first_idx = batch_order[:batch_size]
        torch.index_select(x, 0, first_idx, out=static_x)
        torch.index_select(noisy_target_probs, 0, first_idx, out=static_target)
        static_lr_factor.fill_(lr_schedule_factor(0, steps, hparams))
        static_step_count.fill_(1)

        probability_batch_sse_loss_and_grads(
            weight,
            log_scale,
            h_norm,
            static_x,
            static_target,
        )
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        static_loss: torch.Tensor
        with torch.cuda.graph(graph):
            static_loss, grad, scale_grad = probability_batch_sse_loss_and_grads(
                weight,
                log_scale,
                h_norm,
                static_x,
                static_target,
            )
            apply_matrix_optimizer_step(
                optimizer,
                weight,
                grad,
                momentum_buffer,
                v_buffer,
                exp_avg,
                exp_avg_sq,
                static_step_count,
                base_lr,
                static_lr_factor,
                hparams,
                h_norm,
            )
            adamw_log_scale_step(
                log_scale,
                scale_grad,
                scale_exp_avg,
                scale_exp_avg_sq,
                static_step_count,
                scalar_lr,
                static_lr_factor,
            )

        reset_training_state()
        for step in range(steps):
            lr_factor = lr_schedule_factor(step, steps, hparams)
            batch_start = step * batch_size
            batch_idx = batch_order[batch_start : batch_start + batch_size]
            torch.index_select(x, 0, batch_idx, out=static_x)
            torch.index_select(noisy_target_probs, 0, batch_idx, out=static_target)
            static_lr_factor.fill_(lr_factor)
            static_step_count.fill_(step + 1)
            graph.replay()
            should_log = config.log_every > 0 and (
                (step + 1) % config.log_every == 0 or step + 1 == steps
            )
            if should_log:
                run_last_loss = float(static_loss.detach().item())
                print(
                    f"candidate {candidate_idx + 1:05d}/{num_candidates:05d} "
                    f"{optimizer} step {step + 1:05d}/{steps:05d} "
                    f"loss={run_last_loss:.8g} "
                    f"lr={base_lr * lr_factor:.6g} "
                    f"scalar_lr={scalar_lr * lr_factor:.6g}",
                    flush=True,
                )
        return float(static_loss.detach().item())

    training_start_time = time.time()
    use_cuda_graph = config.cuda_graph and device.type == "cuda"
    used_cuda_graph = False
    if use_cuda_graph:
        try:
            last_loss = run_cuda_graph_training()
            used_cuda_graph = True
        except RuntimeError as exc:
            print(
                "CUDA_GRAPH_FALLBACK "
                + json.dumps(
                    {
                        "candidate_idx": candidate_idx,
                        "reason": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            reset_training_state()
            last_loss = run_eager_training()
    else:
        last_loss = run_eager_training()

    weight.grad = None
    log_scale.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed_sec = time.time() - training_start_time
    eval_weight = effective_train_weight(weight, log_scale, h_norm).detach()
    clean_sse = evaluate_clean_sse_tensor(
        eval_weight, x, clean_target_probs, config, device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time
    peak_allocated_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved_bytes = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    weight_detached = weight.detach()
    scale_detached = log_scale.detach().exp()
    weight_norms = weight_detached.norm(dim=1)
    effective_weight_norms = eval_weight.norm(dim=1)
    summary = {
        **hparams,
        "candidate_idx": candidate_idx,
        "actual_samples": steps * batch_size,
        "compiled": config.compile,
        "final_train_loss": last_loss,
        "clean_sse": clean_sse,
        "clean_train_sse": clean_sse,
        "target_met": clean_sse <= config.target_sse,
        "setting_key": setting_key(hparams),
        "cuda_graph": used_cuda_graph,
        "scalar_lr": scalar_lr,
        "scalar_scale_mean": float(scale_detached.mean().item()),
        "scalar_scale_std": float(scale_detached.std().item())
        if scale_detached.numel() > 1
        else 0.0,
        "scalar_scale_min": float(scale_detached.min().item()),
        "scalar_scale_max": float(scale_detached.max().item()),
        "weight_row_norm_mean": float(weight_norms.mean().item()),
        "weight_row_norm_std": float(weight_norms.std().item()),
        "weight_matrix_norm": float(weight_detached.norm().item()),
        "effective_weight_row_norm_mean": float(effective_weight_norms.mean().item()),
        "effective_weight_row_norm_std": float(effective_weight_norms.std().item()),
        "effective_weight_matrix_norm": float(eval_weight.norm().item()),
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
    config = Config()
    if config.train_size != DATASET_SIZE:
        raise ValueError(f"this experiment fixes --train-size to {DATASET_SIZE}")

    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)
    device = resolve_device(config.device)

    candidate_specs = build_search_hparams(config)
    print(
        "CONFIG "
        + json.dumps({**asdict(config), "device": str(device)}, sort_keys=True),
        flush=True,
    )
    print(
        "SEARCH_PLAN "
        + json.dumps(
            {
                "optimizers": OPTIMIZERS,
                "optimizer_variants": OPTIMIZER_VARIANTS,
                "num_candidates": len(candidate_specs),
                "batch_sizes": BATCH_SIZES,
                "runs_per_setting": RUNS_PER_SETTING,
                "dataset_size": DATASET_SIZE,
                "num_samples": NUM_SAMPLE_OPTIONS,
                "lr_ranges": {str(key): value for key, value in LR_RANGES.items()},
                "lr_decay": LR_DECAY,
                "lr_power": LR_POWER,
                "search_type": "deterministic_log_lr_cycle",
                "adam_eps": ADAM_EPS,
                "beta1_momentum_values": BETA1_MOMENTUM_VALUES,
                "scalar_adamw_beta1": SCALAR_ADAMW_BETA1,
                "scalar_adamw_beta2": SCALAR_ADAMW_BETA2,
                "scalar_adamw_eps": SCALAR_ADAMW_EPS,
                "scalar_adamw_wd": SCALAR_ADAMW_WD,
                "h_norms": H_NORMS,
                "wd": 0.0,
                "compile": config.compile,
                "cuda_graph": config.cuda_graph,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    (
        x,
        clean_target_probs,
        noisy_target_probs,
        ground_truth_row_norms,
        ground_truth_matrix_norm,
    ) = build_problem(config, device)
    initial_weight = normalize_rows(
        torch.randn(config.num_classes, config.input_dim, device=device)
    )
    batch_orders = build_fixed_cycle_orders(config.train_size, device, config.seed)
    warmup_compiled_training(
        config, x, noisy_target_probs, initial_weight, batch_orders, device
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
            ground_truth_row_norms,
            ground_truth_matrix_norm,
            initial_weight,
            batch_orders,
            config,
            device,
            candidate_idx,
            len(candidate_specs),
        )
        summaries.append(summary)
        key = str(summary["setting_key"])
        best = best_by_setting.get(key)
        if best is None or float(summary["clean_train_sse"]) < float(
            best["clean_train_sse"]
        ):
            best_by_setting[key] = summary

    summaries_by_sse = sorted(
        summaries, key=lambda item: float(item["clean_train_sse"])
    )
    print(f"BEST_BY_SETTING {json.dumps(best_by_setting, sort_keys=True)}", flush=True)
    print(f"FINAL_SUMMARY {json.dumps(summaries_by_sse, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
