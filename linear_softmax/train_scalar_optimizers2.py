"""Run optimizer hyperparameter search for fixed-matrix scalar softmax problems."""

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
NUM_SAMPLE_OPTIONS = (2048, 4096, 8192, 16384, 32768)
OPTIMIZERS = ("AdamW",)
H_NORMS = ("row",)
OPTIMIZER_VARIANTS = tuple(
    f"{optimizer}_{h_norm}" for optimizer in OPTIMIZERS for h_norm in H_NORMS
)
BATCH_SIZES = (8, 16, 32, 64, 128, 256, 512)
BETA1_MOMENTUM_VALUES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-10
ADAM_LR_GRID = tuple(2.0**exponent for exponent in range(-10, 5))
LR_DECAY = 4.0
LR_POWER = 1.0


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
    device: str = "cuda"


@dataclass(frozen=True)
class ToyDataset:
    h_norm: str
    fixed_weight: torch.Tensor
    fixed_logits: torch.Tensor
    clean_target_probs: torch.Tensor
    noisy_target_probs: torch.Tensor
    scalar_shape: tuple[int, ...]


@dataclass(frozen=True)
class PreparedRun:
    fixed_logits: torch.Tensor
    noisy_targets: torch.Tensor
    step_lrs: list[float]
    steps: int
    batch_size: int


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


@torch.no_grad()
def build_fixed_logits(
    x: torch.Tensor,
    fixed_weight: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    fixed_logits = torch.empty(config.train_size, config.num_classes, device=x.device)
    for start in range(0, config.train_size, config.eval_batch_size):
        end = min(start + config.eval_batch_size, config.train_size)
        fixed_logits[start:end] = F.linear(x[start:end], fixed_weight)
    return fixed_logits


@torch.no_grad()
def build_target_probs(
    fixed_logits: torch.Tensor,
    target_scalars: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_target_probs = torch.empty(
        config.train_size, config.num_classes, device=fixed_logits.device
    )
    noisy_target_probs = torch.empty(
        config.train_size, config.num_classes, device=fixed_logits.device
    )
    for start in range(0, config.train_size, config.eval_batch_size):
        end = min(start + config.eval_batch_size, config.train_size)
        clean_logits = fixed_logits[start:end] * target_scalars
        clean_probs = clean_logits.softmax(dim=1)
        clean_target_probs[start:end] = clean_probs
        noisy_target_probs[start:end] = make_noisy_probs(
            clean_probs, config.prob_noise_std
        )
    return clean_target_probs, noisy_target_probs


@torch.no_grad()
def build_problem(config: Config, device: torch.device) -> dict[str, ToyDataset]:
    x = normalize_rows(torch.randn(config.train_size, config.input_dim, device=device))
    ground_truth_row_norms = sample_ground_truth_row_norms(
        config.num_classes, device=device
    )
    ground_truth_weight = (
        normalize_rows(torch.randn(config.num_classes, config.input_dim, device=device))
        * ground_truth_row_norms
    )

    row_fixed_weight = normalize_rows(ground_truth_weight)
    row_fixed_logits = build_fixed_logits(x, row_fixed_weight, config)
    row_clean_probs, row_noisy_probs = build_target_probs(
        row_fixed_logits, ground_truth_row_norms.T, config
    )

    datasets = {
        "row": ToyDataset(
            h_norm="row",
            fixed_weight=row_fixed_weight,
            fixed_logits=row_fixed_logits,
            clean_target_probs=row_clean_probs,
            noisy_target_probs=row_noisy_probs,
            scalar_shape=(config.num_classes,),
        ),
    }
    return datasets


def option_specs_for_optimizer(optimizer: str) -> tuple[dict[str, bool], ...]:
    if optimizer == "AdamW":
        return ({"flush_last": False}, {"flush_last": True})
    raise ValueError(f"unknown optimizer: {optimizer}")


def integer_step_size(num_samples: int, batch_size: int) -> int:
    return max(1, int(round(num_samples / batch_size)))


def lr_schedule_factor(step: int, steps: int, hparams: dict[str, Any]) -> float:
    progress = step / max(1, steps)
    return math.exp(
        -float(hparams["lr_decay"]) * progress ** float(hparams["lr_power"])
    )


def effective_weight(
    log_scalars: torch.Tensor, fixed_weight: torch.Tensor
) -> torch.Tensor:
    scalars = log_scalars.exp()
    if scalars.ndim == 1:
        scalars = scalars[:, None]
    return fixed_weight * scalars


def scaled_logits(
    log_scalars: torch.Tensor, fixed_logits: torch.Tensor
) -> torch.Tensor:
    return fixed_logits * log_scalars.exp()


def scalar_batch_sse_loss_and_grad(
    log_scalars: torch.Tensor,
    fixed_logits_batch: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = scaled_logits(log_scalars, fixed_logits_batch).float()
    output_probs = logits.softmax(dim=1)
    diff = output_probs - target_probs
    loss = diff.square().sum(dim=1).mean()
    centered_diff = diff - (diff * output_probs).sum(dim=1, keepdim=True)
    dlogits = (2.0 / fixed_logits_batch.size(0)) * output_probs * centered_diff
    grad = dlogits * logits
    if log_scalars.numel() == 1:
        return loss, grad.sum().reshape_as(log_scalars)
    return loss, grad.sum(dim=0)


@torch.compile(dynamic=True, fullgraph=True)
def scalar_batch_sse_loss_and_grad_compiled(
    log_scalars: torch.Tensor,
    fixed_logits_batch: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return scalar_batch_sse_loss_and_grad(log_scalars, fixed_logits_batch, target_probs)


@torch.compile(dynamic=True, fullgraph=True)
def softmax_probs_compiled(
    log_scalars: torch.Tensor,
    fixed_logits_batch: torch.Tensor,
) -> torch.Tensor:
    logits = scaled_logits(log_scalars, fixed_logits_batch).float()
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


def build_search_hparams(config: Config) -> list[dict[str, Any]]:
    candidate_specs: list[dict[str, Any]] = []

    for optimizer in OPTIMIZERS:
        option_specs = option_specs_for_optimizer(optimizer)
        for h_norm in H_NORMS:
            for num_samples in NUM_SAMPLE_OPTIONS:
                for batch_size in BATCH_SIZES:
                    steps = integer_step_size(num_samples, batch_size)
                    for option_spec in option_specs:
                        grid_idx = 0
                        for lr in ADAM_LR_GRID:
                            for beta1 in BETA1_MOMENTUM_VALUES:
                                hparams: dict[str, Any] = {
                                    "optimizer": optimizer,
                                    "variant": f"{optimizer}_{h_norm}",
                                    "h_norm": h_norm,
                                    "batch_size": batch_size,
                                    "steps": steps,
                                    "step_size": steps,
                                    "num_samples": num_samples,
                                    "sample_mode": "fixed_cycle",
                                    "lr_schedule": "exp_power",
                                    "lr_power": LR_POWER,
                                    "lr_decay": LR_DECAY,
                                    "predicted_lr": math.sqrt(
                                        ADAM_LR_GRID[0] * ADAM_LR_GRID[-1]
                                    ),
                                    "lr": lr,
                                    "beta1": beta1,
                                    "beta2": ADAM_BETA2,
                                    "eps": ADAM_EPS,
                                    "sample_idx": grid_idx,
                                    "grid_idx": grid_idx,
                                    "wd": 0.0,
                                    **option_spec,
                                }
                                if optimizer != "AdamW":
                                    raise ValueError(f"unknown optimizer: {optimizer}")
                                candidate_specs.append(hparams)
                                grid_idx += 1

    return candidate_specs


def setting_key(hparams: dict[str, Any]) -> str:
    parts = [
        f"optimizer={hparams['optimizer']}",
        f"h_norm={hparams['h_norm']}",
    ]
    for key in ("nesterov", "disable_bias1", "adaptive_norm", "flush_last"):
        if key in hparams:
            parts.append(f"{key}={hparams[key]}")
    parts.extend(
        [
            f"num_samples={hparams['num_samples']}",
            f"batch_size={hparams['batch_size']}",
        ]
    )
    return ",".join(parts)


def warmup_compiled_training(
    config: Config,
    datasets: dict[str, ToyDataset],
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
                "h_norms": H_NORMS,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    warmup_num_samples = max(NUM_SAMPLE_OPTIONS)
    for dataset in datasets.values():
        for batch_size in BATCH_SIZES:
            batch_idx = batch_orders[(warmup_num_samples, batch_size)][:batch_size]
            log_scalars = torch.zeros(dataset.scalar_shape, device=device)
            _ = scalar_batch_sse_loss_and_grad_compiled(
                log_scalars,
                dataset.fixed_logits[batch_idx],
                dataset.noisy_target_probs[batch_idx],
            )

    train_size = next(iter(datasets.values())).fixed_logits.size(0)
    eval_shapes = {min(config.eval_batch_size, train_size)}
    final_eval_batch = train_size % config.eval_batch_size
    if final_eval_batch:
        eval_shapes.add(final_eval_batch)
    for dataset in datasets.values():
        log_scalars = torch.zeros(dataset.scalar_shape, device=device)
        for eval_batch_size in sorted(eval_shapes):
            _ = softmax_probs_compiled(
                log_scalars, dataset.fixed_logits[:eval_batch_size]
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        "COMPILE_WARMUP_DONE "
        + json.dumps({"elapsed_sec": time.time() - start_time}, sort_keys=True),
        flush=True,
    )


@torch.no_grad()
def evaluate_clean_sse_tensor(
    log_scalars: torch.Tensor,
    fixed_logits: torch.Tensor,
    clean_target_probs: torch.Tensor,
    config: Config,
    device: torch.device,
) -> float:
    squared_error_sum = 0.0
    num_examples = 0

    for start in range(0, fixed_logits.size(0), config.eval_batch_size):
        end = min(start + config.eval_batch_size, fixed_logits.size(0))
        if config.compile:
            output_probs = softmax_probs_compiled(log_scalars, fixed_logits[start:end])
        else:
            logits = scaled_logits(log_scalars, fixed_logits[start:end]).float()
            output_probs = logits.softmax(dim=1)
        diff = output_probs - clean_target_probs[start:end]
        squared_error_sum += diff.square().sum().item()
        num_examples += diff.size(0)

    return squared_error_sum / num_examples


@torch.no_grad()
def adamw_step(
    log_scalars: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bias1: float | None = None,
    bias2: float | None = None,
    update_scale: float = 1.0,
) -> None:
    if wd:
        log_scalars.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    if bias1 is None:
        bias1 = 1 - beta1**step_count
    if bias2 is None:
        bias2 = 1 - beta2**step_count
    denom = (exp_avg_sq / bias2).sqrt().add_(eps)
    log_scalars.addcdiv_(exp_avg / bias1, denom, value=-lr * update_scale)


def adamw_graph_step(
    log_scalars: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    fixed_logits_batch: torch.Tensor,
    target_probs: torch.Tensor,
    lr: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    eps: torch.Tensor,
    wd: torch.Tensor,
    update_scale: torch.Tensor,
) -> torch.Tensor:
    loss, grad = scalar_batch_sse_loss_and_grad(
        log_scalars, fixed_logits_batch, target_probs
    )
    log_scalars.copy_(log_scalars * (1 - lr * wd))
    exp_avg.copy_(exp_avg * beta1 + grad * (1 - beta1))
    exp_avg_sq.copy_(exp_avg_sq * beta2 + grad.square() * (1 - beta2))
    update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
    log_scalars.copy_(log_scalars - lr * update * update_scale)
    return loss


def should_use_cuda_graph(config: Config, device: torch.device) -> bool:
    return config.compile and device.type == "cuda" and config.log_every == 0


def prepare_run_batches(
    hparams: dict[str, Any],
    dataset: ToyDataset,
    batch_orders: dict[tuple[int, int], torch.Tensor],
    config: Config,
) -> PreparedRun:
    steps = int(hparams["steps"])
    num_samples = int(hparams["num_samples"])
    batch_size = int(hparams["batch_size"])
    batch_order = batch_orders[(num_samples, batch_size)]
    fixed_logits = dataset.fixed_logits[batch_order].view(
        steps, batch_size, config.num_classes
    )
    noisy_targets = dataset.noisy_target_probs[batch_order].view(
        steps, batch_size, config.num_classes
    )
    step_lrs = [
        float(hparams["lr"]) * lr_schedule_factor(step, steps, hparams)
        for step in range(steps)
    ]
    return PreparedRun(
        fixed_logits=fixed_logits,
        noisy_targets=noisy_targets,
        step_lrs=step_lrs,
        steps=steps,
        batch_size=batch_size,
    )


def adamw_bias_corrections(
    hparams: dict[str, Any], steps: int
) -> tuple[float, float, list[float], list[float]]:
    beta1 = float(hparams["beta1"])
    beta2 = float(hparams["beta2"])
    bias1_values = [1 - beta1 ** (step + 1) for step in range(steps)]
    bias2_values = [1 - beta2 ** (step + 1) for step in range(steps)]
    return beta1, beta2, bias1_values, bias2_values


def adamw_update_scales(
    hparams: dict[str, Any], steps: int, beta1: float
) -> list[float]:
    update_scales = [1.0 for _ in range(steps)]
    if bool(hparams["flush_last"]):
        update_scales[-1] = 1.0 / (1.0 - beta1)
    return update_scales


def replay_cuda_graph_training(
    hparams: dict[str, Any],
    dataset: ToyDataset,
    prepared: PreparedRun,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    optimizer = str(hparams["optimizer"])
    log_scalars = torch.zeros(dataset.scalar_shape, device=device)
    static_logits = torch.empty(
        prepared.batch_size,
        config.num_classes,
        device=device,
        dtype=prepared.fixed_logits.dtype,
    )
    static_targets = torch.empty_like(static_logits)
    static_lr = torch.empty(1, device=device)
    static_wd = torch.tensor([float(hparams["wd"])], device=device)
    static_logits.copy_(prepared.fixed_logits[0])
    static_targets.copy_(prepared.noisy_targets[0])
    static_lr.fill_(prepared.step_lrs[0])

    if optimizer != "AdamW":
        raise ValueError(f"unknown optimizer: {optimizer}")
    beta1, beta2, bias1_values, bias2_values = adamw_bias_corrections(
        hparams, prepared.steps
    )
    exp_avg = torch.zeros_like(log_scalars)
    exp_avg_sq = torch.zeros_like(log_scalars)
    static_beta1 = torch.tensor([beta1], device=device)
    static_beta2 = torch.tensor([beta2], device=device)
    static_bias1 = torch.empty(1, device=device)
    static_bias2 = torch.empty(1, device=device)
    static_eps = torch.tensor([float(hparams["eps"])], device=device)
    static_update_scale = torch.empty(1, device=device)
    static_bias1.fill_(bias1_values[0])
    static_bias2.fill_(bias2_values[0])
    update_scale_values = adamw_update_scales(hparams, prepared.steps, beta1)
    static_update_scale.fill_(update_scale_values[0])

    def replayed_step() -> torch.Tensor:
        return adamw_graph_step(
            log_scalars,
            exp_avg,
            exp_avg_sq,
            static_logits,
            static_targets,
            static_lr,
            static_beta1,
            static_beta2,
            static_bias1,
            static_bias2,
            static_eps,
            static_wd,
            static_update_scale,
        )

    last_loss_tensor: torch.Tensor | None = None

    graph_stream = torch.cuda.Stream()
    graph_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph_stream):
        for _ in range(3):
            last_loss_tensor = replayed_step()
    torch.cuda.current_stream().wait_stream(graph_stream)

    log_scalars.zero_()
    exp_avg.zero_()
    exp_avg_sq.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        last_loss_tensor = replayed_step()

    torch.cuda.synchronize(device)
    training_start_time = time.time()
    for step in range(prepared.steps):
        static_logits.copy_(prepared.fixed_logits[step])
        static_targets.copy_(prepared.noisy_targets[step])
        static_lr.fill_(prepared.step_lrs[step])
        static_bias1.fill_(bias1_values[step])
        static_bias2.fill_(bias2_values[step])
        static_update_scale.fill_(update_scale_values[step])
        graph.replay()
    torch.cuda.synchronize(device)
    return (
        log_scalars.detach().clone(),
        last_loss_tensor,
        time.time() - training_start_time,
    )


def eager_training(
    hparams: dict[str, Any],
    dataset: ToyDataset,
    prepared: PreparedRun,
    config: Config,
    device: torch.device,
    candidate_idx: int,
    num_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    log_scalars = torch.zeros(dataset.scalar_shape, device=device)
    optimizer = str(hparams["optimizer"])
    if optimizer != "AdamW":
        raise ValueError(f"unknown optimizer: {optimizer}")
    exp_avg = torch.zeros_like(log_scalars)
    exp_avg_sq = torch.zeros_like(log_scalars)
    beta1, beta2, bias1_values, bias2_values = adamw_bias_corrections(
        hparams, prepared.steps
    )
    update_scale_values = adamw_update_scales(hparams, prepared.steps, beta1)

    last_loss_tensor: torch.Tensor | None = None
    training_start_time = time.time()
    for step in range(prepared.steps):
        step_lr = prepared.step_lrs[step]
        loss_and_grad = (
            scalar_batch_sse_loss_and_grad_compiled
            if config.compile
            else scalar_batch_sse_loss_and_grad
        )
        loss, grad = loss_and_grad(
            log_scalars,
            prepared.fixed_logits[step],
            prepared.noisy_targets[step],
        )

        adamw_step(
            log_scalars,
            grad,
            exp_avg,
            exp_avg_sq,
            step + 1,
            step_lr,
            beta1,
            beta2,
            float(hparams["eps"]),
            float(hparams["wd"]),
            bias1_values[step],
            bias2_values[step],
            update_scale_values[step],
        )

        last_loss_tensor = loss.detach()
        should_log = config.log_every > 0 and (
            (step + 1) % config.log_every == 0 or step + 1 == prepared.steps
        )
        if should_log:
            last_loss = float(last_loss_tensor.item())
            print(
                f"candidate {candidate_idx + 1:05d}/{num_candidates:05d} "
                f"{hparams['variant']} step {step + 1:05d}/{prepared.steps:05d} "
                f"loss={last_loss:.8g} lr={step_lr:.6g}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return log_scalars, last_loss_tensor, time.time() - training_start_time


def train_one_run(
    hparams: dict[str, Any],
    dataset: ToyDataset,
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

    prepared = prepare_run_batches(hparams, dataset, batch_orders, config)
    if should_use_cuda_graph(config, device):
        log_scalars, last_loss_tensor, training_elapsed_sec = (
            replay_cuda_graph_training(hparams, dataset, prepared, config, device)
        )
    else:
        log_scalars, last_loss_tensor, training_elapsed_sec = eager_training(
            hparams,
            dataset,
            prepared,
            config,
            device,
            candidate_idx,
            num_candidates,
        )

    last_loss = (
        float("nan") if last_loss_tensor is None else float(last_loss_tensor.item())
    )
    clean_sse = evaluate_clean_sse_tensor(
        log_scalars,
        dataset.fixed_logits,
        dataset.clean_target_probs,
        config,
        device,
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

    log_scalars_detached = log_scalars.detach()
    scalars = log_scalars_detached.exp()
    eval_weight = effective_weight(log_scalars_detached, dataset.fixed_weight).detach()
    effective_weight_norms = eval_weight.norm(dim=1)
    scalar_values = scalars.reshape(-1)
    log_scalar_values = log_scalars_detached.reshape(-1)
    summary = {
        **hparams,
        "candidate_idx": candidate_idx,
        "actual_samples": prepared.steps * prepared.batch_size,
        "compiled": config.compile,
        "final_train_loss": last_loss,
        "clean_sse": clean_sse,
        "clean_train_sse": clean_sse,
        "target_met": clean_sse <= config.target_sse,
        "setting_key": setting_key(hparams),
        "scalar_count": scalar_values.numel(),
        "scalar_mean": float(scalar_values.mean().item()),
        "scalar_std": float(scalar_values.std(unbiased=False).item()),
        "scalar_min": float(scalar_values.min().item()),
        "scalar_max": float(scalar_values.max().item()),
        "log_scalar_mean": float(log_scalar_values.mean().item()),
        "log_scalar_std": float(log_scalar_values.std(unbiased=False).item()),
        "log_scalar_min": float(log_scalar_values.min().item()),
        "log_scalar_max": float(log_scalar_values.max().item()),
        "effective_weight_row_norm_mean": float(effective_weight_norms.mean().item()),
        "effective_weight_row_norm_std": float(
            effective_weight_norms.std(unbiased=False).item()
        ),
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
                "search_mode": "grid",
                "dataset_size": DATASET_SIZE,
                "num_samples": NUM_SAMPLE_OPTIONS,
                "adam_lr_grid": ADAM_LR_GRID,
                "lr_decay": LR_DECAY,
                "lr_power": LR_POWER,
                "beta1_momentum_values": BETA1_MOMENTUM_VALUES,
                "adam_beta2": ADAM_BETA2,
                "adam_eps": ADAM_EPS,
                "adam_flush_last_values": (False, True),
                "h_norms": H_NORMS,
                "wd": 0.0,
                "compile": config.compile,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    datasets = build_problem(config, device)
    batch_orders = build_fixed_cycle_orders(config.train_size, device, config.seed)
    warmup_compiled_training(config, datasets, batch_orders, device)

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
        dataset = datasets[str(hparams["h_norm"])]
        summary = train_one_run(
            hparams,
            dataset,
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
