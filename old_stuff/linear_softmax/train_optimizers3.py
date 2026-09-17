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

NUM_SAMPLES = 30_000
OPTIMIZERS = ("SGD", "AdamW", "Muon")
H_OPTIMIZERS = ("AdamH", "MuonH", "SGD2")
H_NORMS = ("matrix", "row")
BATCH_SIZES = (8, 64, 512)
RUNS_PER_SETTING = 100
BETA1_MOMENTUM_VALUES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)
ADAM_BETA2S = (0.9, 0.95, 0.99, 0.999)
ADAM_EPS = (1e-8, 1e-7, 1e-6)
SGD2_BETA2S = (0.0, 0.5, 0.8, 0.9, 0.99)
MUON_NS_STEPS = 5
LR_POWER = 1.0
LR_RANGES = {
    ("AdamW", None, 8): (0.000564373985271, 0.564373985271),
    ("AdamW", None, 64): (0.00097082418243, 0.97082418243),
    ("AdamW", None, 512): (0.1, 100),
    ("AdamH", "matrix", 8): (0.000321558668441, 0.321558668441),
    ("AdamH", "matrix", 64): (0.000754226690167, 0.754226690168),
    ("AdamH", "matrix", 512): (0.0041174325842, 4.1174325842),
    ("AdamH", "row", 8): (0.000455041429207, 0.455041429207),
    ("AdamH", "row", 64): (0.00167571065974, 1.67571065974),
    ("AdamH", "row", 512): (0.00867651606696, 8.67651606697),
    ("Muon", None, 8): (0.00224429988238, 2.24429988239),
    ("Muon", None, 64): (0.0028108955144, 2.8108955144),
    ("Muon", None, 512): (0.1, 10),
    ("MuonH", "matrix", 8): (3.22760398855e-05, 0.0322760398855),
    ("MuonH", "matrix", 64): (0.000195387431767, 0.195387431767),
    ("MuonH", "matrix", 512): (0.000670325200695, 0.670325200696),
    ("MuonH", "row", 8): (0.000232280241754, 0.232280241754),
    ("MuonH", "row", 64): (0.00101561549891, 1.01561549892),
    ("MuonH", "row", 512): (0.000694805898839, 0.694805898839),
    ("SGD", None, 8): (10**5, 10**8),
    ("SGD", None, 64): (10**5, 10**8),
    ("SGD", None, 512): (10**6, 10**9),
    ("SGD2", "matrix", 8): (0.000283270118163, 0.283270118163),
    ("SGD2", "matrix", 64): (0.000877609154613, 0.877609154613),
    ("SGD2", "matrix", 512): (0.00669144160666, 6.69144160667),
    ("SGD2", "row", 8): (0.000818460391287, 0.818460391287),
    ("SGD2", "row", 64): (0.00266980679043, 2.66980679043),
    ("SGD2", "row", 512): (0.00538357699407, 5.38357699407),
}
LR_DECAY_RANGES = {
    ("AdamW", 8): (2.630749191, 5.730749191),
    ("AdamW", 64): (2.151553301, 5.251553301),
    ("AdamW", 512): (2.357988924, 5.457988924),
    ("AdamH", 8): (4.11643768, 7.21643768),
    ("AdamH", 64): (3.882404688, 6.982404688),
    ("AdamH", 512): (4.261080186, 7.361080186),
    ("Muon", 8): (3.878510559, 6.978510559),
    ("Muon", 64): (3.732636476, 6.832636476),
    ("Muon", 512): (3.239584468, 6.339584468),
    ("MuonH", 8): (3.411133372, 6.511133372),
    ("MuonH", 64): (3.775048348, 6.875048348),
    ("MuonH", 512): (4.485693111, 7.585693111),
    ("SGD", 8): (3.409363211, 6.509363211),
    ("SGD", 64): (3.239653394, 6.339653394),
    ("SGD", 512): (3.119987218, 6.219987218),
    ("SGD2", 8): (3.534584533, 6.634584533),
    ("SGD2", 64): (3.816028726, 6.916028726),
    ("SGD2", 512): (4.510107057, 7.610107057),
}
POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


@dataclass
class Config:
    seed: int = 0
    input_dim: int = 128
    num_classes: int = 4000
    train_size: int = NUM_SAMPLES
    prob_noise_std: float = 1e-4
    eval_batch_size: int = 512
    target_sse: float = 5.6644e-7
    log_every: int = 0
    compile: bool = True
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


def effective_h_weight(
    weight: torch.Tensor,
    h_norm: str | None,
    ground_truth_row_norms: torch.Tensor,
    ground_truth_matrix_norm: torch.Tensor,
) -> torch.Tensor:
    if h_norm is None:
        return weight
    normalized_weight = normalize_by_h_norm(weight, h_norm)
    if h_norm == "row":
        return normalized_weight * ground_truth_row_norms
    if h_norm == "matrix":
        return normalized_weight * ground_truth_matrix_norm
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


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def integer_step_size(batch_size: int) -> int:
    return max(1, int(round(NUM_SAMPLES / batch_size)))


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
) -> dict[int, torch.Tensor]:
    orders: dict[int, torch.Tensor] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 10_000)
    base_permutation = torch.randperm(train_size, device=device, generator=generator)

    for batch_size in BATCH_SIZES:
        steps = integer_step_size(batch_size)
        needed = steps * batch_size
        repeats = math.ceil(needed / train_size)
        orders[batch_size] = base_permutation.repeat(repeats)[:needed]

    return orders


def zeropower_via_newtonschulz5(update: torch.Tensor) -> torch.Tensor:
    x = update.float()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if x.size(-2) > x.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:MUON_NS_STEPS]:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:MUON_NS_STEPS]:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.to(dtype=update.dtype)


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
    step_lr = lr * lr_factor

    if h_norm is not None:
        weight.add_(normalize_update(update, h_norm), alpha=-step_lr)
        normalize_weight_(weight, h_norm)
    else:
        if wd:
            weight.mul_(1 - step_lr * wd)
        weight.add_(update, alpha=-step_lr)


@torch.no_grad()
def muon_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    lr_factor: float,
    momentum: float,
    wd: float,
    h_norm: str | None,
) -> None:
    momentum_buffer.lerp_(grad, 1 - momentum)
    update = grad.lerp(momentum_buffer, momentum)
    update = zeropower_via_newtonschulz5(update)
    step_lr = lr * lr_factor * max(1.0, weight.size(0) / weight.size(1)) ** 0.5

    if h_norm is not None:
        weight.add_(normalize_update(update, h_norm), alpha=-step_lr)
        normalize_weight_(weight, h_norm)
    else:
        if wd:
            weight.mul_(1 - step_lr * wd)
        weight.add_(update, alpha=-step_lr)


@torch.no_grad()
def sgd_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    lr_factor: float,
    momentum: float,
    wd: float,
) -> None:
    step_lr = lr * lr_factor
    if wd:
        weight.mul_(1 - step_lr * wd)
    momentum_buffer.mul_(momentum).add_(grad)
    weight.add_(momentum_buffer, alpha=-step_lr)


@torch.no_grad()
def sgd2_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    lr: float,
    lr_factor: float,
    beta1: float,
    beta2: float,
    step_count: int,
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
    v_hat = v_buffer / max(1e-12, 1 - beta2**step_count)

    if nesterov:
        update = grad * (1 - beta1) + momentum_buffer * beta1
    else:
        update = momentum_buffer

    weight.add_(update / v_hat.clamp_min(1e-12), alpha=-(lr * lr_factor))
    normalize_weight_(weight, h_norm)


def build_search_hparams(config: Config) -> list[dict[str, Any]]:
    rng = random.Random(config.seed + 1_000_003)
    candidate_specs: list[dict[str, Any]] = []

    for optimizer in OPTIMIZERS:
        for batch_size in BATCH_SIZES:
            h_norms: tuple[str | None, ...]
            if optimizer in H_OPTIMIZERS:
                h_norms = H_NORMS
            else:
                h_norms = (None,)
            steps = integer_step_size(batch_size)
            lr_decay_range = LR_DECAY_RANGES[(optimizer, batch_size)]
            for h_norm in h_norms:
                lr_range = LR_RANGES[(optimizer, h_norm, batch_size)]
                lr_center = math.sqrt(lr_range[0] * lr_range[1])
                for sample_idx in range(RUNS_PER_SETTING):
                    hparams: dict[str, Any] = {
                        "optimizer": optimizer,
                        "variant": optimizer
                        if h_norm is None
                        else f"{optimizer}_{h_norm}",
                        "batch_size": batch_size,
                        "steps": steps,
                        "step_size": steps,
                        "num_samples": NUM_SAMPLES,
                        "sample_mode": "fixed_cycle",
                        "lr_schedule": "exp_power",
                        "lr_power": LR_POWER,
                        "lr_decay": rng.uniform(*lr_decay_range),
                        "predicted_lr": lr_center,
                        "lr": log_uniform(rng, *lr_range),
                        "sample_idx": sample_idx,
                    }
                    if h_norm is not None:
                        hparams["h_norm"] = h_norm
                    if optimizer in ("Muon", "MuonH", "SGD"):
                        hparams["momentum"] = rng.choice(BETA1_MOMENTUM_VALUES)
                    if optimizer in ("AdamW", "AdamH"):
                        hparams["beta1"] = rng.choice(BETA1_MOMENTUM_VALUES)
                        hparams["beta2"] = rng.choice(ADAM_BETA2S)
                        hparams["eps"] = rng.choice(ADAM_EPS)
                    if optimizer == "SGD2":
                        hparams["beta1"] = rng.choice(BETA1_MOMENTUM_VALUES)
                        hparams["beta2"] = rng.choice(SGD2_BETA2S)
                        hparams["nesterov"] = False
                    if optimizer in ("AdamW", "Muon", "SGD"):
                        hparams["wd"] = 0.0
                    candidate_specs.append(hparams)

    return candidate_specs


def setting_key(hparams: dict[str, Any]) -> str:
    key = f"optimizer={hparams['optimizer']},batch_size={hparams['batch_size']}"
    if "h_norm" in hparams:
        key += f",h_norm={hparams['h_norm']}"
    return key


def warmup_compiled_training(
    config: Config,
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    batch_orders: dict[int, torch.Tensor],
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
            {"batch_sizes": BATCH_SIZES, "optimizer_count": len(OPTIMIZERS)},
            sort_keys=True,
        ),
        flush=True,
    )

    for batch_size in BATCH_SIZES:
        batch_idx = batch_orders[batch_size][:batch_size]
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


def train_one_run(
    hparams: dict[str, Any],
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    clean_target_probs: torch.Tensor,
    ground_truth_row_norms: torch.Tensor,
    ground_truth_matrix_norm: torch.Tensor,
    initial_weight: torch.Tensor,
    batch_orders: dict[int, torch.Tensor],
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
    momentum_buffer = torch.zeros_like(weight)
    if h_norm == "matrix":
        v_buffer = torch.zeros((), device=device, dtype=weight.dtype)
    else:
        v_buffer = torch.zeros(weight.size(0), 1, device=device, dtype=weight.dtype)
    exp_avg = torch.zeros_like(weight)
    exp_avg_sq = torch.zeros_like(weight)

    last_loss = float("nan")
    optimizer = str(hparams["optimizer"])
    if optimizer in H_OPTIMIZERS and h_norm is None:
        raise ValueError(f"{optimizer} requires h_norm")
    if optimizer not in H_OPTIMIZERS and h_norm is not None:
        raise ValueError(f"{optimizer} does not support h_norm")
    base_lr = float(hparams["lr"])
    steps = int(hparams["steps"])
    batch_size = int(hparams["batch_size"])
    batch_order = batch_orders[batch_size]

    training_start_time = time.time()
    for step in range(steps):
        lr_factor = lr_schedule_factor(step, steps, hparams)
        batch_start = step * batch_size
        batch_idx = batch_order[batch_start : batch_start + batch_size]
        weight.grad = None
        loss_weight = effective_h_weight(
            weight, h_norm, ground_truth_row_norms, ground_truth_matrix_norm
        )
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
        if grad is None:
            raise RuntimeError("loss did not produce a weight gradient")

        if optimizer in ("AdamW", "AdamH"):
            adam_step(
                weight,
                grad,
                exp_avg,
                exp_avg_sq,
                step + 1,
                base_lr,
                lr_factor,
                float(hparams["beta1"]),
                float(hparams["beta2"]),
                float(hparams["eps"]),
                float(hparams.get("wd", 0.0)),
                h_norm,
            )
        elif optimizer in ("Muon", "MuonH"):
            muon_step(
                weight,
                grad,
                momentum_buffer,
                base_lr,
                lr_factor,
                float(hparams["momentum"]),
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
                step + 1,
                bool(hparams["nesterov"]),
                str(h_norm),
            )
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        last_loss = float(loss.detach().item())
        should_log = config.log_every > 0 and (
            (step + 1) % config.log_every == 0 or step + 1 == steps
        )
        if should_log:
            print(
                f"candidate {candidate_idx + 1:05d}/{num_candidates:05d} "
                f"{optimizer} step {step + 1:05d}/{steps:05d} "
                f"loss={last_loss:.8g} lr={base_lr * lr_factor:.6g}",
                flush=True,
            )

    weight.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed_sec = time.time() - training_start_time
    eval_weight = effective_h_weight(
        weight, h_norm, ground_truth_row_norms, ground_truth_matrix_norm
    ).detach()
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
    if config.train_size != NUM_SAMPLES:
        raise ValueError(f"this experiment fixes --train-size to {NUM_SAMPLES}")

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
                "num_candidates": len(candidate_specs),
                "batch_sizes": BATCH_SIZES,
                "runs_per_setting": RUNS_PER_SETTING,
                "num_samples": NUM_SAMPLES,
                "lr_ranges": {str(key): value for key, value in LR_RANGES.items()},
                "lr_decay_ranges": {
                    str(key): value for key, value in LR_DECAY_RANGES.items()
                },
                "lr_power": LR_POWER,
                "beta1_momentum_values": BETA1_MOMENTUM_VALUES,
                "h_norms": H_NORMS,
                "wd": 0.0,
                "compile": config.compile,
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
