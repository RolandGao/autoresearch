"""Run the fixed SGDH learnable-softmax ablation study.

The experiment trains all 16 SGDH boolean variants across four batch sizes
using the provided per-batch median hyperparameters. Each optimizer and batch
size configuration is repeated five times with different seeds, and the final
logging reports mean +- std clean RMSE for every setting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch._dynamo
import torch.nn.functional as F


torch._dynamo.config.recompile_limit = 128

NUM_SAMPLES = 30_000
BATCH_SIZES = (8, 32, 128, 512)
RUNS_PER_SETTING = 5

OPTIMIZER_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "g_projection": False,
        "g_norm": False,
        "nesterov": False,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.01277244328, "lr_decay": 4.935193964, "momentum": 0.7802834571},
            32: {
                "lr": 0.04984072265,
                "lr_decay": 4.930870536,
                "momentum": 0.5110573541,
            },
            128: {
                "lr": 0.1633161669,
                "lr_decay": 5.031387552,
                "momentum": 0.1211513238,
            },
            512: {
                "lr": 0.5164882685,
                "lr_decay": 5.197982814,
                "momentum": 0.2040395917,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": False,
        "nesterov": False,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.01948860035, "lr_decay": 5.673267698, "momentum": 0.8198820362},
            32: {
                "lr": 0.0405227925,
                "lr_decay": 4.851177153,
                "momentum": 0.4227609333,
            },
            128: {
                "lr": 0.168829218,
                "lr_decay": 5.226893872,
                "momentum": 0.1150380844,
            },
            512: {
                "lr": 0.5839329062,
                "lr_decay": 5.030973648,
                "momentum": 0.2128585189,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": False,
        "nesterov": True,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.01339767227, "lr_decay": 5.1671711, "momentum": 0.8515824501},
            32: {
                "lr": 0.04984318826,
                "lr_decay": 5.410940882,
                "momentum": 0.6385174932,
            },
            128: {
                "lr": 0.2016143443,
                "lr_decay": 5.787030157,
                "momentum": 0.1388157923,
            },
            512: {
                "lr": 0.6067188119,
                "lr_decay": 5.341998024,
                "momentum": 0.2528993598,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": False,
        "nesterov": True,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.01056263618, "lr_decay": 4.96085181, "momentum": 0.8845369472},
            32: {
                "lr": 0.04768729738,
                "lr_decay": 5.092425929,
                "momentum": 0.4936145246,
            },
            128: {
                "lr": 0.173659043,
                "lr_decay": 4.959041282,
                "momentum": 0.1572948204,
            },
            512: {
                "lr": 0.5281132623,
                "lr_decay": 4.799819204,
                "momentum": 0.04465116588,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": True,
        "nesterov": False,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.04348873329, "lr_decay": 5.699757182, "momentum": 0.7500311653},
            32: {
                "lr": 0.03433407801,
                "lr_decay": 4.260182806,
                "momentum": 0.254712266,
            },
            128: {
                "lr": 0.1333926559,
                "lr_decay": 5.522785145,
                "momentum": 0.437045969,
            },
            512: {
                "lr": 0.4537396988,
                "lr_decay": 5.381339976,
                "momentum": 0.275487969,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": True,
        "nesterov": False,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.07447155913, "lr_decay": 5.795533712, "momentum": 0.2535107959},
            32: {
                "lr": 0.04071851047,
                "lr_decay": 4.534846735,
                "momentum": 0.465314808,
            },
            128: {
                "lr": 0.146287317,
                "lr_decay": 5.110504119,
                "momentum": 0.2954081905,
            },
            512: {
                "lr": 0.4806663508,
                "lr_decay": 5.392827619,
                "momentum": 0.2681712613,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": True,
        "nesterov": True,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.09898013441, "lr_decay": 5.376365701, "momentum": 0.3273779544},
            32: {
                "lr": 0.04982233784,
                "lr_decay": 4.452280737,
                "momentum": 0.2841602504,
            },
            128: {
                "lr": 0.1710950601,
                "lr_decay": 5.243666849,
                "momentum": 0.2332032082,
            },
            512: {
                "lr": 0.4423097546,
                "lr_decay": 5.53725425,
                "momentum": 0.3409937918,
            },
        },
    },
    {
        "g_projection": False,
        "g_norm": True,
        "nesterov": True,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.1109527937, "lr_decay": 5.490359577, "momentum": 0.3732062195},
            32: {
                "lr": 0.05518987659,
                "lr_decay": 4.47601835,
                "momentum": 0.08110259586,
            },
            128: {
                "lr": 0.1795663656,
                "lr_decay": 5.254453231,
                "momentum": 0.4996262203,
            },
            512: {
                "lr": 0.5817912197,
                "lr_decay": 5.49235543,
                "momentum": 0.2052869147,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": False,
        "nesterov": False,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.01492869594, "lr_decay": 5.386949158, "momentum": 0.7587509663},
            32: {
                "lr": 0.05338747201,
                "lr_decay": 5.240492332,
                "momentum": 0.2826106235,
            },
            128: {
                "lr": 0.1568986108,
                "lr_decay": 5.227436541,
                "momentum": 0.2921177614,
            },
            512: {
                "lr": 0.4793553177,
                "lr_decay": 5.039097221,
                "momentum": 0.2494080584,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": False,
        "nesterov": False,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.01077346979, "lr_decay": 4.847196315, "momentum": 0.8272235062},
            32: {
                "lr": 0.04753873653,
                "lr_decay": 5.073255392,
                "momentum": 0.26616753,
            },
            128: {
                "lr": 0.1397355806,
                "lr_decay": 4.8100477,
                "momentum": 0.2552222986,
            },
            512: {
                "lr": 0.7286169843,
                "lr_decay": 5.527746503,
                "momentum": 0.171482136,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": False,
        "nesterov": True,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.01748660264, "lr_decay": 4.980364942, "momentum": 0.7624815689},
            32: {
                "lr": 0.04442707274,
                "lr_decay": 5.486521506,
                "momentum": 0.5482385537,
            },
            128: {
                "lr": 0.2065866386,
                "lr_decay": 4.788898397,
                "momentum": 0.3944818342,
            },
            512: {
                "lr": 0.491243496,
                "lr_decay": 4.923746937,
                "momentum": 0.1795051719,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": False,
        "nesterov": True,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.01571638453, "lr_decay": 4.601125239, "momentum": 0.7318932433},
            32: {
                "lr": 0.04656199172,
                "lr_decay": 4.478073053,
                "momentum": 0.6607527284,
            },
            128: {
                "lr": 0.1767496134,
                "lr_decay": 5.080348912,
                "momentum": 0.3536064499,
            },
            512: {
                "lr": 0.5874815206,
                "lr_decay": 5.337923241,
                "momentum": 0.3184433038,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": True,
        "nesterov": False,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.07283710412, "lr_decay": 5.790746946, "momentum": 0.2799906765},
            32: {
                "lr": 0.04347903186,
                "lr_decay": 4.188731235,
                "momentum": 0.346770669,
            },
            128: {
                "lr": 0.1616251713,
                "lr_decay": 4.887228778,
                "momentum": 0.202780528,
            },
            512: {
                "lr": 0.5383700919,
                "lr_decay": 5.804914318,
                "momentum": 0.1583728512,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": True,
        "nesterov": False,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.07333567631, "lr_decay": 5.790919205, "momentum": 0.2507250729},
            32: {
                "lr": 0.04435161192,
                "lr_decay": 4.864731647,
                "momentum": 0.3270756055,
            },
            128: {
                "lr": 0.1456262444,
                "lr_decay": 5.028420774,
                "momentum": 0.3050736239,
            },
            512: {
                "lr": 0.4459636163,
                "lr_decay": 5.227256787,
                "momentum": 0.4114768582,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": True,
        "nesterov": True,
        "m_projection": False,
        "batch_params": {
            8: {"lr": 0.1090883446, "lr_decay": 5.690092638, "momentum": 0.2134277703},
            32: {
                "lr": 0.04061476493,
                "lr_decay": 4.643555196,
                "momentum": 0.3994464593,
            },
            128: {
                "lr": 0.1789848719,
                "lr_decay": 5.250829109,
                "momentum": 0.2280288879,
            },
            512: {
                "lr": 0.4437857476,
                "lr_decay": 4.896766312,
                "momentum": 0.3972373522,
            },
        },
    },
    {
        "g_projection": True,
        "g_norm": True,
        "nesterov": True,
        "m_projection": True,
        "batch_params": {
            8: {"lr": 0.1072763768, "lr_decay": 5.684955597, "momentum": 0.2444523365},
            32: {
                "lr": 0.05357844465,
                "lr_decay": 4.464117632,
                "momentum": 0.264829831,
            },
            128: {
                "lr": 0.1428085867,
                "lr_decay": 5.026764129,
                "momentum": 0.5304127547,
            },
            512: {
                "lr": 0.5212321892,
                "lr_decay": 5.039943998,
                "momentum": 0.07174007342,
            },
        },
    },
)


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
    compile: bool = True
    device: str = "cuda"


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
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile and use the eager tensor path.",
    )
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


def bool_flag(value: bool) -> str:
    return "T" if value else "F"


def optimizer_key(hparams: dict[str, Any]) -> str:
    return (
        f"g_projection={bool_flag(bool(hparams['g_projection']))},"
        f"g_norm={bool_flag(bool(hparams['g_norm']))},"
        f"nesterov={bool_flag(bool(hparams['nesterov']))},"
        f"m_projection={bool_flag(bool(hparams['m_projection']))}"
    )


def setting_key(hparams: dict[str, Any]) -> str:
    return f"{optimizer_key(hparams)},batch_size={int(hparams['batch_size'])}"


def rounded_steps_for_num_samples(num_samples: int, batch_size: int) -> int:
    return max(1, int(round(num_samples / batch_size)))


def build_experiment_settings(config: Config) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for variant in OPTIMIZER_VARIANTS:
        for batch_size in BATCH_SIZES:
            batch_params = variant["batch_params"][batch_size]
            settings.append(
                {
                    "variant": "SGDH",
                    "g_projection": variant["g_projection"],
                    "g_norm": variant["g_norm"],
                    "nesterov": variant["nesterov"],
                    "m_projection": variant["m_projection"],
                    "batch_size": batch_size,
                    "steps": rounded_steps_for_num_samples(config.train_size, batch_size),
                    "num_samples": config.train_size,
                    "sample_mode": "fixed_cycle",
                    "lr_schedule": "exp_power",
                    "lr_power": 1.0,
                    **batch_params,
                }
            )
    return settings


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
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    target_probs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    logits = softmax_scale * F.linear(x_batch, weight).float()
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().mean().sqrt()


@torch.compile(dynamic=True, fullgraph=True)
def probability_rmse_loss_compiled(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    target_probs: torch.Tensor,
    softmax_scale_t: torch.Tensor,
) -> torch.Tensor:
    logits = softmax_scale_t * F.linear(x_batch, weight).float()
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().mean().sqrt()


@torch.compile(dynamic=True, fullgraph=True)
def softmax_probs_compiled(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    softmax_scale_t: torch.Tensor,
) -> torch.Tensor:
    logits = softmax_scale_t * F.linear(x_batch, weight).float()
    return logits.softmax(dim=1)


@torch.compile(dynamic=False, fullgraph=True)
def sgdh_step_compiled(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr_t: torch.Tensor,
    momentum_t: torch.Tensor,
    lr_factor_t: torch.Tensor,
    g_projection: bool,
    g_norm: bool,
    nesterov: bool,
    m_projection: bool,
) -> None:
    g = grad
    if g_projection:
        g = project_rows(g, weight)
    if g_norm:
        g = row_normalize_update(g)

    momentum_buffer.mul_(momentum_t).add_(g)
    if m_projection:
        momentum_buffer.copy_(project_rows(momentum_buffer, weight))

    update = g + momentum_t * momentum_buffer if nesterov else momentum_buffer
    update_norm = update.norm(dim=1, keepdim=True)
    normalized_update = update / update_norm.clamp_min(1e-12)
    normalized_update = torch.where(
        update_norm > 0, normalized_update, torch.zeros_like(normalized_update)
    )
    weight.add_(normalized_update * (-lr_t * lr_factor_t))
    weight.copy_(weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def sgdh_step_eager(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    momentum: float,
    lr_factor: float,
    g_projection: bool,
    g_norm: bool,
    nesterov: bool,
    m_projection: bool,
) -> None:
    if g_projection:
        grad = project_rows(grad, weight)
    if g_norm:
        grad = row_normalize_update(grad)

    momentum_buffer.mul_(momentum).add_(grad)
    if m_projection:
        momentum_buffer.copy_(project_rows(momentum_buffer, weight))

    update = grad + momentum * momentum_buffer if nesterov else momentum_buffer
    weight.add_(row_normalize_update(update), alpha=-lr * lr_factor)
    weight.copy_(normalize_rows(weight))


def lr_schedule_factor(step: int, steps: int, hparams: dict[str, Any]) -> float:
    progress = step / max(1, steps)
    return math.exp(-float(hparams["lr_decay"]) * progress)


def build_fixed_cycle_orders(
    train_size: int, device: torch.device, seed: int
) -> dict[int, torch.Tensor]:
    orders: dict[int, torch.Tensor] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 10_000)
    base_permutation = torch.randperm(train_size, device=device, generator=generator)

    for batch_size in BATCH_SIZES:
        steps = rounded_steps_for_num_samples(train_size, batch_size)
        needed = steps * batch_size
        repeats = math.ceil(needed / train_size)
        orders[batch_size] = base_permutation.repeat(repeats)[:needed]

    return orders


@torch.no_grad()
def evaluate_clean_rmse_tensor(
    weight: torch.Tensor,
    x: torch.Tensor,
    clean_target_probs: torch.Tensor,
    config: Config,
) -> float:
    squared_error_sum = 0.0
    num_values = 0
    softmax_scale_t = torch.tensor(
        config.softmax_scale, dtype=torch.float32, device="cpu"
    )

    for start in range(0, x.size(0), config.eval_batch_size):
        end = min(start + config.eval_batch_size, x.size(0))
        if config.compile:
            output_probs = softmax_probs_compiled(
                weight, x[start:end], softmax_scale_t
            )
        else:
            logits = config.softmax_scale * F.linear(x[start:end], weight).float()
            output_probs = logits.softmax(dim=1)
        diff = output_probs - clean_target_probs[start:end]
        squared_error_sum += diff.square().sum().item()
        num_values += diff.numel()

    return math.sqrt(squared_error_sum / num_values)


def summarize_metric(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_setting_summaries(
    summaries: list[dict[str, Any]], settings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries_by_key: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        summaries_by_key.setdefault(str(summary["setting_key"]), []).append(summary)

    aggregates: list[dict[str, Any]] = []
    for setting in settings:
        key = setting_key(setting)
        runs = summaries_by_key[key]
        clean_rmses = [float(run["clean_rmse"]) for run in runs]
        train_losses = [float(run["final_train_loss"]) for run in runs]
        durations = [float(run["duration_sec"]) for run in runs]
        clean_rmse_mean, clean_rmse_std = summarize_metric(clean_rmses)
        train_loss_mean, train_loss_std = summarize_metric(train_losses)
        duration_mean, duration_std = summarize_metric(durations)
        aggregates.append(
            {
                **setting,
                "optimizer_key": optimizer_key(setting),
                "setting_key": key,
                "num_runs": len(runs),
                "clean_rmse_mean": clean_rmse_mean,
                "clean_rmse_std": clean_rmse_std,
                "clean_rmse_min": min(clean_rmses),
                "clean_rmse_max": max(clean_rmses),
                "final_train_loss_mean": train_loss_mean,
                "final_train_loss_std": train_loss_std,
                "duration_sec_mean": duration_mean,
                "duration_sec_std": duration_std,
                "repeat_seeds": [int(run["repeat_seed"]) for run in runs],
            }
        )
    return aggregates


def print_aggregate_summary(aggregates: list[dict[str, Any]]) -> None:
    current_optimizer_key: str | None = None
    for aggregate in aggregates:
        optimizer_label = str(aggregate["optimizer_key"])
        if optimizer_label != current_optimizer_key:
            current_optimizer_key = optimizer_label
            print(f"SUMMARY {optimizer_label}", flush=True)
        print(
            "SUMMARY "
            f"  batch_size={aggregate['batch_size']}: "
            f"clean_rmse={aggregate['clean_rmse_mean']:.10g} +- "
            f"{aggregate['clean_rmse_std']:.10g} "
            f"final_train_loss={aggregate['final_train_loss_mean']:.10g} +- "
            f"{aggregate['final_train_loss_std']:.10g}",
            flush=True,
        )


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
            {
                "batch_sizes": BATCH_SIZES,
                "optimizer_variant_count": len(OPTIMIZER_VARIANTS),
                "setting_count": len(OPTIMIZER_VARIANTS) * len(BATCH_SIZES),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    scalar_device = torch.device("cpu")
    softmax_scale_t = torch.tensor(
        config.softmax_scale, dtype=torch.float32, device=scalar_device
    )
    lr_t = torch.tensor(0.01, dtype=torch.float32, device=scalar_device)
    lr_factor_t = torch.tensor(1.0, dtype=torch.float32, device=scalar_device)
    momentum_t = torch.tensor(0.5, dtype=torch.float32, device=scalar_device)

    for batch_size in BATCH_SIZES:
        batch_idx = batch_orders[batch_size][:batch_size]
        x_batch = x[batch_idx]
        target_batch = noisy_target_probs[batch_idx]
        for g_projection in (False, True):
            for g_norm in (False, True):
                for nesterov in (False, True):
                    for m_projection in (False, True):
                        weight = (
                            initial_weight.detach().clone().to(device).requires_grad_()
                        )
                        loss = probability_rmse_loss_compiled(
                            weight, x_batch, target_batch, softmax_scale_t
                        )
                        loss.backward()
                        grad = weight.grad
                        if grad is None:
                            raise RuntimeError(
                                "compile warmup did not produce a weight gradient"
                            )
                        with torch.no_grad():
                            sgdh_step_compiled(
                                weight,
                                grad,
                                torch.zeros_like(weight),
                                lr_t,
                                momentum_t,
                                lr_factor_t,
                                g_projection,
                                g_norm,
                                nesterov,
                                m_projection,
                            )

    eval_shapes = {min(config.eval_batch_size, x.size(0))}
    final_eval_batch = x.size(0) % config.eval_batch_size
    if final_eval_batch:
        eval_shapes.add(final_eval_batch)
    for eval_batch_size in sorted(eval_shapes):
        _ = softmax_probs_compiled(
            initial_weight.detach(), x[:eval_batch_size], softmax_scale_t
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time
    print(
        "COMPILE_WARMUP_DONE "
        + json.dumps({"elapsed_sec": elapsed_sec}, sort_keys=True),
        flush=True,
    )


def train_one_run(
    hparams: dict[str, Any],
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    clean_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    batch_orders: dict[int, torch.Tensor],
    config: Config,
    device: torch.device,
    run_number: int,
    total_runs: int,
    repeat_idx: int,
    repeat_seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    weight = initial_weight.detach().clone().to(device).requires_grad_(True)
    momentum_buffer = torch.zeros_like(weight)

    scalar_device = torch.device("cpu")
    softmax_scale_t = torch.tensor(
        config.softmax_scale, dtype=torch.float32, device=scalar_device
    )
    lr_t = torch.tensor(float(hparams["lr"]), dtype=torch.float32, device=scalar_device)
    lr_factor_t = torch.tensor(1.0, dtype=torch.float32, device=scalar_device)
    momentum_t = torch.tensor(
        float(hparams["momentum"]), dtype=torch.float32, device=scalar_device
    )

    last_loss = float("nan")
    base_lr = float(hparams["lr"])
    momentum = float(hparams["momentum"])
    steps = int(hparams["steps"])
    batch_size = int(hparams["batch_size"])
    batch_order = batch_orders[batch_size]
    g_projection = bool(hparams["g_projection"])
    g_norm = bool(hparams["g_norm"])
    nesterov = bool(hparams["nesterov"])
    m_projection = bool(hparams["m_projection"])

    training_start_time = time.time()
    for step in range(steps):
        lr_factor = lr_schedule_factor(step, steps, hparams)
        batch_start = step * batch_size
        batch_idx = batch_order[batch_start : batch_start + batch_size]
        weight.grad = None
        if config.compile:
            lr_factor_t.fill_(lr_factor)
            loss = probability_rmse_loss_compiled(
                weight, x[batch_idx], noisy_target_probs[batch_idx], softmax_scale_t
            )
        else:
            loss = probability_rmse_loss(
                weight, x[batch_idx], noisy_target_probs[batch_idx], config.softmax_scale
            )

        loss.backward()
        grad = weight.grad
        if grad is None:
            raise RuntimeError("loss did not produce a weight gradient")
        if config.compile:
            with torch.no_grad():
                sgdh_step_compiled(
                    weight,
                    grad,
                    momentum_buffer,
                    lr_t,
                    momentum_t,
                    lr_factor_t,
                    g_projection,
                    g_norm,
                    nesterov,
                    m_projection,
                )
        else:
            sgdh_step_eager(
                weight,
                grad,
                momentum_buffer,
                base_lr,
                momentum,
                lr_factor,
                g_projection,
                g_norm,
                nesterov,
                m_projection,
            )

        last_loss = float(loss.detach().item())
        should_log = config.log_every > 0 and (
            (step + 1) % config.log_every == 0 or step + 1 == steps
        )
        if should_log:
            print(
                f"run {run_number:03d}/{total_runs:03d} "
                f"repeat {repeat_idx + 1:02d}/{RUNS_PER_SETTING:02d} "
                f"SGDH step {step + 1:05d}/{steps:05d} "
                f"loss_rmse={last_loss:.8g} lr={base_lr * lr_factor:.6g}",
                flush=True,
            )

    weight.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed_sec = time.time() - training_start_time
    clean_rmse = evaluate_clean_rmse_tensor(
        weight.detach(), x, clean_target_probs, config
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
    weight_norms = weight.detach().norm(dim=1)
    summary = {
        **hparams,
        "run_number": run_number,
        "repeat_idx": repeat_idx,
        "repeat_seed": repeat_seed,
        "actual_samples": steps * batch_size,
        "compiled": config.compile,
        "final_train_loss": last_loss,
        "clean_rmse": clean_rmse,
        "clean_train_rmse": clean_rmse,
        "target_met": clean_rmse <= config.target_rmse,
        "optimizer_key": optimizer_key(hparams),
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
    if config.train_size != NUM_SAMPLES:
        raise ValueError(f"this experiment fixes --train-size to {NUM_SAMPLES}")
    if config.softmax_scale != 10.0:
        raise ValueError("this experiment fixes --softmax-scale to 10.0")
    if config.ground_truth_softmax_scale != 10.0:
        raise ValueError("this experiment fixes --ground-truth-softmax-scale to 10.0")

    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    settings = build_experiment_settings(config)
    total_runs = len(settings) * RUNS_PER_SETTING

    print(
        "CONFIG "
        + json.dumps({**asdict(config), "device": str(device)}, sort_keys=True),
        flush=True,
    )
    print(
        "EXPERIMENT_PLAN "
        + json.dumps(
            {
                "optimizer": "SGDH",
                "num_optimizer_variants": len(OPTIMIZER_VARIANTS),
                "batch_sizes": BATCH_SIZES,
                "settings_per_repeat": len(settings),
                "runs_per_setting": RUNS_PER_SETTING,
                "total_runs": total_runs,
                "num_samples": NUM_SAMPLES,
                "compile": config.compile,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for repeat_idx in range(RUNS_PER_SETTING):
        repeat_seed = config.seed + repeat_idx
        set_seed(repeat_seed)
        print(
            "REPEAT_START "
            + json.dumps(
                {
                    "repeat_idx": repeat_idx,
                    "repeat_number": repeat_idx + 1,
                    "num_repeats": RUNS_PER_SETTING,
                    "repeat_seed": repeat_seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        x, clean_target_probs, noisy_target_probs = build_problem(config, device)
        initial_weight = normalize_rows(
            torch.randn(config.num_classes, config.input_dim, device=device)
        )
        batch_orders = build_fixed_cycle_orders(config.train_size, device, repeat_seed)

        if repeat_idx == 0:
            warmup_compiled_training(
                config, x, noisy_target_probs, initial_weight, batch_orders, device
            )

        for setting in settings:
            run_number = len(summaries) + 1
            print(
                "RUN_START "
                + json.dumps(
                    {
                        **setting,
                        "optimizer_key": optimizer_key(setting),
                        "setting_key": setting_key(setting),
                        "run_number": run_number,
                        "total_runs": total_runs,
                        "repeat_idx": repeat_idx,
                        "repeat_seed": repeat_seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            summary = train_one_run(
                setting,
                x,
                noisy_target_probs,
                clean_target_probs,
                initial_weight,
                batch_orders,
                config,
                device,
                run_number,
                total_runs,
                repeat_idx,
                repeat_seed,
            )
            summaries.append(summary)

    aggregates = aggregate_setting_summaries(summaries, settings)
    print_aggregate_summary(aggregates)
    print(f"FINAL_SUMMARY {json.dumps(aggregates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
