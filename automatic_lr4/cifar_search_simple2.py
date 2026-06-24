"""
CIFAR-10 learning-rate and momentum search built around the Airbench94 Muon model.
"""

#############################################
#                  Setup                    #
#############################################

import copy
import os
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from types import SimpleNamespace

from math import ceil, floor, isclose, log, log10

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


def _cuda_capability():
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


CUDA_CAPABILITY = _cuda_capability()
IS_AMPERE_OR_NEWER = CUDA_CAPABILITY[0] >= 8
USE_CUDNN_BENCHMARK = False
USE_TF32 = IS_AMPERE_OR_NEWER

torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.matmul.allow_tf32 = USE_TF32

USE_COMPILED_MUON = False
MUON_DTYPE = torch.bfloat16
TRAINING_SEED = 0


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


#############################################
#               Muon optimizer              #
#############################################


def zeropower_via_newtonschulz5(G, steps=3, eps=0):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(MUON_DTYPE)
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


if USE_COMPILED_MUON:
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())  # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(
                    g.shape
                )  # whiten the update
                p.data.add_(update, alpha=-lr)  # take a step


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    y = torch.arange(crop_size, device=images.device).view(1, crop_size, 1)
    x = torch.arange(crop_size, device=images.device).view(1, 1, crop_size)
    y_idx = y + (r + shifts[:, 0]).view(-1, 1, 1)
    x_idx = x + (r + shifts[:, 1]).view(-1, 1, 1)
    gathered_rows = images.gather(
        2, y_idx[:, None, :, :].expand(-1, images.size(1), -1, images.size(-1))
    )
    return gathered_rows.gather(
        3, x_idx[:, None, :, :].expand(-1, images.size(1), crop_size, crop_size)
    )


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels = data["images"], data["labels"]
        # It's faster to load+process uint8 data than to load preprocessed data
        self.images = (
            (self.images.float() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}  # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def _ensure_proc_images(self):
        if "norm" in self.proc_images:
            return
        images = self.proc_images["norm"] = self.normalize(self.images)
        if self.aug.get("flip", False):
            images = self.proc_images["flip"] = batch_flip_lr(images)
        pad = self.aug.get("translate", 0)
        if pad > 0:
            self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

    def normalized_images(self):
        self._ensure_proc_images()
        return self.proc_images["norm"]

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):

        if self.epoch == 0:
            self._ensure_proc_images()

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(
            3, whiten_width, whiten_kernel_size, padding=0, bias=True
        )
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.data[:] = torch.cat(
            (eigenvectors_scaled, -eigenvectors_scaled)
        )

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


############################################
#                 Logging                  #
############################################


def format_log_value(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return "%.6g" % value
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def log_event(name, **fields):
    field_parts = [
        "%s=%s" % (key, format_log_value(value)) for key, value in fields.items()
    ]
    print(" ".join([name, *field_parts]), flush=True)


def log_train_hparams(
    run,
    interval_index,
    start_step,
    completed_steps,
    total_steps,
    result,
    muon_nesterov,
):
    log_event(
        "train_hparams",
        run=run,
        interval=interval_index,
        start_step=start_step,
        completed_steps=completed_steps,
        total_steps=total_steps,
        **hparam_log_fields(result),
        muon_nesterov=muon_nesterov,
    )


def log_train_loss(run, interval_index, step, loss):
    log_event(
        "train_loss",
        run=run,
        interval=interval_index,
        step=step,
        loss=loss,
    )


def log_interval_boundary_eval(run, interval_index, step, total_steps, tta_val_acc):
    log_event(
        "interval_boundary_eval",
        run=run,
        interval=interval_index,
        step=step,
        total_steps=total_steps,
        tta_val_acc=tta_val_acc,
    )


def hparam_log_fields(values, names=None):
    names = SEARCH_HPARAMS.keys() if names is None else names
    return {name: values[name] for name in names if name in values}


def copy_fields(source, fields):
    return {key: source[key] for key in fields}


def pack(values, fields):
    return {key: values[key] for key in fields.split()}


def namespace(values, fields, **extra):
    return SimpleNamespace(**pack(values, fields), **extra)


def finite(value):
    return bool(torch.isfinite(torch.as_tensor(value)).item())


RUN_SUMMARY_SPECS = "\nBatch size:          %d|batch_size\nTrain epochs:        %.3g|train_epochs\nTrain steps:         %d|train_steps\nN steps:             %d|n_steps\nM cooldown steps:    %d|m_steps\nSearch hparams:      %s|search_names_text\nCooldown hparams:    %s|cooldown_search_names_text\nMuon nesterov:       %s|muon_nesterov\n".strip().splitlines()
RUN_FOOTER_SPECS = "\nVal acc:             %.4f|val_acc\nTTA val acc:         %.4f|tta_val_acc\nRun seconds:         %.3f|run_seconds\n".strip().splitlines()


def print_summary_specs(specs, result):
    for spec in specs:
        format_string, key = spec.split("|")
        print(format_string % result[key])


def log_run_summary(result):
    print_summary_specs(RUN_SUMMARY_SPECS, result)
    print_summary_specs(RUN_FOOTER_SPECS, result)


def hparam_text(values):
    return " ".join(
        "%s=%s" % (name, format_log_value(values[name]))
        for name in SEARCH_HPARAMS
        if name in values
    )


def log_line(text):
    print(text, flush=True)


def log_main_hparams(result):
    prefix = "main hparams: %s" % hparam_text(result)
    if result.get("blocked", False):
        log_line("%s -> blocked" % prefix)
        return
    metrics = ["main=%s" % format_log_value(result["main_tta_val_acc"])]
    cooldown_result = result.get("cooldown_result")
    if cooldown_result is not None:
        metrics.append(
            "best_cooldown=%s" % format_log_value(cooldown_result["tta_val_acc"])
        )
    log_line("%s %s" % (prefix, ", ".join(metrics)))


def log_candidate_result(result):
    prefix = hparam_text(result)
    if result.get("blocked", False):
        log_line("%s -> blocked" % prefix)
        return
    log_line("%s -> tta_val_acc=%s" % (prefix, format_log_value(result["tta_val_acc"])))


def log_candidate_results(results):
    for result in results:
        log_candidate_result(result)


def log_search_path(path):
    for step, result in enumerate(path):
        prefix = "search_path step=%d %s" % (step, hparam_text(result))
        if result.get("blocked", False):
            log_line("%s blocked=True" % prefix)
            continue
        log_line(
            "%s tta_val_acc=%s" % (prefix, format_log_value(result["tta_val_acc"]))
        )


############################################
#               Evaluation                 #
############################################


def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [
            infer_mirror(inputs_translate, net)
            for inputs_translate in inputs_translate_list
        ]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.train()
    test_images = loader.normalized_images()
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.inference_mode():
        return torch.cat(
            [infer_fn(inputs, model) for inputs in test_images.split(2000)]
        )


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def evaluate_tta_val_acc(model, loader):
    return evaluate(model, loader, tta_level=2)


############################################
#                Training                  #
############################################

TRAIN_EPOCHS = 8
LABEL_SMOOTHING = 0.2
SEARCH_STEP_CONFIGS = [(40, 40)]
PRINT_OUTPUT_FILENAME = "cifar_search_4hparam.log"
LR_SEARCH_FACTOR = 0.6
LR_SEARCH_SIG_FIGS = 2
LR_SEARCH_MAX_MOVES = 60
MOMENTUM_SEARCH_VALUES = [round(i / 10, 1) for i in range(10)] + [0.95, 0.99]
INITIAL_MOMENTUM = 0.6
MUON_NESTEROV = False
RUN_CONFIGS = [
    dict(batch_size=2000),
]


@dataclass(frozen=True)
class SearchHparam:
    kind: str
    initial_value: float = None
    search: bool = False
    cooldown_search: bool = False
    factor: float = None
    values: tuple = ()


SEARCH_HPARAMS = {
    "muon_lr": SearchHparam(
        kind="log_lr",
        initial_value=0.2,
        search=True,
        cooldown_search=True,
        factor=LR_SEARCH_FACTOR,
    ),
    "muon_momentum": SearchHparam(
        kind="choice",
        initial_value=INITIAL_MOMENTUM,
        search=True,
        cooldown_search=False,
        values=tuple(MOMENTUM_SEARCH_VALUES),
    ),
    "bias_lr": SearchHparam(
        kind="log_lr",
        initial_value=104,
        search=True,
        cooldown_search=True,
        factor=LR_SEARCH_FACTOR,
    ),
    "head_lr": SearchHparam(
        kind="log_lr",
        initial_value=1340,
        search=True,
        cooldown_search=True,
        factor=LR_SEARCH_FACTOR,
    ),
}


def rounded_lr(value):
    if value == 0:
        return 0.0
    return round(value, LR_SEARCH_SIG_FIGS - 1 - floor(log10(abs(value))))


def lr_from_k(k, initial_value, factor):
    return rounded_lr(initial_value * factor**k)


def nearest_lr_k(lr, initial_value, factor):
    if lr <= 0:
        raise ValueError(f"LR must be positive, got {lr}")
    return int(round(log(lr / initial_value) / log(factor)))


def nearest_momentum_index(momentum, values=MOMENTUM_SEARCH_VALUES):
    for index, search_momentum in enumerate(values):
        if isclose(momentum, search_momentum, rel_tol=0.0, abs_tol=1e-12):
            return index
    raise ValueError("momentum must be in %s, got %s" % (values, momentum))


def active_search_hparam_names():
    return tuple(name for name, spec in SEARCH_HPARAMS.items() if spec.search)


def cooldown_search_hparam_names():
    return tuple(name for name, spec in SEARCH_HPARAMS.items() if spec.cooldown_search)


def hparam_from_state(name, state):
    spec = SEARCH_HPARAMS[name]
    if spec.kind == "log_lr":
        return lr_from_k(state, spec.initial_value, spec.factor)
    if spec.kind == "choice":
        return spec.values[state]
    raise ValueError(f"Unrecognized hparam kind: {spec.kind}")


def nearest_hparam_state(name, value):
    spec = SEARCH_HPARAMS[name]
    if spec.kind == "log_lr":
        return nearest_lr_k(value, spec.initial_value, spec.factor)
    if spec.kind == "choice":
        return nearest_momentum_index(value, spec.values)
    raise ValueError(f"Unrecognized hparam kind: {spec.kind}")


def initial_hparams_from_cfg(cfg):
    hparams = {}
    for name, spec in SEARCH_HPARAMS.items():
        hparams[name] = getattr(cfg, f"initial_{name}", spec.initial_value)
    return hparams


def point_from_hparams(hparams, search_names):
    return tuple(nearest_hparam_state(name, hparams[name]) for name in search_names)


def point_to_hparams(point, search_names, fixed_hparams):
    hparams = dict(fixed_hparams)
    hparams.update(
        {
            name: hparam_from_state(name, state)
            for name, state in zip(search_names, point)
        }
    )
    return hparams


def point_state(point, search_names, name):
    if name not in search_names:
        return None
    return point[search_names.index(name)]


def format_hparam_names(names):
    return ",".join(names) if names else "none"


class FullDatasetBatchStream:
    def __init__(self, loader):
        self.loader = loader
        self.images = None
        self.indices = None
        self.batch_index = 0

    def _rng_state(self):
        return dict(
            cpu=torch.random.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    def _load_rng_state(self, state):
        torch.random.set_rng_state(state["cpu"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def state_dict(self):
        return dict(
            loader_epoch=self.loader.epoch,
            images=self.images,
            indices=self.indices,
            batch_index=self.batch_index,
            rng=self._rng_state(),
        )

    def load_state_dict(self, state):
        self.loader.epoch = state["loader_epoch"]
        self.images = state["images"]
        self.indices = state["indices"]
        self.batch_index = state["batch_index"]
        self._load_rng_state(state["rng"])

    def _prepare_epoch(self):
        if self.loader.epoch == 0:
            self.loader._ensure_proc_images()
        if self.loader.aug.get("translate", 0) > 0:
            images = batch_crop(
                self.loader.proc_images["pad"], self.loader.images.shape[-2]
            )
        elif self.loader.aug.get("flip", False):
            images = self.loader.proc_images["flip"]
        else:
            images = self.loader.proc_images["norm"]
        if self.loader.aug.get("flip", False) and self.loader.epoch % 2 == 1:
            images = images.flip(-1)
        self.loader.epoch += 1
        self.images = images
        self.indices = (torch.randperm if self.loader.shuffle else torch.arange)(
            len(images), device=images.device
        )
        self.batch_index = 0

    def next_batch(self):
        if self.images is None or self.batch_index >= len(self.loader):
            self._prepare_epoch()
        start = self.batch_index * self.loader.batch_size
        end = (self.batch_index + 1) * self.loader.batch_size
        idxs = self.indices[start:end]
        self.batch_index += 1
        return self.images[idxs], self.loader.labels[idxs]


def make_optimizers(model, cfg):
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for (n, p) in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(
            params=[model.whiten.bias],
            lr=cfg.bias_lr,
            weight_decay=0,
            lr_name="bias_lr",
        ),
        dict(
            params=norm_biases,
            lr=cfg.bias_lr,
            weight_decay=0,
            lr_name="bias_lr",
        ),
        dict(
            params=[model.head.weight],
            lr=cfg.head_lr,
            weight_decay=0,
            lr_name="head_lr",
        ),
    ]
    sgd_optimizer = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    muon_optimizer = Muon(
        filter_params,
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
    )
    return sgd_optimizer, muon_optimizer


def set_muon_hparams(muon_optimizer, muon_lr, muon_momentum, muon_nesterov):
    muon_lr = rounded_lr(muon_lr)
    for group in muon_optimizer.param_groups:
        group["lr"] = muon_lr
        group["momentum"] = muon_momentum
        group["nesterov"] = muon_nesterov
    return muon_lr


def set_sgd_lrs(ctx, hparams):
    for group in ctx.sgd_optimizer.param_groups:
        group["lr"] = hparams[group["lr_name"]]


def snapshot_training_state(model, optimizers, batch_stream):
    return dict(
        model={
            name: value.detach().clone() for (name, value) in model.state_dict().items()
        },
        optimizers=[copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers],
        batch_stream=batch_stream.state_dict(),
    )


def load_training_state(model, optimizers, batch_stream, state):
    model.load_state_dict(state["model"])
    for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
        # Optimizer state tensors are installed by reference and then mutated by
        # optimizer.step(), so every replay/candidate needs its own copy.
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    batch_stream.load_state_dict(state["batch_stream"])
    model.zero_grad(set_to_none=True)


def train_one_step(ctx, hparams, global_step):
    set_muon_hparams(
        ctx.muon_optimizer,
        hparams["muon_lr"],
        hparams["muon_momentum"],
        ctx.muon_nesterov,
    )
    set_sgd_lrs(ctx, hparams)
    inputs, labels = ctx.batch_stream.next_batch()
    ctx.model.train()
    ctx.model.zero_grad(set_to_none=True)
    outputs = ctx.model(
        inputs, whiten_bias_grad=(global_step < ctx.whiten_bias_train_steps)
    )
    loss = F.cross_entropy(
        outputs, labels, label_smoothing=LABEL_SMOOTHING, reduction="mean"
    )
    loss.backward()
    for optimizer in ctx.optimizers:
        optimizer.step()
    ctx.model.zero_grad(set_to_none=True)
    return loss.item()


def train_interval(
    ctx,
    hparams,
    interval_steps,
    start_step,
    capture_end_state=True,
    capture_step_metrics=False,
):
    hparams = dict(hparams)
    hparams["muon_lr"] = rounded_lr(hparams["muon_lr"])
    losses = [] if capture_step_metrics else None
    last_train_loss = float("inf")
    completed_steps = 0
    for offset in range(interval_steps):
        step_loss = train_one_step(
            ctx,
            hparams,
            start_step + offset,
        )
        last_train_loss = step_loss
        if capture_step_metrics:
            losses.append(step_loss)
        completed_steps += 1
        if not finite(step_loss):
            break
    result = dict(
        last_train_loss=last_train_loss,
        completed_steps=completed_steps,
    )
    result.update(hparams)
    if capture_step_metrics:
        result["losses"] = losses
    if capture_end_state:
        result["end_state"] = snapshot_training_state(
            ctx.model, ctx.optimizers, ctx.batch_stream
        )
    return result


def interval_result_loss(result):
    return result["last_train_loss"]


def point_sort_key(point, search_names):
    log_states = [
        state
        for name, state in zip(search_names, point)
        if SEARCH_HPARAMS[name].kind == "log_lr"
    ]
    values = tuple(
        hparam_from_state(name, state) for name, state in zip(search_names, point)
    )
    return sum(abs(state) for state in log_states), values, point


def better_point(point, incumbent_point, results_by_point, search_names):
    if incumbent_point is None:
        return True
    tta_val_acc = results_by_point[point]["tta_val_acc"]
    incumbent_tta_val_acc = results_by_point[incumbent_point]["tta_val_acc"]
    return (-tta_val_acc, point_sort_key(point, search_names)) < (
        -incumbent_tta_val_acc,
        point_sort_key(incumbent_point, search_names),
    )


def better_tta_val_acc(point, incumbent_point, results_by_point):
    return (
        results_by_point[point]["tta_val_acc"]
        > results_by_point[incumbent_point]["tta_val_acc"]
    )


def point_with_state(point, index, state):
    return point[:index] + (state,) + point[index + 1 :]


def neighbor_states(name, state, step=1):
    spec = SEARCH_HPARAMS[name]
    if spec.kind == "log_lr":
        return [state - step, state + step]
    if spec.kind == "choice":
        states = []
        if state > 0:
            states.append(state - 1)
        if state + 1 < len(spec.values):
            states.append(state + 1)
        return states
    raise ValueError(f"Unrecognized hparam kind: {spec.kind}")


def lower_value_first(points, search_names, index):
    name = search_names[index]
    return sorted(points, key=lambda point: hparam_from_state(name, point[index]))


def uses_opposite_neighbor_block(name):
    return SEARCH_HPARAMS[name].kind == "log_lr"


def neighbor_point_groups(point, search_names):
    for index, name in enumerate(search_names):
        group = [
            point_with_state(point, index, state)
            for state in neighbor_states(name, point[index])
        ]
        if group:
            yield name, lower_value_first(group, search_names, index)


def neighbor_points(point, search_names):
    for _, group in neighbor_point_groups(point, search_names):
        yield from group


def best_neighbor_point(middle_point, results_by_point, search_names):
    best_point = middle_point
    for point in neighbor_points(middle_point, search_names):
        if better_point(point, best_point, results_by_point, search_names):
            best_point = point
    return best_point


def find_best_hparam_point(
    initial_point,
    search_names,
    evaluate,
    results_by_point,
    block,
):
    def evaluate_neighbors(middle_point):
        results = []
        for name, group in neighbor_point_groups(middle_point, search_names):
            first_point = group[0]
            results.append(evaluate(first_point, cooldown_seed_point=middle_point))
            if len(group) == 1:
                continue
            second_point = group[1]
            if uses_opposite_neighbor_block(name) and better_tta_val_acc(
                first_point, middle_point, results_by_point
            ):
                block(second_point)
                results.append(results_by_point[second_point])
                continue
            results.append(evaluate(second_point, cooldown_seed_point=middle_point))
        return results

    middle_point = initial_point
    center_path = [middle_point]
    evaluate(middle_point)
    initial_points = [middle_point]
    evaluate_neighbors(middle_point)
    initial_points.extend(
        point for point in neighbor_points(middle_point, search_names)
    )
    best_initial_point = middle_point
    for point in initial_points:
        if better_point(point, best_initial_point, results_by_point, search_names):
            best_initial_point = point
    if best_initial_point != middle_point:
        middle_point = best_initial_point
        center_path.append(middle_point)
    for _ in range(LR_SEARCH_MAX_MOVES):
        evaluate_neighbors(middle_point)
        next_point = best_neighbor_point(middle_point, results_by_point, search_names)
        if next_point == middle_point:
            break
        middle_point = next_point
        center_path.append(middle_point)
    else:
        log_event(
            "lr_momentum_search_warning",
            did_not_converge_within=LR_SEARCH_MAX_MOVES,
            using_point=middle_point,
            tta_val_acc=results_by_point[middle_point]["tta_val_acc"],
            loss=interval_result_loss(results_by_point[middle_point]),
        )
    return middle_point, center_path


BEST_RESULT_FIELDS = list(SEARCH_HPARAMS)


def add_best_result_fields(result, best_result):
    result.update(copy_fields(best_result, BEST_RESULT_FIELDS))


def point_states(point, search_names, names=None):
    if names is None:
        names = search_names
    return {
        name: point_state(point, search_names, name)
        for name in names
        if name in search_names
    }


def search_hparam_segment(
    ctx,
    initial_point,
    search_names,
    fixed_hparams,
    steps,
    start_step,
    start_state,
    interval_info=None,
):
    results_by_point = {}
    candidate_evaluations = []
    is_main_interval = interval_info is not None

    def evaluate(point, cooldown_seed_point=None):
        if point in results_by_point:
            return results_by_point[point]
        hparams = point_to_hparams(point, search_names, fixed_hparams)
        load_training_state(ctx.model, ctx.optimizers, ctx.batch_stream, start_state)
        needs_cooldown_state = (
            interval_info is not None and interval_info["use_cooldown"]
        )
        result = train_interval(
            ctx,
            hparams,
            steps,
            start_step,
            capture_end_state=needs_cooldown_state,
        )
        result_loss = interval_result_loss(result)
        should_evaluate_tta = result["completed_steps"] == steps and finite(result_loss)
        if interval_info is not None:
            result["cooldown_result"] = None
            result["main_tta_val_acc"] = float("-inf")
            if should_evaluate_tta:
                result["main_tta_val_acc"] = evaluate_tta_val_acc(
                    ctx.model, ctx.test_loader
                )
            if interval_info["use_cooldown"] and should_evaluate_tta:
                cooldown_search_names = cooldown_search_hparam_names()
                cooldown_start_state = result["end_state"]
                cooldown_initial_states = dict(
                    interval_info["cooldown_initial_states"] or {}
                )
                if cooldown_seed_point is not None:
                    seed_cooldown_states = results_by_point[cooldown_seed_point].get(
                        "cooldown_best_states"
                    )
                    if seed_cooldown_states is not None:
                        cooldown_initial_states.update(seed_cooldown_states)
                cooldown_initial_hparams = dict(hparams)
                for name in cooldown_search_names:
                    initial_state = cooldown_initial_states.get(name)
                    if initial_state is None:
                        initial_state = nearest_hparam_state(name, hparams[name])
                    cooldown_initial_hparams[name] = hparam_from_state(
                        name, initial_state
                    )
                cooldown_result = search_hparam_segment(
                    ctx,
                    point_from_hparams(cooldown_initial_hparams, cooldown_search_names),
                    cooldown_search_names,
                    hparams,
                    interval_info["cooldown_steps"],
                    start_step + steps,
                    cooldown_start_state,
                )
                result["cooldown_result"] = cooldown_result
                result["cooldown_best_states"] = cooldown_result["best_states"]
                result["tta_val_acc"] = cooldown_result["tta_val_acc"]
                should_evaluate_tta = False
            result.pop("end_state", None)
        if should_evaluate_tta:
            if interval_info is not None:
                result["tta_val_acc"] = result["main_tta_val_acc"]
            else:
                result["tta_val_acc"] = evaluate_tta_val_acc(ctx.model, ctx.test_loader)
        elif "tta_val_acc" not in result:
            result["tta_val_acc"] = float("-inf")
        results_by_point[point] = result
        candidate_evaluations.append(result)
        if is_main_interval:
            log_main_hparams(result)
            cooldown_result = result.get("cooldown_result")
            if cooldown_result is not None:
                log_candidate_results(cooldown_result["candidate_evaluations"])
                log_search_path(cooldown_result["search_path"])
                cooldown_result.pop("candidate_evaluations", None)
                cooldown_result.pop("search_path", None)
        return result

    def block(point):
        if point in results_by_point:
            return results_by_point[point]
        hparams = point_to_hparams(point, search_names, fixed_hparams)
        result = dict(
            blocked=True,
            last_train_loss=float("inf"),
            completed_steps=0,
            tta_val_acc=float("-inf"),
        )
        result.update(hparams)
        if interval_info is not None:
            result.update(
                main_tta_val_acc=float("-inf"),
                cooldown_result=None,
            )
        results_by_point[point] = result
        candidate_evaluations.append(result)
        if is_main_interval:
            log_main_hparams(result)
        return result

    best_point, center_path_points = find_best_hparam_point(
        initial_point=initial_point,
        search_names=search_names,
        evaluate=evaluate,
        results_by_point=results_by_point,
        block=block,
    )
    search_path = [results_by_point[point] for point in center_path_points]
    best_result = results_by_point[best_point]
    if interval_info is None:
        best_states = point_states(best_point, search_names)
        load_training_state(ctx.model, ctx.optimizers, ctx.batch_stream, start_state)
        result = dict(
            best_states=best_states,
            tta_val_acc=best_result["tta_val_acc"],
            candidate_evaluations=list(candidate_evaluations),
            search_path=list(search_path),
        )
        add_best_result_fields(result, best_result)
        return result
    load_training_state(ctx.model, ctx.optimizers, ctx.batch_stream, start_state)
    selected_hparams = copy_fields(best_result, SEARCH_HPARAMS)
    actual_result = train_interval(
        ctx,
        selected_hparams,
        steps,
        start_step,
        capture_end_state=False,
        capture_step_metrics=True,
    )
    best_cooldown_result = best_result["cooldown_result"]
    log_search_path(search_path)
    result = dict(
        interval_index=interval_info["interval_index"],
        best_point=best_point,
        cooldown_best_states=best_cooldown_result["best_states"]
        if best_cooldown_result is not None
        else None,
    )
    add_best_result_fields(result, best_result)
    result.update(
        completed_steps=actual_result["completed_steps"],
        losses=list(actual_result["losses"]),
    )
    return result


def search_interval_hparams(
    ctx,
    initial_point,
    interval_steps,
    interval_index,
    interval_start_step,
    cooldown_initial_states=None,
):
    remaining_steps_after_interval = (
        ctx.total_steps - interval_start_step - interval_steps
    )
    cooldown_steps = min(ctx.cooldown_steps, remaining_steps_after_interval)
    use_cooldown = cooldown_steps > 0
    interval_info = dict(
        interval_index=interval_index,
        cooldown_steps=cooldown_steps,
        cooldown_initial_states=cooldown_initial_states,
        use_cooldown=use_cooldown,
    )
    return search_hparam_segment(
        ctx,
        initial_point,
        ctx.search_names,
        ctx.fixed_hparams,
        interval_steps,
        interval_start_step,
        snapshot_training_state(ctx.model, ctx.optimizers, ctx.batch_stream),
        interval_info,
    )


def run_full_dataset_search(cfg):
    set_training_seed()
    search_names = active_search_hparam_names()
    initial_hparams = initial_hparams_from_cfg(cfg)
    initial_point = point_from_hparams(initial_hparams, search_names)

    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=cfg.batch_size,
        aug=dict(flip=True, translate=2),
    )
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    batch_stream = FullDatasetBatchStream(train_loader)
    cfg.train_steps = ceil(cfg.train_epochs * len(train_loader))
    cfg.whiten_bias_train_steps = ceil(3 * len(train_loader))

    cfg.model.reset()
    cfg.model.init_whiten(train_loader.normalized_images()[:5000])
    sgd_optimizer, muon_optimizer = make_optimizers(
        cfg.model,
        namespace(
            vars(cfg),
            "muon_nesterov",
            **initial_hparams,
        ),
    )
    optimizers = [sgd_optimizer, muon_optimizer]
    search_ctx = namespace(
        vars(cfg),
        "run model batch_size n_steps muon_nesterov",
        optimizers=optimizers,
        sgd_optimizer=sgd_optimizer,
        muon_optimizer=muon_optimizer,
        batch_stream=batch_stream,
        test_loader=test_loader,
        cooldown_steps=cfg.m_steps,
        total_steps=cfg.train_steps,
        whiten_bias_train_steps=cfg.whiten_bias_train_steps,
        search_names=search_names,
        fixed_hparams=initial_hparams,
    )

    last_loss = None
    interval_initial_point = initial_point
    interval_cooldown_initial_states = None
    completed_steps = 0
    interval_index = 0
    while completed_steps < cfg.train_steps:
        interval_steps = min(cfg.n_steps, cfg.train_steps - completed_steps)
        interval_result = search_interval_hparams(
            search_ctx,
            interval_initial_point,
            interval_steps,
            interval_index,
            completed_steps,
            interval_cooldown_initial_states,
        )
        interval_initial_point = interval_result["best_point"]
        if interval_result["cooldown_best_states"] is not None:
            interval_cooldown_initial_states = interval_result["cooldown_best_states"]
        actual_losses = interval_result["losses"][: interval_result["completed_steps"]]
        log_train_hparams(
            cfg.run,
            interval_result["interval_index"],
            completed_steps,
            interval_result["completed_steps"],
            cfg.train_steps,
            interval_result,
            cfg.muon_nesterov,
        )
        for local_offset, loss in enumerate(actual_losses, start=1):
            global_step = completed_steps + local_offset
            last_loss = loss
            log_train_loss(
                run=cfg.run,
                interval_index=interval_result["interval_index"],
                step=global_step,
                loss=loss,
            )
        completed_steps += interval_result["completed_steps"]
        if (
            completed_steps < cfg.train_steps
            and last_loss is not None
            and finite(last_loss)
        ):
            boundary_tta_val_acc = evaluate_tta_val_acc(cfg.model, test_loader)
            log_interval_boundary_eval(
                cfg.run,
                interval_result["interval_index"],
                completed_steps,
                cfg.train_steps,
                boundary_tta_val_acc,
            )
        interval_index += 1
        if last_loss is not None and not finite(last_loss):
            break

    val_acc = evaluate(cfg.model, test_loader, tta_level=0)
    tta_val_acc = evaluate(cfg.model, test_loader, tta_level=2)
    result = pack(
        vars(cfg),
        "run batch_size train_epochs train_steps n_steps m_steps search_names_text cooldown_search_names_text muon_nesterov",
    )
    result.update(
        val_acc=val_acc,
        tta_val_acc=tta_val_acc,
    )
    return result


def iter_run_settings():
    for config, steps in product(RUN_CONFIGS, SEARCH_STEP_CONFIGS):
        n_steps, m_steps = steps
        batch_size = config["batch_size"]
        bias_lr_mult = batch_size / 2000
        momentum_spec = SEARCH_HPARAMS["muon_momentum"]
        bias_lr_spec = SEARCH_HPARAMS["bias_lr"]
        head_lr_spec = SEARCH_HPARAMS["head_lr"]
        yield namespace(
            locals(),
            "n_steps m_steps",
            batch_size=batch_size,
            train_epochs=TRAIN_EPOCHS,
            initial_muon_momentum=momentum_spec.initial_value,
            muon_nesterov=MUON_NESTEROV,
            initial_bias_lr=bias_lr_spec.initial_value * bias_lr_mult,
            initial_head_lr=head_lr_spec.initial_value * bias_lr_mult,
        )


def print_run_banner(cfg):
    initial_hparams = initial_hparams_from_cfg(cfg)
    initial_muon_momentum = initial_hparams["muon_momentum"]
    log_event(
        "cifar_search_simple",
        run=cfg.run,
        batch_size=cfg.batch_size,
        train_epochs=cfg.train_epochs,
        n_steps=cfg.n_steps,
        m_steps=cfg.m_steps,
        search_hparams=cfg.search_names_text,
        cooldown_search_hparams=cfg.cooldown_search_names_text,
        muon_nesterov=cfg.muon_nesterov,
        initial_muon_lr=initial_hparams["muon_lr"],
        initial_muon_momentum=initial_muon_momentum,
        initial_muon_momentum_index=nearest_hparam_state(
            "muon_momentum", initial_muon_momentum
        ),
        initial_bias_lr=initial_hparams["bias_lr"],
        initial_head_lr=initial_hparams["head_lr"],
    )


def main():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print_output_path = os.path.join(current_time, PRINT_OUTPUT_FILENAME)
    print_output_dir = os.path.dirname(print_output_path)
    if print_output_dir:
        os.makedirs(print_output_dir, exist_ok=True)
    with (
        open(print_output_path, "w") as print_output_file,
        redirect_stdout(print_output_file),
    ):
        run_main()


def run_main():
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    for run, cfg in enumerate(iter_run_settings()):
        cfg.run = run
        cfg.model = model
        cfg.search_names_text = format_hparam_names(active_search_hparam_names())
        cfg.cooldown_search_names_text = format_hparam_names(
            cooldown_search_hparam_names()
        )
        print_run_banner(cfg)
        run_start_time = time.perf_counter()
        result = run_full_dataset_search(cfg)
        result["run_seconds"] = time.perf_counter() - run_start_time
        log_run_summary(result)


if __name__ == "__main__":
    main()
