"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#                  Setup                    #
#############################################

import copy
import os
import sys
import time
from itertools import product
from types import SimpleNamespace

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
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

    def grad_momentum_norm_ratio(self):
        grad_sq = None
        momentum_sq = None
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                g_sq = g.detach().float().square().sum()
                grad_sq = g_sq if grad_sq is None else grad_sq + g_sq
                buf = self.state[p].get("momentum_buffer")
                if buf is not None:
                    m_sq = buf.detach().float().square().sum() * momentum * momentum
                    momentum_sq = m_sq if momentum_sq is None else momentum_sq + m_sq
        if grad_sq is None:
            return None
        grad_norm = grad_sq.sqrt()
        momentum_norm = (
            torch.zeros((), device=grad_norm.device)
            if momentum_sq is None
            else momentum_sq.sqrt()
        )
        denom = grad_norm + momentum_norm
        if denom.item() == 0:
            return float("nan")
        return (grad_norm / denom).item()

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
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
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


def log_train_loss(
    run,
    step,
    total_steps,
    loss,
    head_lr,
    muon_lr,
    muon_momentum,
    muon_nesterov,
    muon_grad_momentum_norm_ratio,
):
    ratio_field = (
        "none"
        if muon_grad_momentum_norm_ratio is None
        else "%.6g" % muon_grad_momentum_norm_ratio
    )
    print(
        "train_loss run=%s step=%d/%d loss=%.4f head_lr=%.6g "
        "muon_lr=%.6g muon_momentum=%s muon_nesterov=%s "
        "muon_grad_momentum_norm_ratio=%s"
        % (
            run,
            step,
            total_steps,
            loss,
            head_lr,
            muon_lr,
            format_momentum(muon_momentum),
            muon_nesterov,
            ratio_field,
        ),
        flush=True,
    )


def log_eval(run, epoch, val_acc, time_seconds):
    run_info = f" run={run}" if run is not None else ""
    print(
        f"eval{run_info} epoch={epoch} val_acc={val_acc:.4f} "
        f"time_seconds={time_seconds:.4f}",
        flush=True,
    )


format_k = lambda k: "%g" % k
format_momentum = lambda momentum: "%g" % momentum


def copy_fields(source, fields, list_fields=()):
    list_fields = set(list_fields)
    return {
        key: list(source[key]) if key in list_fields else source[key] for key in fields
    }


def pack(values, fields):
    return {key: values[key] for key in fields.split()}


def namespace(values, fields, **extra):
    return SimpleNamespace(**pack(values, fields), **extra)


def optional_mapped(source, pairs):
    return {
        dst: source[src] if source is not None else None
        for (dst, src) in (pair.split(":") for pair in pairs.split())
    }


finite = lambda value: torch.isfinite(torch.tensor(value))
FIELD_FORMATTERS = {"k": format_k, "momentum": format_momentum}
RUN_SUMMARY_SPECS = "\nBatch size:          %d|batch_size\nTrain epochs:        %.3g|train_epochs\nTrain steps:         %d|train_steps\nN steps:             %d|n_steps\nM cooldown steps:    %d|m_steps\nInterval scheduler:  %s|interval_scheduler\nLR connectedness:    %s|lr_connectedness\nMomentum config:     %s|momentum_config_name\nSearch momentum:     %s|search_momentum\nMuon nesterov:       %s|muon_nesterov\nInitial Muon lr:     %.6g|initial_lr\nInitial Muon lr k:   %d|initial_lr_k\nInitial Muon mom:    %s|initial_momentum|momentum\nInitial Muon mom i:  %d|initial_momentum_index\nFinal start lr:      %.6g|final_start_muon_lr\nFinal start lr k:    %s|final_start_muon_lr_k|k\nFinal Muon lr:       %.6g|final_muon_lr\nFinal Muon lr k:     %s|final_muon_lr_k|k\nFinal end lr:        %.6g|final_end_muon_lr\nFinal end lr k:      %s|final_end_muon_lr_k|k\nFinal Muon momentum: %s|final_muon_momentum|momentum\nFinal Muon nesterov: %s|final_muon_nesterov\nFinal Muon mom i:    %s|final_muon_momentum_index|k\n".strip().splitlines()
COOLDOWN_SUMMARY_SPECS = "\nFinal cooldown lr:   %.6g|final_cooldown_muon_lr\nFinal cooldown lr k: %s|final_cooldown_muon_lr_k|k\nFinal cooldown mom:  %s|final_cooldown_muon_momentum|momentum\nFinal cooldown nest: %s|final_cooldown_muon_nesterov\nFinal cooldown mom i: %s|final_cooldown_muon_momentum_index|k\n".strip().splitlines()
RUN_FOOTER_SPECS = "\nSGD lr mult:         %.6g|sgd_lr_mult\nInitial train loss:  %.6f|initial_train_loss\nFinal train loss:    %.6f|final_train_loss\nVal acc:             %.4f|val_acc\nTTA val acc:         %.4f|tta_val_acc\nRun seconds:         %.3f|run_seconds\n".strip().splitlines()


def print_summary_specs(specs, result):
    for spec in specs:
        format_string, key, *formatter_key = spec.split("|")
        value = result[key]
        if formatter_key:
            value = FIELD_FORMATTERS[formatter_key[0]](value)
        print(format_string % value)


def log_run_summary(result):
    print_summary_specs(RUN_SUMMARY_SPECS, result)
    if result["final_cooldown_muon_lr"] is not None:
        print_summary_specs(COOLDOWN_SUMMARY_SPECS, result)
    print_summary_specs(RUN_FOOTER_SPECS, result)


def log_interval_lr_landscape(results_by_point):
    for interval_result in results_by_point.values():
        cooldown_result = interval_result["cooldown_result"]
        print(
            "main interval: %.6g %.6g %s loss=%.6f"
            % (
                interval_result["start_muon_lr"],
                interval_result["end_muon_lr"],
                format_momentum(interval_result["muon_momentum"]),
                interval_result["interval_final_loss"],
            ),
            flush=True,
        )
        if cooldown_result is None:
            continue
        for cooldown_eval in cooldown_result["search_evaluations"]:
            print(
                "%.6g %.6g -> %.6f"
                % (
                    cooldown_eval["start_muon_lr"],
                    cooldown_eval["end_muon_lr"],
                    cooldown_eval["final_loss"],
                ),
                flush=True,
            )


def count_evaluated_configs(results_by_point):
    count = 0
    for result in results_by_point.values():
        cooldown_result = result["cooldown_result"]
        count += (
            len(cooldown_result["search_evaluations"])
            if cooldown_result is not None
            else 1
        )
    return count


def log_interval_lr_search_complete(best_result, best_cooldown_result, results_by_k):
    best_muon_lrs = [best_result["start_muon_lr"], best_result["end_muon_lr"]]
    if best_cooldown_result is not None:
        best_muon_lrs.extend(
            [best_cooldown_result["start_muon_lr"], best_cooldown_result["end_muon_lr"]]
        )
    best_muon_lr = "[%s]" % ",".join("%.6g" % lr for lr in best_muon_lrs)
    print(
        "best_muon_lr=%s best_muon_momentum=%s interval_loss=%.6f "
        "final_loss=%.6f evaluated_interval_configs=%d evaluated_configs=%d"
        % (
            best_muon_lr,
            format_momentum(best_result["muon_momentum"]),
            best_result["interval_final_loss"],
            best_result["final_loss"],
            len(results_by_k),
            count_evaluated_configs(results_by_k),
        ),
        flush=True,
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

    model.eval()
    test_images = loader.normalized_images()
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.inference_mode():
        return torch.cat(
            [infer_fn(inputs, model) for inputs in test_images.split(2000)]
        )


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


############################################
#                Training                  #
############################################

TRAIN_EPOCHS = 8
LABEL_SMOOTHING = 0.2
SEARCH_STEP_CONFIGS = [(n_steps, 0) for n_steps in [1, 2, 3, 4, 5, 7, 9]]
INTERVAL_SCHEDULERS = ["constant", "linear"]
ENDPOINT_INTERVAL_SCHEDULERS = {"linear", "exp_linear"}
LINEAR_LR_CONNECTEDNESSES = ["jump_allowed", "continuous_double", "continuous_single"]
DEFAULT_LR_CONNECTEDNESS = "jump_allowed"
VALID_LR_CONNECTEDNESSES = ["jump_allowed", "continuous_double", "continuous_single"]
LR_SEARCH_BASE = 0.2
LR_SEARCH_FACTOR = 0.6
LR_SEARCH_SIG_FIGS = 2
LR_SEARCH_MAX_MOVES = 60
MOMENTUM_SEARCH_VALUES = [round(i / 10, 1) for i in range(10)] + [0.95, 0.99]
MUON_MOMENTUM_CONFIGS = [
    dict(
        momentum_config_name="search_momentum_fixed_nesterov_false",
        initial_momentum=0.6,
        search_momentum=True,
        muon_nesterov=False,
    )
]
RUN_CONFIGS = [
    dict(batch_size=125, initial_lr=0.04),
    dict(batch_size=500, initial_lr=0.079),
    dict(batch_size=2000, initial_lr=0.19),
]


def rounded_lr(value):
    if value == 0:
        return 0.0
    return round(value, LR_SEARCH_SIG_FIGS - 1 - floor(log10(abs(value))))


lr_from_k = lambda k: rounded_lr(LR_SEARCH_BASE * LR_SEARCH_FACTOR**k)


def nearest_lr_k(lr):
    if lr <= 0:
        raise ValueError(f"LR must be positive, got {lr}")
    return int(round(log(lr / LR_SEARCH_BASE) / log(LR_SEARCH_FACTOR)))


momentum_from_index = lambda momentum_index: MOMENTUM_SEARCH_VALUES[momentum_index]


def nearest_momentum_index(momentum):
    for index, search_momentum in enumerate(MOMENTUM_SEARCH_VALUES):
        if isclose(momentum, search_momentum, rel_tol=0.0, abs_tol=1e-12):
            return index
    raise ValueError(
        "momentum must be in %s, got %s" % (MOMENTUM_SEARCH_VALUES, momentum)
    )


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


def peek_stream_loss(model, batch_stream):
    state = batch_stream.state_dict()
    inputs, labels = batch_stream.next_batch()
    model.train()
    with torch.inference_mode():
        outputs = model(inputs)
        loss = F.cross_entropy(
            outputs.float(),
            labels,
            label_smoothing=LABEL_SMOOTHING,
            reduction="mean",
        )
    batch_stream.load_state_dict(state)
    return loss.item()


def make_optimizers(model, cfg):
    bias_lr = 104 * cfg.sgd_lr_mult
    head_lr = 1340 * cfg.sgd_lr_mult
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for (n, p) in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=0),
        dict(params=norm_biases, lr=bias_lr, weight_decay=0),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=0),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    optimizer2 = Muon(
        filter_params,
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
    )
    for optimizer in (optimizer1, optimizer2):
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizer1, optimizer2


def set_muon_hparams(muon_optimizer, muon_lr, muon_momentum, muon_nesterov):
    muon_lr = rounded_lr(muon_lr)
    for group in muon_optimizer.param_groups:
        group["lr"] = muon_lr
        group["momentum"] = muon_momentum
        group["nesterov"] = muon_nesterov
    return muon_lr


def set_sgd_lrs(ctx, global_step):
    whiten_scale = max(0.0, 1 - global_step / ctx.whiten_bias_train_steps)
    train_scale = max(0.0, 1 - global_step / ctx.total_steps)
    for group in ctx.sgd_optimizer.param_groups[:1]:
        group["lr"] = group["initial_lr"] * whiten_scale
    for group in ctx.sgd_optimizer.param_groups[1:]:
        group["lr"] = group["initial_lr"] * train_scale


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
        optimizer.load_state_dict(optimizer_state)
    batch_stream.load_state_dict(state["batch_stream"])
    model.zero_grad(set_to_none=True)


def train_one_step(ctx, muon_lr, muon_momentum, global_step):
    set_muon_hparams(ctx.muon_optimizer, muon_lr, muon_momentum, ctx.muon_nesterov)
    set_sgd_lrs(ctx, global_step)
    head_lr = ctx.sgd_optimizer.param_groups[2]["lr"]
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
    muon_grad_momentum_norm_ratio = ctx.muon_optimizer.grad_momentum_norm_ratio()
    for optimizer in ctx.optimizers:
        optimizer.step()
    ctx.model.zero_grad(set_to_none=True)
    return loss.item(), muon_grad_momentum_norm_ratio, head_lr


def train_interval(ctx, muon_lrs, muon_momentum, interval_steps, start_step):
    muon_lrs = [rounded_lr(muon_lr) for muon_lr in muon_lrs]
    assert len(muon_lrs) == interval_steps
    losses = [peek_stream_loss(ctx.model, ctx.batch_stream)]
    step_losses = []
    head_lrs = []
    muon_grad_momentum_norm_ratios = []
    completed_steps = 0
    for offset, muon_lr in enumerate(muon_lrs):
        step_loss, muon_grad_momentum_norm_ratio, head_lr = train_one_step(
            ctx, muon_lr, muon_momentum, start_step + offset
        )
        step_losses.append(step_loss)
        head_lrs.append(head_lr)
        muon_grad_momentum_norm_ratios.append(muon_grad_momentum_norm_ratio)
        eval_loss = peek_stream_loss(ctx.model, ctx.batch_stream)
        losses.append(eval_loss)
        completed_steps += 1
        if not finite(eval_loss):
            break
    while len(losses) <= interval_steps:
        losses.append(float("inf"))
    return dict(
        muon_lr=muon_lrs[-1] if muon_lrs else None,
        muon_lrs=muon_lrs,
        start_muon_lr=muon_lrs[0] if muon_lrs else None,
        end_muon_lr=muon_lrs[-1] if muon_lrs else None,
        muon_momentum=muon_momentum,
        muon_nesterov=ctx.muon_nesterov,
        losses=losses,
        step_losses=step_losses,
        head_lrs=head_lrs,
        muon_grad_momentum_norm_ratios=muon_grad_momentum_norm_ratios,
        completed_steps=completed_steps,
        final_loss=losses[interval_steps],
        end_state=snapshot_training_state(ctx.model, ctx.optimizers, ctx.batch_stream),
    )


def lr_schedule_from_endpoints(
    interval_scheduler,
    start_lr,
    end_lr,
    interval_steps,
    lr_connectedness="jump_allowed",
    fixed_start=False,
):
    if interval_scheduler == "constant" or interval_steps == 1:
        if (
            interval_scheduler in ENDPOINT_INTERVAL_SCHEDULERS
            and interval_steps == 1
            and fixed_start
            and lr_connectedness == "continuous_single"
        ):
            return [rounded_lr(end_lr)]
        return [rounded_lr(start_lr)] * interval_steps
    if interval_scheduler == "linear":
        if fixed_start and lr_connectedness == "continuous_single":
            return [
                rounded_lr(start_lr + (end_lr - start_lr) * i / interval_steps)
                for i in range(1, interval_steps + 1)
            ]
        return [
            rounded_lr(start_lr + (end_lr - start_lr) * i / (interval_steps - 1))
            for i in range(interval_steps)
        ]
    if interval_scheduler == "exp_linear":
        start_lr_k = nearest_lr_k(start_lr)
        end_lr_k = nearest_lr_k(end_lr)
        if fixed_start and lr_connectedness == "continuous_single":
            return [
                lr_from_k(start_lr_k + (end_lr_k - start_lr_k) * i / interval_steps)
                for i in range(1, interval_steps + 1)
            ]
        return [
            lr_from_k(start_lr_k + (end_lr_k - start_lr_k) * i / (interval_steps - 1))
            for i in range(interval_steps)
        ]
    raise ValueError(f"Unknown interval scheduler: {interval_scheduler}")


def lr_connectedness_uses_fixed_start(lr_connectedness, fixed_start_lr_k):
    if lr_connectedness not in VALID_LR_CONNECTEDNESSES:
        raise ValueError(f"Unknown LR connectedness: {lr_connectedness}")
    return lr_connectedness.startswith("continuous_") and fixed_start_lr_k is not None


def make_initial_search_point(
    interval_scheduler,
    interval_steps,
    initial_lr_k,
    initial_momentum_index,
    lr_connectedness="jump_allowed",
    fixed_start_lr_k=None,
):
    if lr_connectedness_uses_fixed_start(lr_connectedness, fixed_start_lr_k):
        return initial_lr_k, initial_momentum_index
    if interval_scheduler in ENDPOINT_INTERVAL_SCHEDULERS and interval_steps > 1:
        return initial_lr_k, initial_lr_k, initial_momentum_index
    return initial_lr_k, initial_momentum_index


def unpack_search_point(
    interval_scheduler,
    interval_steps,
    point,
    lr_connectedness="jump_allowed",
    fixed_start_lr_k=None,
):
    if lr_connectedness_uses_fixed_start(lr_connectedness, fixed_start_lr_k):
        end_k, momentum_index = point
        return fixed_start_lr_k, end_k, momentum_index
    if interval_scheduler in ENDPOINT_INTERVAL_SCHEDULERS and interval_steps > 1:
        start_k, end_k, momentum_index = point
    else:
        start_k, momentum_index = point
        end_k = start_k
    return start_k, end_k, momentum_index


def point_sort_key(point):
    lr_ks = point[:-1]
    momentum_index = point[-1]
    return sum(abs(k) for k in lr_ks), tuple(lr_ks), momentum_index


def better_point(point, incumbent_point, results_by_point):
    if incumbent_point is None:
        return True
    loss = results_by_point[point]["final_loss"]
    incumbent_loss = results_by_point[incumbent_point]["final_loss"]
    return (loss, point_sort_key(point)) < (
        incumbent_loss,
        point_sort_key(incumbent_point),
    )


def neighbor_points(point, lr_step, search_momentum):
    lr_ks = point[:-1]
    momentum_index = point[-1]
    if len(lr_ks) == 1:
        k = lr_ks[0]
        yield (k - lr_step, momentum_index)
        yield (k + lr_step, momentum_index)
    elif len(lr_ks) == 2:
        start_k, end_k = lr_ks
        for start_delta in (-lr_step, 0, lr_step):
            for end_delta in (-lr_step, 0, lr_step):
                if start_delta == 0 and end_delta == 0:
                    continue
                yield (start_k + start_delta, end_k + end_delta, momentum_index)
    else:
        raise ValueError(f"Unsupported LR point: {point}")
    if search_momentum:
        if momentum_index > 0:
            yield (*lr_ks, momentum_index - 1)
        if momentum_index + 1 < len(MOMENTUM_SEARCH_VALUES):
            yield (*lr_ks, momentum_index + 1)


def best_neighbor_point(middle_point, results_by_point, lr_step, search_momentum):
    best_point = middle_point
    for point in neighbor_points(middle_point, lr_step, search_momentum):
        if better_point(point, best_point, results_by_point):
            best_point = point
    return best_point


def find_best_lr_momentum_point(
    initial_point, evaluate, results_by_point, search_momentum
):
    middle_point = initial_point
    initial_points = [middle_point]
    evaluate(middle_point)
    for point in neighbor_points(middle_point, 1, search_momentum):
        evaluate(point)
        initial_points.append(point)
    for point in initial_points:
        if better_point(point, middle_point, results_by_point):
            middle_point = point
    for _ in range(LR_SEARCH_MAX_MOVES):
        for point in neighbor_points(middle_point, 1, search_momentum):
            evaluate(point)
        next_point = best_neighbor_point(
            middle_point, results_by_point, 1, search_momentum
        )
        if next_point == middle_point:
            break
        middle_point = next_point
    else:
        print(
            "lr_momentum_search_warning did_not_converge_within=%d "
            "using_point=%s final_loss=%.6f"
            % (
                LR_SEARCH_MAX_MOVES,
                middle_point,
                results_by_point[middle_point]["final_loss"],
            ),
            flush=True,
        )
    return middle_point


def add_search_point_metadata(
    result,
    start_k,
    end_k,
    momentum_index,
    initial_lr_k,
    initial_lr,
    initial_momentum_index,
    initial_momentum,
):
    result.update(
        k=end_k,
        start_lr_k=start_k,
        end_lr_k=end_k,
        start_lr=lr_from_k(start_k),
        end_lr=lr_from_k(end_k),
        momentum_index=momentum_index,
        initial_lr_k=initial_lr_k,
        initial_lr=initial_lr,
        initial_momentum_index=initial_momentum_index,
        initial_momentum=initial_momentum,
    )


def search_evaluation_summary(
    result, include_interval_final_loss=False, include_cooldown_result=False
):
    fields = "k start_lr_k end_lr_k momentum_index muon_lr muon_lrs start_muon_lr end_muon_lr muon_momentum muon_nesterov initial_lr_k initial_lr initial_momentum_index initial_momentum".split()
    summary = copy_fields(result, fields, list_fields=("muon_lrs",))
    if include_interval_final_loss:
        summary["interval_final_loss"] = result["interval_final_loss"]
    summary.update(
        copy_fields(
            result,
            "losses muon_grad_momentum_norm_ratios final_loss completed_steps".split(),
            list_fields=("losses", "muon_grad_momentum_norm_ratios"),
        )
    )
    if include_cooldown_result:
        summary["cooldown_result"] = result["cooldown_result"]
    return summary


BEST_RESULT_FIELDS = "muon_lr muon_lrs start_muon_lr end_muon_lr muon_momentum muon_nesterov completed_steps losses step_losses head_lrs muon_grad_momentum_norm_ratios".split()
BEST_RESULT_LIST_FIELDS = (
    "muon_lrs",
    "losses",
    "step_losses",
    "head_lrs",
    "muon_grad_momentum_norm_ratios",
)


def add_best_result_fields(result, best_result):
    result.update(
        copy_fields(
            best_result, BEST_RESULT_FIELDS, list_fields=BEST_RESULT_LIST_FIELDS
        )
    )


def segment_evaluations(results_by_point, include_interval=False):
    return [
        search_evaluation_summary(
            result,
            include_interval_final_loss=include_interval,
            include_cooldown_result=include_interval,
        )
        for result in results_by_point.values()
    ]


def search_lr_segment(
    ctx,
    search_name,
    ilk,
    imi,
    steps,
    start_step,
    start_state,
    search_momentum,
    fsk=None,
    ii=None,
):
    results_by_point = {}
    initial_lr = lr_from_k(ilk)
    initial_momentum = momentum_from_index(imi)
    scheduler = ctx.interval_scheduler
    connectedness = ctx.lr_connectedness
    fixed_start = lr_connectedness_uses_fixed_start(connectedness, fsk)
    initial_point = make_initial_search_point(
        scheduler, steps, ilk, imi, lr_connectedness=connectedness, fixed_start_lr_k=fsk
    )

    def evaluate(point):
        start_k, end_k, momentum_index = unpack_search_point(
            scheduler,
            steps,
            point,
            lr_connectedness=connectedness,
            fixed_start_lr_k=fsk,
        )
        if point in results_by_point:
            return results_by_point[point]
        start_lr = lr_from_k(start_k)
        end_lr = lr_from_k(end_k)
        momentum = momentum_from_index(momentum_index)
        load_training_state(ctx.model, ctx.optimizers, ctx.batch_stream, start_state)
        muon_lrs = lr_schedule_from_endpoints(
            scheduler,
            start_lr,
            end_lr,
            steps,
            lr_connectedness=connectedness,
            fixed_start=fixed_start,
        )
        result = train_interval(ctx, muon_lrs, momentum, steps, start_step)
        result["start_muon_lr"] = start_lr
        result["end_muon_lr"] = end_lr
        if ii is not None:
            result["interval_final_loss"] = result["final_loss"]
            result["cooldown_result"] = None
            if (
                ii["use_cooldown"]
                and result["completed_steps"] == steps
                and finite(result["final_loss"])
            ):
                cooldown_search_name = "%s_cooldown_for_start%s_end%s_m%s_n%s" % (
                    search_name,
                    format_k(start_k),
                    format_k(end_k),
                    format_momentum(momentum),
                    ctx.muon_nesterov,
                )
                fixed_cooldown_start = (
                    end_k if connectedness.startswith("continuous_") else None
                )
                cooldown_result = search_lr_segment(
                    ctx,
                    cooldown_search_name,
                    end_k,
                    momentum_index,
                    ctx.cooldown_steps,
                    start_step + steps,
                    result["end_state"],
                    False,
                    fixed_cooldown_start,
                )
                result["cooldown_result"] = cooldown_result
                result["final_loss"] = cooldown_result["final_train_loss"]
        add_search_point_metadata(
            result,
            start_k,
            end_k,
            momentum_index,
            ilk,
            initial_lr,
            imi,
            initial_momentum,
        )
        results_by_point[point] = result
        return result

    best_point = find_best_lr_momentum_point(
        initial_point=initial_point,
        evaluate=evaluate,
        results_by_point=results_by_point,
        search_momentum=search_momentum,
    )
    best_start_k, best_end_k, best_momentum_index = unpack_search_point(
        scheduler,
        steps,
        best_point,
        lr_connectedness=connectedness,
        fixed_start_lr_k=fsk,
    )
    best_result = results_by_point[best_point]
    if ii is None:
        load_training_state(ctx.model, ctx.optimizers, ctx.batch_stream, start_state)
        result = pack(
            locals(), "search_name search_momentum initial_lr initial_momentum"
        )
        result.update(initial_lr_k=ilk, initial_momentum_index=imi)
        result.update(
            cooldown_steps=steps,
            interval_scheduler=scheduler,
            lr_connectedness=connectedness,
            best_k=best_end_k,
            best_start_lr_k=best_start_k,
            best_end_lr_k=best_end_k,
            best_momentum_index=best_momentum_index,
            initial_train_loss=best_result["losses"][0],
            final_train_loss=best_result["final_loss"],
        )
        result.update(search_evaluations=segment_evaluations(results_by_point))
        add_best_result_fields(result, best_result)
        return result
    load_training_state(
        ctx.model, ctx.optimizers, ctx.batch_stream, best_result["end_state"]
    )
    best_cooldown_result = best_result["cooldown_result"]
    log_interval_lr_landscape(results_by_point)
    log_interval_lr_search_complete(best_result, best_cooldown_result, results_by_point)
    result = pack(locals(), "search_name search_momentum initial_lr initial_momentum")
    result.update(initial_lr_k=ilk, initial_momentum_index=imi)
    result.update(
        pack(ii, "interval_index interval_start_step cooldown_steps"),
        interval_steps=steps,
        interval_scheduler=scheduler,
        lr_connectedness=connectedness,
        best_k=best_end_k,
        best_start_lr_k=best_start_k,
        best_end_lr_k=best_end_k,
        best_momentum_index=best_momentum_index,
        cooldown_initial_lr_k=best_end_k,
        cooldown_initial_lr=lr_from_k(best_end_k),
        cooldown_initial_momentum_index=best_momentum_index,
        cooldown_initial_momentum=momentum_from_index(best_momentum_index),
        initial_train_loss=best_result["losses"][0],
        interval_final_train_loss=best_result["interval_final_loss"],
        final_train_loss=best_result["final_loss"],
        cooldown_result=best_cooldown_result,
    )
    add_best_result_fields(result, best_result)
    result.update(
        optional_mapped(
            best_cooldown_result,
            "cooldown_best_k:best_k cooldown_best_momentum_index:best_momentum_index cooldown_muon_lr:muon_lr cooldown_muon_momentum:muon_momentum cooldown_muon_nesterov:muon_nesterov",
        )
    )
    result.update(search_evaluations=segment_evaluations(results_by_point, True))
    return result


def search_interval_lr(
    ctx,
    initial_lr_k,
    initial_momentum_index,
    interval_steps,
    interval_index,
    interval_start_step,
    fixed_start_lr_k=None,
):
    search_name = "run%d_bs%d_N%d_M%d_%s_%s_interval%d_step%d" % (
        ctx.run,
        ctx.batch_size,
        ctx.n_steps,
        ctx.cooldown_steps,
        ctx.interval_scheduler,
        ctx.lr_connectedness,
        interval_index,
        interval_start_step,
    )
    use_cooldown = (
        interval_start_step + interval_steps + ctx.cooldown_steps <= ctx.total_steps
    )
    use_cooldown = use_cooldown and ctx.cooldown_steps > 0
    interval_info = dict(
        interval_index=interval_index,
        interval_start_step=interval_start_step,
        cooldown_steps=ctx.cooldown_steps if use_cooldown else 0,
        use_cooldown=use_cooldown,
    )
    return search_lr_segment(
        ctx,
        search_name,
        initial_lr_k,
        initial_momentum_index,
        interval_steps,
        interval_start_step,
        snapshot_training_state(ctx.model, ctx.optimizers, ctx.batch_stream),
        ctx.search_momentum,
        fixed_start_lr_k,
        interval_info,
    )


def run_full_dataset_search(cfg):
    set_training_seed()
    initial_lr_k = nearest_lr_k(cfg.initial_lr)
    initial_lr = lr_from_k(initial_lr_k)
    initial_momentum_index = nearest_momentum_index(cfg.initial_momentum)
    initial_momentum = momentum_from_index(initial_momentum_index)

    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=cfg.batch_size, aug=dict(flip=True, translate=2)
    )
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    batch_stream = FullDatasetBatchStream(train_loader)
    cfg.train_steps = ceil(cfg.train_epochs * len(train_loader))
    cfg.whiten_bias_train_steps = ceil(3 * len(train_loader))

    cfg.model.reset()
    cfg.model.init_whiten(train_loader.normalized_images()[:5000])
    optimizer1, optimizer2 = make_optimizers(
        cfg.model,
        namespace(
            vars(cfg),
            "muon_nesterov sgd_lr_mult",
            muon_lr=rounded_lr(initial_lr),
            muon_momentum=initial_momentum,
        ),
    )
    optimizers = [optimizer1, optimizer2]
    search_ctx = namespace(
        vars(cfg),
        "run model batch_size n_steps search_momentum muon_nesterov interval_scheduler lr_connectedness",
        optimizers=optimizers,
        sgd_optimizer=optimizer1,
        muon_optimizer=optimizer2,
        batch_stream=batch_stream,
        cooldown_steps=cfg.m_steps,
        total_steps=cfg.train_steps,
        whiten_bias_train_steps=cfg.whiten_bias_train_steps,
    )

    losses = [peek_stream_loss(cfg.model, batch_stream)]
    log_train_loss(
        run=cfg.run,
        step=0,
        total_steps=cfg.train_steps,
        loss=losses[-1],
        head_lr=optimizer1.param_groups[2]["lr"],
        muon_lr=optimizer2.param_groups[0]["lr"],
        muon_momentum=optimizer2.param_groups[0]["momentum"],
        muon_nesterov=optimizer2.param_groups[0]["nesterov"],
        muon_grad_momentum_norm_ratio=None,
    )

    interval_results = []
    interval_initial_lr_k = initial_lr_k
    interval_initial_momentum_index = initial_momentum_index
    completed_steps = 0
    interval_index = 0
    while completed_steps < cfg.train_steps:
        interval_steps = min(cfg.n_steps, cfg.train_steps - completed_steps)
        previous_interval_end_lr_k = None
        if interval_index > 0 and cfg.lr_connectedness.startswith("continuous_"):
            previous_interval_end_lr_k = interval_initial_lr_k
        interval_result = search_interval_lr(
            search_ctx,
            interval_initial_lr_k,
            interval_initial_momentum_index,
            interval_steps,
            interval_index,
            completed_steps,
            previous_interval_end_lr_k,
        )
        interval_results.append(interval_result)
        interval_initial_lr_k = interval_result["best_k"]
        interval_initial_momentum_index = interval_result["best_momentum_index"]
        actual_losses = interval_result["losses"][
            1 : 1 + interval_result["completed_steps"]
        ]
        actual_muon_lrs = interval_result["muon_lrs"][
            : interval_result["completed_steps"]
        ]
        actual_muon_grad_momentum_norm_ratios = interval_result[
            "muon_grad_momentum_norm_ratios"
        ][: interval_result["completed_steps"]]
        actual_head_lrs = interval_result["head_lrs"][
            : interval_result["completed_steps"]
        ]
        for local_offset, (
            loss,
            muon_lr,
            head_lr,
            muon_grad_momentum_norm_ratio,
        ) in enumerate(
            zip(
                actual_losses,
                actual_muon_lrs,
                actual_head_lrs,
                actual_muon_grad_momentum_norm_ratios,
            ),
            start=1,
        ):
            global_step = completed_steps + local_offset
            losses.append(loss)
            log_train_loss(
                run=cfg.run,
                step=global_step,
                total_steps=cfg.train_steps,
                loss=loss,
                head_lr=head_lr,
                muon_lr=muon_lr,
                muon_momentum=interval_result["muon_momentum"],
                muon_nesterov=interval_result["muon_nesterov"],
                muon_grad_momentum_norm_ratio=muon_grad_momentum_norm_ratio,
            )
        completed_steps += interval_result["completed_steps"]
        interval_index += 1
        if not torch.isfinite(torch.tensor(losses[-1])):
            break

    while len(losses) <= cfg.train_steps:
        losses.append(float("inf"))
    last_cooldown_result = next(
        (
            interval_result
            for interval_result in reversed(interval_results)
            if interval_result["cooldown_best_k"] is not None
        ),
        None,
    )
    last_interval = interval_results[-1] if interval_results else None
    val_acc = evaluate(cfg.model, test_loader, tta_level=0)
    tta_val_acc = evaluate(cfg.model, test_loader, tta_level=2)
    result = pack(
        vars(cfg),
        "run batch_size train_epochs train_steps n_steps m_steps momentum_config_name search_momentum muon_nesterov interval_scheduler lr_connectedness sgd_lr_mult",
    )
    result.update(
        initial_lr_k=initial_lr_k,
        initial_lr=initial_lr,
        initial_momentum_index=initial_momentum_index,
        initial_momentum=initial_momentum,
        losses=losses,
        interval_results=interval_results,
    )
    result.update(
        optional_mapped(
            last_interval,
            "final_muon_lr:muon_lr final_muon_lr_k:best_k final_start_muon_lr:start_muon_lr final_start_muon_lr_k:best_start_lr_k final_end_muon_lr:end_muon_lr final_end_muon_lr_k:best_end_lr_k final_muon_momentum:muon_momentum final_muon_nesterov:muon_nesterov final_muon_momentum_index:best_momentum_index",
        )
    )
    result.update(
        optional_mapped(
            last_cooldown_result,
            "final_cooldown_muon_lr:cooldown_muon_lr final_cooldown_muon_momentum:cooldown_muon_momentum final_cooldown_muon_nesterov:cooldown_muon_nesterov final_cooldown_muon_lr_k:cooldown_best_k final_cooldown_muon_momentum_index:cooldown_best_momentum_index",
        )
    )
    result.update(
        initial_train_loss=losses[0],
        final_train_loss=losses[cfg.train_steps],
        val_acc=val_acc,
        tta_val_acc=tta_val_acc,
        steps=completed_steps,
        target_steps=cfg.train_steps,
        muon_grad_momentum_norm_ratios=[
            ratio
            for interval_result in interval_results
            for ratio in interval_result["muon_grad_momentum_norm_ratios"][
                : interval_result["completed_steps"]
            ]
        ],
        head_lrs=[
            head_lr
            for interval_result in interval_results
            for head_lr in interval_result["head_lrs"][
                : interval_result["completed_steps"]
            ]
        ],
    )
    return result


def iter_run_settings():
    for config, steps, interval_scheduler in product(
        RUN_CONFIGS, SEARCH_STEP_CONFIGS, INTERVAL_SCHEDULERS
    ):
        lr_connectednesses = (
            LINEAR_LR_CONNECTEDNESSES
            if interval_scheduler == "linear"
            else [DEFAULT_LR_CONNECTEDNESS]
        )
        for lr_connectedness, momentum_config in product(
            lr_connectednesses, MUON_MOMENTUM_CONFIGS
        ):
            n_steps, m_steps = steps
            batch_size = config["batch_size"]
            yield namespace(
                locals(),
                "n_steps m_steps interval_scheduler lr_connectedness",
                batch_size=batch_size,
                train_epochs=TRAIN_EPOCHS,
                initial_lr=config["initial_lr"],
                sgd_lr_mult=batch_size / 2000,
                **momentum_config,
            )


RUN_BANNER_FIELDS = "run batch_size train_epochs n_steps m_steps interval_scheduler lr_connectedness momentum_config_name search_momentum muon_nesterov initial_lr initial_lr_k initial_momentum_text initial_momentum_index".split()


def print_run_banner(cfg):
    print(
        "cifar_baseline2_full_search run=%d batch_size=%d train_epochs=%.3g "
        "N=%d M=%d interval_scheduler=%s lr_connectedness=%s "
        "momentum_config=%s search_momentum=%s muon_nesterov=%s "
        "initial_muon_lr=%.6g initial_muon_lr_k=%d "
        "initial_muon_momentum=%s initial_muon_momentum_index=%d"
        % tuple(getattr(cfg, field) for field in RUN_BANNER_FIELDS),
        flush=True,
    )


def main():
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    results = []
    for cfg in iter_run_settings():
        cfg.run = len(results)
        cfg.model = model
        cfg.initial_lr_k = nearest_lr_k(cfg.initial_lr)
        cfg.initial_momentum_text = format_momentum(cfg.initial_momentum)
        cfg.initial_momentum_index = nearest_momentum_index(cfg.initial_momentum)
        print_run_banner(cfg)
        run_start_time = time.perf_counter()
        result = run_full_dataset_search(cfg)
        result["run_seconds"] = time.perf_counter() - run_start_time
        results.append(result)
        log_run_summary(result)
    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results), log_path)
    print(os.path.abspath(log_path), flush=True)


if __name__ == "__main__":
    main()
