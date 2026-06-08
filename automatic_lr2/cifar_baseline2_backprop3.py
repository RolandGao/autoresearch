"""
CIFAR-10 target-backprop experiment.
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#                  Setup                    #
#############################################

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil, isfinite

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

TRAINING_SEED = 0
TARGET_HEAD_LAMBDA = float(os.environ.get("TARGET_HEAD_LAMBDA", "22000"))
TARGET_X_LAMBDA = float(os.environ.get("TARGET_X_LAMBDA", "0.0001"))
PINV_RTOL = 1e-5
CONV_BATCH_CHUNK = int(os.environ.get("CONV_BATCH_CHUNK", "128"))
INNER_TARGET_SWEEPS = int(os.environ.get("INNER_TARGET_SWEEPS", "1"))


DEBUG_TARGET_BACKPROP = False
DEBUG_FORWARD_LOG = False
DEBUG_LAYER_LOG = False
DEBUG_MATRIX_LOG = False
DEBUG_CHUNK_LOG = False
DEBUG_PINV_LOG = False
DEBUG_PARAM_LOG = False

_DEBUG_CONTEXT = dict(step=None, sweep=None, phase=None)


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


def _debug_context_text():
    parts = []
    for key in ("step", "sweep", "phase"):
        value = _DEBUG_CONTEXT.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _debug_log(message):
    return


def _debug_tensor(name, tensor, include_sample_rms=True):
    if not DEBUG_TARGET_BACKPROP:
        return
    if tensor is None:
        _debug_log(f"{name}: none")
        return
    with torch.no_grad():
        t = tensor.detach()
        shape = tuple(t.shape)
        if t.numel() == 0:
            _debug_log(f"{name} | shape={shape} | dtype={t.dtype} | device={t.device} | empty")
            return
        tf = t.float()
        finite = torch.isfinite(tf)
        finite_count = int(finite.sum().item())
        total = tf.numel()
        nan_count = int(torch.isnan(tf).sum().item())
        posinf_count = int(torch.isposinf(tf).sum().item())
        neginf_count = int(torch.isneginf(tf).sum().item())
        if finite_count > 0:
            values = tf[finite]
            mean = values.mean().item()
            std = values.std(unbiased=False).item()
            rms = values.square().mean().sqrt().item()
            min_value = values.min().item()
            max_value = values.max().item()
            absmax = values.abs().max().item()
        else:
            mean = std = rms = min_value = max_value = absmax = float("nan")
        sample_text = ""
        if include_sample_rms and t.ndim >= 2:
            sample_values = torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0)
            sample_rms = sample_values.reshape(len(t), -1).square().mean(dim=1).sqrt()
            sample_text = (
                " | sample_rms mean=%.6g min=%.6g max=%.6g"
                % (
                    sample_rms.mean().item(),
                    sample_rms.min().item(),
                    sample_rms.max().item(),
                )
            )
        _debug_log(
            "%s | shape=%s | dtype=%s | device=%s | finite=%d/%d "
            "(nan=%d, +inf=%d, -inf=%d) | mean=%.6g std=%.6g rms=%.6g "
            "min=%.6g max=%.6g absmax=%.6g%s"
            % (
                name,
                shape,
                t.dtype,
                t.device,
                finite_count,
                total,
                nan_count,
                posinf_count,
                neginf_count,
                mean,
                std,
                rms,
                min_value,
                max_value,
                absmax,
                sample_text,
            )
        )


def _debug_scalar(name, value):
    if not DEBUG_TARGET_BACKPROP:
        return
    if torch.is_tensor(value):
        value = value.detach().float().item()
    _debug_log(f"{name}=%.9g" % float(value))


def _debug_fraction(name, mask):
    if not DEBUG_TARGET_BACKPROP:
        return
    value = mask.float().mean().item() if mask.numel() else float("nan")
    _debug_log(f"{name}=%.9g" % value)


def _debug_module_name(module):
    return getattr(module, "_debug_name", module.__class__.__name__)


def register_debug_names(model):
    for name, module in model.named_modules():
        module._debug_name = name or "model"


def _debug_model_parameters(model, label):
    if not (DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG):
        return
    _debug_log(f"model_parameters label={label}")
    for name, param in model.named_parameters():
        _debug_log(f"param name={name} requires_grad={param.requires_grad}")
        _debug_tensor(f"{label}.{name}", param.data, include_sample_rms=False)


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


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _conv_padding_tuple(padding, kernel_size, dilation):
    if padding == "same":
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)
        effective_kernel = tuple(
            d * (k - 1) + 1 for k, d in zip(kernel_size, dilation)
        )
        if any(k % 2 == 0 for k in effective_kernel):
            raise ValueError(
                '"same" padding helper only supports odd effective kernels'
            )
        return tuple(k // 2 for k in effective_kernel)
    if padding == "valid":
        return (0, 0)
    return _pair(padding)


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])

class Linear(nn.Linear):
    pass


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.ReLU()

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
        self.whiten.bias.requires_grad = False
        self.layers = nn.Sequential(
            nn.ReLU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear, Linear):
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

    def forward(self, x):
        return forward_no_cache(self, x)


############################################
#       Pseudoinverse target backprop       #
############################################


def _safe_pinv(x, name="pinv"):
    x = torch.nan_to_num(x.float())
    if DEBUG_PINV_LOG:
        _debug_tensor(f"{name}.pinv_input", x, include_sample_rms=False)
    last_error = None
    for rtol in (PINV_RTOL, 1e-4, 1e-3):
        try:
            result = torch.linalg.pinv(x, rtol=rtol)
            if DEBUG_PINV_LOG:
                _debug_log(f"{name}.pinv_success rtol={rtol:.6g}")
                _debug_tensor(f"{name}.pinv_output", result, include_sample_rms=False)
            return result
        except RuntimeError as err:
            last_error = err
            if DEBUG_PINV_LOG:
                _debug_log(f"{name}.pinv_failed rtol={rtol:.6g} error={err}")
    raise last_error


def _ridge_delta(xtx, xtdy, lambda_value=0.0, name="weight_solve"):
    if DEBUG_MATRIX_LOG:
        _debug_log(f"{name}.ridge_delta lambda={lambda_value:.6g}")
        _debug_tensor(f"{name}.xtx_before_ridge", xtx, include_sample_rms=False)
        _debug_tensor(f"{name}.xtdy", xtdy, include_sample_rms=False)
    if lambda_value != 0:
        eye = torch.eye(xtx.size(0), device=xtx.device, dtype=xtx.dtype)
        xtx = xtx + lambda_value * eye
        if DEBUG_MATRIX_LOG:
            _debug_tensor(f"{name}.xtx_after_ridge", xtx, include_sample_rms=False)
    delta = _safe_pinv(xtx, name=f"{name}.xtx") @ xtdy.float()
    if DEBUG_MATRIX_LOG:
        _debug_tensor(f"{name}.delta", delta, include_sample_rms=False)
        residual = xtx @ delta - xtdy.float()
        _debug_tensor(f"{name}.normal_equation_residual", residual, include_sample_rms=False)
    return delta


def _ridge_input_delta(dy, weight_eff, name="input_delta"):
    weight_eff = weight_eff.float()
    dy = dy.float()
    if DEBUG_MATRIX_LOG:
        _debug_log(f"{name}.ridge_input_delta lambda_x={TARGET_X_LAMBDA:.6g}")
        _debug_tensor(f"{name}.dy", dy, include_sample_rms=False)
        _debug_tensor(f"{name}.weight_eff", weight_eff, include_sample_rms=False)
    if TARGET_X_LAMBDA == 0:
        dx = dy @ _safe_pinv(weight_eff, name=f"{name}.weight_eff")
        if DEBUG_MATRIX_LOG:
            _debug_tensor(f"{name}.dx", dx, include_sample_rms=False)
        return dx
    gram = weight_eff.T @ weight_eff
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    solve_matrix = gram + TARGET_X_LAMBDA * eye
    if DEBUG_MATRIX_LOG:
        _debug_tensor(f"{name}.gram", gram, include_sample_rms=False)
        _debug_tensor(f"{name}.solve_matrix", solve_matrix, include_sample_rms=False)
    dx = dy @ _safe_pinv(solve_matrix, name=f"{name}.solve_matrix") @ weight_eff.T
    if DEBUG_MATRIX_LOG:
        _debug_tensor(f"{name}.dx", dx, include_sample_rms=False)
    return dx


def _project_weight_(p, max_rms=1.0, name="weight"):
    if p is None or not p.requires_grad:
        if DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
            _debug_log(f"{name}.projection skipped=1")
        return
    rms = p.data.float().square().mean().sqrt()
    if torch.isfinite(rms) and rms > 0:
        scale = max_rms / rms
        p.data.mul_(scale.to(dtype=p.data.dtype))
        if DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
            new_rms = p.data.float().square().mean().sqrt()
            _debug_log(
                "%s.projection target_rms=%.6g old_rms=%.6g "
                "scale=%.6g new_rms=%.6g"
                % (name, max_rms, rms.item(), scale.item(), new_rms.item())
            )
            _debug_tensor(f"{name}.after_projection", p.data, include_sample_rms=False)
    elif DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
        _debug_log(f"{name}.projection skipped_nonfinite_or_zero_rms={rms.item():.6g}")


def _project_bias_(p, name="bias"):
    if p is not None and p.requires_grad:
        if DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
            old_min = p.data.float().min().item()
            old_max = p.data.float().max().item()
        p.data.clamp_(-1, 1)
        if DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
            _debug_log(
                "%s.projection old_min=%.6g old_max=%.6g new_min=%.6g new_max=%.6g"
                % (
                    name,
                    old_min,
                    old_max,
                    p.data.float().min().item(),
                    p.data.float().max().item(),
                )
            )
            _debug_tensor(f"{name}.after_projection", p.data, include_sample_rms=False)
    elif DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
        _debug_log(f"{name}.projection skipped=1")


def project_trainable_parameters(model):
    for name, module in model.named_modules():
        if module is getattr(model, "head", None):
            if DEBUG_TARGET_BACKPROP and DEBUG_PARAM_LOG:
                _debug_log(f"{name}.head_projection skipped_weight_projection=1")
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        _project_weight_(weight, name=f"{name}.weight")
        _project_bias_(bias, name=f"{name}.bias")


def _bound_sample_rms(x):
    return x


def _bound_delta_to_sample_rms(x, dx):
    return dx


def _smooth_targets(labels, num_classes, smoothing, dtype):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    targets = torch.full(
        (len(labels), num_classes),
        off_value,
        device=labels.device,
        dtype=dtype,
    )
    targets.scatter_(1, labels.view(-1, 1), on_value)
    return targets


def _low_loss_logit_targets(labels, num_classes, smoothing, dtype):
    probs = _smooth_targets(labels, num_classes, smoothing, torch.float32)
    logits = probs.log()
    logits = logits - logits.mean(dim=1, keepdim=True)
    return logits.to(dtype=dtype)


def cross_entropy_loss_and_delta(logits, labels):
    logits_f = logits.float()
    loss = F.cross_entropy(
        logits_f,
        labels,
        label_smoothing=LABEL_SMOOTHING,
        reduction="mean",
    )
    targets = _low_loss_logit_targets(
        labels, logits.size(1), LABEL_SMOOTHING, logits_f.dtype
    )
    delta = targets - logits_f
    if DEBUG_TARGET_BACKPROP:
        with torch.no_grad():
            target_loss = F.cross_entropy(
                targets,
                labels,
                label_smoothing=LABEL_SMOOTHING,
                reduction="mean",
            )
            predicted = logits_f.argmax(dim=1)
            accuracy = (predicted == labels).float().mean()
            _debug_scalar("cross_entropy.loss", loss)
            _debug_scalar("cross_entropy.target_logits_loss", target_loss)
            _debug_scalar("cross_entropy.batch_accuracy", accuracy)
            _debug_tensor("cross_entropy.logits", logits_f)
            _debug_tensor("cross_entropy.target_logits", targets)
            _debug_tensor("cross_entropy.delta", delta)
    return loss, delta.to(dtype=logits.dtype)


def _forward_conv(module, x, cache):
    y = module(x)
    if cache is not None:
        name = _debug_module_name(module)
        if DEBUG_FORWARD_LOG:
            _debug_tensor(f"forward.{name}.input", x)
            _debug_tensor(f"forward.{name}.output", y)
        cache.append(dict(kind="conv", module=module, name=name, input=x.detach()))
    return y


def _forward_relu(module, x, cache):
    y = module(x)
    if cache is not None:
        name = _debug_module_name(module)
        if DEBUG_FORWARD_LOG:
            _debug_tensor(f"forward.{name}.relu_input", x)
            _debug_tensor(f"forward.{name}.relu_output", y)
            _debug_fraction(f"forward.{name}.relu_active_fraction", y > 0)
        cache.append(dict(kind="relu", name=name, input=x.detach()))
    return y


def _forward_pool(module, x, cache):
    if cache is None:
        return module(x)
    y, indices = F.max_pool2d(
        x,
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        module.ceil_mode,
        return_indices=True,
    )
    name = _debug_module_name(module)
    if DEBUG_FORWARD_LOG:
        _debug_tensor(f"forward.{name}.pool_input", x)
        _debug_tensor(f"forward.{name}.pool_output", y)
    cache.append(
        dict(kind="pool", module=module, name=name, input=x.detach(), indices=indices)
    )
    return y


def _forward_batchnorm(module, x, cache):
    if cache is None:
        return module(x)
    if not module.training:
        y = module(x)
        name = _debug_module_name(module)
        if DEBUG_FORWARD_LOG:
            _debug_tensor(f"forward.{name}.eval_bn_identity_input", x)
            _debug_tensor(f"forward.{name}.eval_bn_identity_output", y)
        cache.append(dict(kind="identity", name=name, input=x.detach()))
        return y

    axes = (0, 2, 3)
    mean = x.mean(dim=axes, keepdim=True)
    var = x.var(dim=axes, unbiased=False, keepdim=True)
    invstd = torch.rsqrt(var + module.eps)
    x_hat = (x - mean) * invstd
    y = F.batch_norm(
        x,
        module.running_mean,
        module.running_var,
        module.weight,
        module.bias,
        True,
        module.momentum,
        module.eps,
    )
    cache.append(
        dict(
            kind="batchnorm",
            module=module,
            name=_debug_module_name(module),
            input=x.detach(),
            x_hat=x_hat.detach(),
            invstd=invstd.detach(),
        )
    )
    if DEBUG_FORWARD_LOG:
        name = _debug_module_name(module)
        _debug_tensor(f"forward.{name}.bn_input", x)
        _debug_tensor(f"forward.{name}.bn_mean", mean, include_sample_rms=False)
        _debug_tensor(f"forward.{name}.bn_var", var, include_sample_rms=False)
        _debug_tensor(f"forward.{name}.bn_invstd", invstd, include_sample_rms=False)
        _debug_tensor(f"forward.{name}.bn_x_hat", x_hat)
        _debug_tensor(f"forward.{name}.bn_output", y)
    return y


def _forward_linear(module, x, cache, scale=1):
    y = module(x) / scale
    if cache is not None:
        name = _debug_module_name(module)
        if DEBUG_FORWARD_LOG:
            _debug_log(f"forward.{name}.linear_scale={scale}")
            _debug_tensor(f"forward.{name}.linear_input", x)
            _debug_tensor(f"forward.{name}.linear_output", y)
        cache.append(
            dict(kind="linear", module=module, name=name, input=x.detach(), scale=scale)
        )
    return y


def _forward_conv_group(group, x, cache):
    x = _forward_conv(group.conv1, x, cache)
    x = _bound_sample_rms(x)
    x = _forward_pool(group.pool, x, cache)
    x = _bound_sample_rms(x)
    x = _forward_batchnorm(group.norm1, x, cache)
    x = _bound_sample_rms(x)
    x = _forward_relu(group.activ, x, cache)
    x = _bound_sample_rms(x)
    x = _forward_conv(group.conv2, x, cache)
    x = _bound_sample_rms(x)
    x = _forward_batchnorm(group.norm2, x, cache)
    x = _bound_sample_rms(x)
    x = _forward_relu(group.activ, x, cache)
    x = _bound_sample_rms(x)
    return x


def _forward_impl(model, x, cache):
    x = _bound_sample_rms(x)
    x = _forward_conv(model.whiten, x, cache)
    x = _bound_sample_rms(x)
    for module in model.layers:
        if isinstance(module, ConvGroup):
            x = _forward_conv_group(module, x, cache)
        elif isinstance(module, nn.ReLU):
            x = _forward_relu(module, x, cache)
            x = _bound_sample_rms(x)
        elif isinstance(module, nn.MaxPool2d):
            x = _forward_pool(module, x, cache)
            x = _bound_sample_rms(x)
        else:
            raise TypeError(f"Unsupported module in manual forward: {type(module)}")
    if cache is not None:
        cache.append(dict(kind="flatten", input=x.detach(), shape=x.shape))
    x = x.view(len(x), -1)
    x = _bound_sample_rms(x)
    return _forward_linear(model.head, x, cache, scale=x.size(-1)), cache


def forward_no_cache(model, x):
    outputs, _ = _forward_impl(model, x, None)
    return outputs


def forward_with_cache(model, x):
    return _forward_impl(model, x, [])


def _linear_target_backward(module, x, dy, scale, name="linear"):
    if DEBUG_LAYER_LOG:
        _debug_log(f"{name}.linear_backward_begin scale={scale}")
        _debug_tensor(f"{name}.linear.x", x)
        _debug_tensor(f"{name}.linear.dy_in", dy)
        _debug_tensor(f"{name}.linear.weight_before", module.weight.data, include_sample_rms=False)
    x_mat = _bound_sample_rms(x).reshape(-1, x.shape[-1])
    dy_mat = dy.reshape(-1, dy.shape[-1])
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.linear.x_mat", x_mat, include_sample_rms=False)
        _debug_tensor(f"{name}.linear.dy_mat", dy_mat, include_sample_rms=False)

    if module.weight.requires_grad:
        weight_before = module.weight.data.detach().clone()
        xtx = x_mat.float().T @ x_mat.float()
        xtdy = x_mat.float().T @ dy_mat.float()
        delta_eff = _ridge_delta(
            xtx, xtdy, TARGET_HEAD_LAMBDA, name=f"{name}.linear_weight"
        )
        weight_update = (delta_eff * scale).T.to(dtype=module.weight.dtype)
        if DEBUG_LAYER_LOG:
            local_pred = x_mat.float() @ delta_eff
            local_residual = dy_mat.float() - local_pred
            _debug_tensor(f"{name}.linear.delta_eff", delta_eff, include_sample_rms=False)
            _debug_tensor(f"{name}.linear.local_predicted_dy", local_pred)
            _debug_tensor(f"{name}.linear.local_residual", local_residual)
            _debug_tensor(f"{name}.linear.weight_update", weight_update, include_sample_rms=False)
        module.weight.data.add_(weight_update)
        if DEBUG_LAYER_LOG:
            _debug_tensor(
                f"{name}.linear.actual_weight_delta",
                module.weight.data - weight_before,
                include_sample_rms=False,
            )
            _debug_tensor(f"{name}.linear.weight_after", module.weight.data, include_sample_rms=False)
    elif DEBUG_LAYER_LOG:
        _debug_log(f"{name}.linear.weight_update skipped_requires_grad_false=1")

    if module.bias is not None and module.bias.requires_grad:
        bias_before = module.bias.data.detach().clone()
        bias_update = dy_mat.mean(dim=0).mul(scale).to(module.bias.dtype)
        module.bias.data.add_(bias_update)
        _project_bias_(module.bias, name=f"{name}.bias")
        if DEBUG_LAYER_LOG:
            _debug_tensor(f"{name}.linear.bias_update", bias_update, include_sample_rms=False)
            _debug_tensor(
                f"{name}.linear.actual_bias_delta",
                module.bias.data - bias_before,
                include_sample_rms=False,
            )
    elif DEBUG_LAYER_LOG:
        _debug_log(f"{name}.linear.bias_update skipped=1")

    weight_eff = module.weight.data.T.float() / scale
    dx = _ridge_input_delta(dy_mat, weight_eff, name=f"{name}.linear_input")
    dx = dx.to(dtype=x.dtype).reshape_as(x)
    dx = _bound_delta_to_sample_rms(x, dx)
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.linear.weight_eff_after", weight_eff, include_sample_rms=False)
        _debug_tensor(f"{name}.linear.dx_out", dx)
    return dx


def _conv_weight_eff(module, group, in_per_group, out_per_group):
    kh, kw = _pair(module.kernel_size)
    patch_size = in_per_group * kh * kw
    out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
    return module.weight.data[out_slice].reshape(out_per_group, patch_size).T


def _conv_target_backward(module, x, dy, name="conv"):
    if DEBUG_LAYER_LOG:
        _debug_log(f"{name}.conv_backward_begin")
        _debug_tensor(f"{name}.conv.x", x)
        _debug_tensor(f"{name}.conv.dy_in", dy)
        _debug_tensor(f"{name}.conv.weight_before", module.weight.data, include_sample_rms=False)
    kernel_size = _pair(module.kernel_size)
    stride = _pair(module.stride)
    padding = _conv_padding_tuple(module.padding, module.kernel_size, module.dilation)
    dilation = _pair(module.dilation)
    groups = module.groups
    batch = len(x)
    in_channels = x.size(1)
    out_channels = dy.size(1)
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    kh, kw = kernel_size
    patch_size = in_per_group * kh * kw

    if module.weight.requires_grad:
        weight_before = module.weight.data.detach().clone()
        delta_weight = torch.zeros_like(module.weight.data)
        for group in range(groups):
            patch_slice = slice(group * patch_size, (group + 1) * patch_size)
            out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
            group_name = f"{name}.conv_group{group}"
            xtx = torch.zeros(
                patch_size, patch_size, device=x.device, dtype=torch.float32
            )
            xtdy = torch.zeros(
                patch_size, out_per_group, device=x.device, dtype=torch.float32
            )
            for start in range(0, batch, CONV_BATCH_CHUNK):
                end = min(start + CONV_BATCH_CHUNK, batch)
                patches = F.unfold(
                    _bound_sample_rms(x[start:end]),
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                    stride=stride,
                )
                x_group = (
                    patches[:, patch_slice, :]
                    .transpose(1, 2)
                    .reshape(-1, patch_size)
                    .float()
                )
                dy_group = (
                    dy[start:end, out_slice]
                    .flatten(2)
                    .transpose(1, 2)
                    .reshape(-1, out_per_group)
                    .float()
                )
                if DEBUG_CHUNK_LOG:
                    _debug_log(
                        "%s.chunk start=%d end=%d patch_rows=%d"
                        % (group_name, start, end, x_group.size(0))
                    )
                    _debug_tensor(
                        f"{group_name}.chunk{start}_{end}.x_group",
                        x_group,
                        include_sample_rms=False,
                    )
                    _debug_tensor(
                        f"{group_name}.chunk{start}_{end}.dy_group",
                        dy_group,
                        include_sample_rms=False,
                    )
                xtx.add_(x_group.T @ x_group)
                xtdy.add_(x_group.T @ dy_group)
            delta_eff = _ridge_delta(xtx, xtdy, name=f"{group_name}.weight")
            if DEBUG_LAYER_LOG:
                local_pred = x_group @ delta_eff
                local_residual = dy_group - local_pred
                _debug_tensor(
                    f"{group_name}.delta_eff",
                    delta_eff,
                    include_sample_rms=False,
                )
                _debug_tensor(f"{group_name}.last_chunk_predicted_dy", local_pred)
                _debug_tensor(f"{group_name}.last_chunk_residual", local_residual)
            delta_weight[out_slice] = delta_eff.T.reshape(
                out_per_group, in_per_group, kh, kw
            ).to(dtype=delta_weight.dtype)
            if DEBUG_LAYER_LOG:
                _debug_tensor(
                    f"{group_name}.delta_weight",
                    delta_weight[out_slice],
                    include_sample_rms=False,
                )
        module.weight.data.add_(delta_weight)
        if DEBUG_LAYER_LOG:
            _debug_tensor(f"{name}.conv.weight_update_before_projection", delta_weight, include_sample_rms=False)
        _project_weight_(module.weight, name=f"{name}.weight")
        if DEBUG_LAYER_LOG:
            _debug_tensor(
                f"{name}.conv.actual_weight_delta_after_projection",
                module.weight.data - weight_before,
                include_sample_rms=False,
            )
            _debug_tensor(f"{name}.conv.weight_after", module.weight.data, include_sample_rms=False)
    elif DEBUG_LAYER_LOG:
        _debug_log(f"{name}.conv.weight_update skipped_requires_grad_false=1")

    if module.bias is not None and module.bias.requires_grad:
        bias_before = module.bias.data.detach().clone()
        bias_update = dy.mean(dim=(0, 2, 3)).to(module.bias.dtype)
        module.bias.data.add_(bias_update)
        _project_bias_(module.bias, name=f"{name}.bias")
        if DEBUG_LAYER_LOG:
            _debug_tensor(f"{name}.conv.bias_update", bias_update, include_sample_rms=False)
            _debug_tensor(
                f"{name}.conv.actual_bias_delta",
                module.bias.data - bias_before,
                include_sample_rms=False,
            )
    elif DEBUG_LAYER_LOG:
        _debug_log(f"{name}.conv.bias_update skipped=1")

    weight_effs = [
        _conv_weight_eff(module, group, in_per_group, out_per_group)
        for group in range(groups)
    ]
    if DEBUG_LAYER_LOG:
        for group, weight_eff in enumerate(weight_effs):
            _debug_tensor(
                f"{name}.conv_group{group}.weight_eff_after",
                weight_eff,
                include_sample_rms=False,
            )
    dx = torch.zeros_like(x)
    for start in range(0, batch, CONV_BATCH_CHUNK):
        end = min(start + CONV_BATCH_CHUNK, batch)
        dx_chunk = torch.zeros_like(x[start:end])
        for group in range(groups):
            in_slice = slice(group * in_per_group, (group + 1) * in_per_group)
            out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
            dy_group = (
                dy[start:end, out_slice]
                .flatten(2)
                .transpose(1, 2)
                .reshape(-1, out_per_group)
                .float()
            )
            patch_delta = _ridge_input_delta(
                dy_group,
                weight_effs[group],
                name=f"{name}.conv_group{group}.input_chunk{start}_{end}",
            )
            patch_delta = (
                patch_delta.reshape(end - start, -1, patch_size)
                .transpose(1, 2)
                .to(dtype=x.dtype)
            )
            if DEBUG_CHUNK_LOG:
                _debug_tensor(
                    f"{name}.conv_group{group}.patch_delta_chunk{start}_{end}",
                    patch_delta,
                )
            folded = F.fold(
                patch_delta,
                output_size=x.shape[-2:],
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride,
            )
            overlap = F.fold(
                torch.ones_like(patch_delta),
                output_size=x.shape[-2:],
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride,
            ).clamp_min(1)
            dx_chunk[:, in_slice] = folded / overlap
        dx[start:end] = dx_chunk
        if DEBUG_CHUNK_LOG:
            _debug_tensor(f"{name}.conv.dx_chunk{start}_{end}", dx_chunk)
    dx = _bound_delta_to_sample_rms(x, dx)
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.conv.dx_out", dx)
    return dx


def _batchnorm_target_backward(module, x, x_hat, invstd, dy, name="batchnorm"):
    if DEBUG_LAYER_LOG:
        _debug_log(f"{name}.batchnorm_backward_begin")
        _debug_tensor(f"{name}.batchnorm.x", x)
        _debug_tensor(f"{name}.batchnorm.x_hat", x_hat)
        _debug_tensor(f"{name}.batchnorm.invstd", invstd, include_sample_rms=False)
        _debug_tensor(f"{name}.batchnorm.dy_in", dy)
    axes = (0, 2, 3)
    dtype = dy.dtype
    bias_delta = torch.zeros(1, x.size(1), 1, 1, device=x.device, dtype=dtype)
    if module.bias is not None and module.bias.requires_grad:
        old_bias = module.bias.data.detach().clone()
        module.bias.data.add_(dy.mean(dim=axes).to(module.bias.dtype))
        _project_bias_(module.bias, name=f"{name}.bias")
        bias_delta = (module.bias.data - old_bias).to(dtype=dtype).view(1, -1, 1, 1)
        if DEBUG_LAYER_LOG:
            _debug_tensor(f"{name}.batchnorm.bias_delta", bias_delta, include_sample_rms=False)
            _debug_tensor(
                f"{name}.batchnorm.bias_after",
                module.bias.data,
                include_sample_rms=False,
            )
    elif DEBUG_LAYER_LOG:
        _debug_log(f"{name}.batchnorm.bias_update skipped=1")

    target_hat = x_hat.to(dtype=dtype)
    dy_after_bias = dy - bias_delta
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.batchnorm.dy_after_bias", dy_after_bias)
    if module.weight is None:
        target_hat = target_hat + dy_after_bias
    else:
        gamma = module.weight.data.to(dtype=dtype).view(1, -1, 1, 1)
        if DEBUG_LAYER_LOG:
            _debug_tensor(f"{name}.batchnorm.gamma", gamma, include_sample_rms=False)
        safe_gamma = torch.where(
            gamma >= 0,
            gamma.abs().clamp_min(1e-12),
            -gamma.abs().clamp_min(1e-12),
        )
        target_hat = torch.where(
            gamma.abs() > 1e-12,
            target_hat + dy_after_bias / safe_gamma,
            target_hat,
        )

    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.batchnorm.target_hat_before_projection", target_hat)
    target_hat = target_hat - target_hat.mean(dim=axes, keepdim=True)
    target_rms = target_hat.square().mean(dim=axes, keepdim=True).sqrt()
    target_hat = torch.where(
        target_rms > 1e-12,
        target_hat / target_rms.clamp_min(1e-12),
        x_hat.to(dtype=dtype),
    )
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.batchnorm.target_hat_rms_before_projection", target_rms, include_sample_rms=False)
        _debug_tensor(f"{name}.batchnorm.target_hat_after_projection", target_hat)

    mean = x.float().mean(dim=axes, keepdim=True).to(dtype=dtype)
    std = invstd.to(dtype=dtype).reciprocal()
    x_target = mean + target_hat * std
    dx = x_target.to(dtype=x.dtype) - x
    dx = _bound_delta_to_sample_rms(x, dx)
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.batchnorm.mean", mean, include_sample_rms=False)
        _debug_tensor(f"{name}.batchnorm.std", std, include_sample_rms=False)
        _debug_tensor(f"{name}.batchnorm.x_target", x_target)
        _debug_tensor(f"{name}.batchnorm.dx_out", dx)
    return dx


def _relu_target_backward(x, dy, name="relu"):
    if DEBUG_LAYER_LOG:
        _debug_log(f"{name}.relu_backward_begin")
        _debug_tensor(f"{name}.relu.x", x)
        _debug_tensor(f"{name}.relu.dy_in", dy)
    active = x > 0
    inactive_positive_target = (~active) & (dy > 0)
    jump = torch.minimum(torch.ones_like(dy), dy - x)
    dx = torch.where(active, dy, torch.zeros_like(dy))
    dx = torch.where(inactive_positive_target, jump, dx)
    dx = _bound_delta_to_sample_rms(x, dx)
    if DEBUG_LAYER_LOG:
        _debug_fraction(f"{name}.relu.active_fraction", active)
        _debug_fraction(
            f"{name}.relu.inactive_positive_target_fraction",
            inactive_positive_target,
        )
        _debug_tensor(f"{name}.relu.jump", jump)
        _debug_tensor(f"{name}.relu.dx_out", dx)
    return dx


def _pool_target_backward(module, x, indices, dy, name="pool"):
    if DEBUG_LAYER_LOG:
        _debug_log(f"{name}.pool_backward_begin")
        _debug_tensor(f"{name}.pool.x", x)
        _debug_tensor(f"{name}.pool.dy_in", dy)
    dx = F.max_unpool2d(
        dy,
        indices,
        module.kernel_size,
        module.stride,
        module.padding,
        output_size=x.shape,
    )
    dx = _bound_delta_to_sample_rms(x, dx)
    if DEBUG_LAYER_LOG:
        _debug_tensor(f"{name}.pool.dx_out", dx)
    return dx


def target_backward(cache, delta):
    dy = delta
    if DEBUG_TARGET_BACKPROP:
        kinds = {}
        for record in cache:
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        _debug_log(f"target_backward_begin cache_len={len(cache)} kinds={kinds}")
        _debug_tensor("target_backward.initial_delta", dy)
    for reverse_index, record in enumerate(reversed(cache)):
        kind = record["kind"]
        name = record.get("name", kind)
        if DEBUG_LAYER_LOG:
            _debug_log(
                f"target_backward_layer_begin reverse_index={reverse_index} kind={kind} name={name}"
            )
            _debug_tensor(f"{name}.{kind}.dy_in_to_layer", dy)
        if kind == "linear":
            dy = _linear_target_backward(
                record["module"], record["input"], dy, record["scale"], name=name
            )
        elif kind == "flatten":
            x = record["input"]
            dy = _bound_delta_to_sample_rms(x, dy.reshape(record["shape"]))
            if DEBUG_LAYER_LOG:
                _debug_tensor(f"{name}.flatten.dx_out", dy)
        elif kind == "pool":
            dy = _pool_target_backward(
                record["module"], record["input"], record["indices"], dy, name=name
            )
        elif kind == "relu":
            dy = _relu_target_backward(record["input"], dy, name=name)
        elif kind == "batchnorm":
            dy = _batchnorm_target_backward(
                record["module"],
                record["input"],
                record["x_hat"],
                record["invstd"],
                dy,
                name=name,
            )
        elif kind == "conv":
            dy = _conv_target_backward(record["module"], record["input"], dy, name=name)
        elif kind == "identity":
            dy = _bound_delta_to_sample_rms(record["input"], dy)
            if DEBUG_LAYER_LOG:
                _debug_tensor(f"{name}.identity.dx_out", dy)
        else:
            raise TypeError(f"Unsupported cached op: {kind}")
        if DEBUG_LAYER_LOG:
            _debug_tensor(f"{name}.{kind}.dy_out_from_layer", dy)
    if DEBUG_TARGET_BACKPROP:
        _debug_tensor("target_backward.final_input_delta", dy)
    return dy


def target_train_step(model, inputs, labels, sweeps=1, step=None):
    model.train()
    with torch.no_grad():
        first_loss = None
        old_context = dict(_DEBUG_CONTEXT)
        for sweep in range(sweeps):
            _DEBUG_CONTEXT.update(step=step, sweep=sweep + 1, phase="forward")
            if DEBUG_TARGET_BACKPROP:
                _debug_log("target_train_sweep_begin")
                _debug_tensor("target_train.inputs", inputs)
            outputs, cache = forward_with_cache(model, inputs)
            loss, delta = cross_entropy_loss_and_delta(outputs, labels)
            if first_loss is None:
                first_loss = loss
            if torch.isfinite(loss):
                _DEBUG_CONTEXT.update(step=step, sweep=sweep + 1, phase="backward")
                target_backward(cache, delta)
                if DEBUG_TARGET_BACKPROP:
                    _debug_log("target_train_sweep_end finite_loss=1")
            else:
                _debug_log("target_train_sweep_skip_backward finite_loss=0")
        _DEBUG_CONTEXT.clear()
        _DEBUG_CONTEXT.update(old_context)
    return first_loss.item()


############################################
#                 Logging                  #
############################################


def log_eval(run, epoch, val_acc, time_seconds):
    pass


def log_final_eval(train25_loss, val_acc, tta_val_acc, time_seconds):
    pass


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


def evaluate_train_loss(model, batches):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for inputs, labels in batches:
            outputs = model(inputs)
            total_loss += F.cross_entropy(
                outputs.float(), labels, label_smoothing=0.2, reduction="sum"
            ).item()
            total_examples += len(labels)
    return total_loss / total_examples


############################################
#                Training                  #
############################################

OVERFIT_BATCH_SIZE = 2000
OVERFIT_STEPS = 20
LABEL_SMOOTHING = 0.2


def make_first_train_batch():
    train_loader = CifarLoader("cifar10", train=True, batch_size=OVERFIT_BATCH_SIZE)
    train_loader.shuffle = False
    inputs, labels = next(iter(train_loader))
    train_images = train_loader.normalized_images()[:5000]
    return inputs.detach(), labels.detach(), train_images


def first_batch_loss(model, inputs, labels):
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)
        loss = F.cross_entropy(
            outputs.float(),
            labels,
            label_smoothing=LABEL_SMOOTHING,
            reduction="mean",
        )
    return loss.item()


def overfit_first_batch(
    model,
    inputs,
    labels,
    train_images,
):
    set_training_seed()
    register_debug_names(model)
    _DEBUG_CONTEXT.update(step=0, sweep=None, phase="setup")
    if DEBUG_TARGET_BACKPROP:
        _debug_log(
            "config | forward=%d | layer=%d | matrix=%d | chunk=%d | pinv=%d | param=%d"
            % (
                DEBUG_FORWARD_LOG,
                DEBUG_LAYER_LOG,
                DEBUG_MATRIX_LOG,
                DEBUG_CHUNK_LOG,
                DEBUG_PINV_LOG,
                DEBUG_PARAM_LOG,
            )
        )
        _debug_tensor("setup.inputs", inputs)
        _debug_tensor("setup.train_images_for_whiten", train_images)
    model.reset()
    _debug_model_parameters(model, "after_reset_before_whiten")
    model.init_whiten(train_images)
    _debug_model_parameters(model, "after_whiten_init_before_projection")
    project_trainable_parameters(model)
    _debug_model_parameters(model, "after_initial_projection")

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    _DEBUG_CONTEXT.update(step=0, sweep=None, phase="initial_eval")
    initial_loss = first_batch_loss(model, inputs, labels)
    _debug_scalar("initial_loss", initial_loss)
    final_loss = float("inf")
    last_step_loss = float("inf")
    completed_steps = 0

    start_timer()
    for step in range(OVERFIT_STEPS):
        _DEBUG_CONTEXT.update(step=step + 1, sweep=None, phase="step_start")
        if DEBUG_TARGET_BACKPROP:
            _debug_log(f"target_backprop_outer_step_begin step={step + 1}")
        pre_step_loss = target_train_step(
            model,
            inputs,
            labels,
            sweeps=INNER_TARGET_SWEEPS,
            step=step + 1,
        )
        _DEBUG_CONTEXT.update(step=step + 1, sweep=None, phase="post_eval")
        last_step_loss = first_batch_loss(model, inputs, labels)
        _debug_scalar("step_pre_update_loss", pre_step_loss)
        _debug_scalar("step_post_update_loss", last_step_loss)
        print(
            "loss_step %02d/%d loss=%.6f"
            % (step + 1, OVERFIT_STEPS, last_step_loss),
            flush=True,
        )
        _debug_model_parameters(model, f"after_step_{step + 1}")
        if not isfinite(last_step_loss):
            break
        completed_steps = step + 1
    stop_timer()

    if completed_steps == OVERFIT_STEPS:
        _DEBUG_CONTEXT.update(step=completed_steps, sweep=None, phase="final_eval")
        final_loss = first_batch_loss(model, inputs, labels)
        _debug_scalar("final_loss", final_loss)

    return dict(
        batch_size=OVERFIT_BATCH_SIZE,
        steps=completed_steps,
        target_steps=OVERFIT_STEPS,
        initial_train_loss=initial_loss,
        last_step_loss=last_step_loss,
        final_train_loss=final_loss,
        target_head_lambda=TARGET_HEAD_LAMBDA,
        target_x_lambda=TARGET_X_LAMBDA,
        inner_target_sweeps=INNER_TARGET_SWEEPS,
        pinv_rtol=PINV_RTOL,
        conv_batch_chunk=CONV_BATCH_CHUNK,
        time_seconds=time_seconds,
    )


def main():
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")
    inputs, labels, train_images = make_first_train_batch()

    result = overfit_first_batch(
        model=model,
        inputs=inputs,
        labels=labels,
        train_images=train_images,
    )

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, result=result), log_path)


if __name__ == "__main__":
    main()
