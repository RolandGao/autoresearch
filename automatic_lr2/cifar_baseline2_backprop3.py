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


_LOG_CONTEXT = dict(step=None, sweep=None, phase=None)


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


def _module_log_name(module):
    return getattr(module, "_log_name", module.__class__.__name__)


def register_operation_names(model):
    for name, module in model.named_modules():
        module._log_name = name or "model"


def _log_context_fields():
    fields = []
    for key in ("step", "sweep", "phase"):
        value = _LOG_CONTEXT.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    return fields


def _format_log_value(value):
    if value is None:
        return "none"
    value = float(value)
    if not isfinite(value):
        return str(value)
    return "%.6g" % value


def _tensor_norm(tensor):
    if tensor is None:
        return None
    with torch.no_grad():
        value = tensor.detach().float()
        if value.numel() == 0:
            return 0.0
        return value.norm().item()


def _print_log_line(tag, fields):
    print(" ".join([tag] + _log_context_fields() + fields), flush=True)


def _log_forward_norm(kind, name, op_index, activation):
    _print_log_line(
        "forward_norm",
        [
            f"op={op_index}",
            f"kind={kind}",
            f"name={name}",
            f"activation_norm={_format_log_value(_tensor_norm(activation))}",
        ],
    )


def _log_backward_norm(
    kind,
    name,
    op_index,
    dy,
    dx,
    dw_norm,
    train_loss_before,
    train_loss_after,
):
    _print_log_line(
        "backward_norm",
        [
            f"op={op_index}",
            f"kind={kind}",
            f"name={name}",
            f"dy_norm={_format_log_value(_tensor_norm(dy))}",
            f"dw_norm={_format_log_value(dw_norm)}",
            f"dx_norm={_format_log_value(_tensor_norm(dx))}",
            f"train_loss_before_dx_dw={_format_log_value(train_loss_before)}",
            f"train_loss_after_dx_dw={_format_log_value(train_loss_after)}",
        ],
    )


def _module_param_snapshot(module):
    if module is None:
        return {}
    snapshot = {}
    for name in ("weight", "bias"):
        param = getattr(module, name, None)
        if param is not None:
            snapshot[name] = param.data.detach().clone()
    return snapshot


def _module_param_delta_norm(module, snapshot):
    if module is None or not snapshot:
        return 0.0
    total = 0.0
    for name, before in snapshot.items():
        param = getattr(module, name, None)
        if param is None:
            continue
        delta = param.data.detach().float() - before.float()
        total += delta.square().sum().item()
    return total**0.5


def _loss_preserving_mode(model, inputs, labels):
    training_states = [(module, module.training) for module in model.modules()]
    try:
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
    finally:
        for module, training in training_states:
            module.train(training)


def _operation_train_loss(model, inputs, labels):
    if model is None or inputs is None or labels is None:
        return None
    return _loss_preserving_mode(model, inputs, labels)


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


def _safe_pinv(x):
    x = torch.nan_to_num(x.float())
    last_error = None
    for rtol in (PINV_RTOL, 1e-4, 1e-3):
        try:
            return torch.linalg.pinv(x, rtol=rtol)
        except RuntimeError as err:
            last_error = err
    raise last_error


def _ridge_delta(xtx, xtdy, lambda_value=0.0):
    if lambda_value != 0:
        eye = torch.eye(xtx.size(0), device=xtx.device, dtype=xtx.dtype)
        xtx = xtx + lambda_value * eye
    return _safe_pinv(xtx) @ xtdy.float()


def _ridge_input_delta(dy, weight_eff):
    weight_eff = weight_eff.float()
    dy = dy.float()
    if TARGET_X_LAMBDA == 0:
        return dy @ _safe_pinv(weight_eff)
    gram = weight_eff.T @ weight_eff
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    solve_matrix = gram + TARGET_X_LAMBDA * eye
    return dy @ _safe_pinv(solve_matrix) @ weight_eff.T


def _project_weight_(p, max_rms=1.0):
    if p is None or not p.requires_grad:
        return
    rms = p.data.float().square().mean().sqrt()
    if torch.isfinite(rms) and rms > 0:
        scale = max_rms / rms
        p.data.mul_(scale.to(dtype=p.data.dtype))


def _project_bias_(p):
    if p is not None and p.requires_grad:
        p.data.clamp_(-1, 1)


def project_trainable_parameters(model):
    for module in model.modules():
        if module is getattr(model, "head", None):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        _project_weight_(weight)
        _project_bias_(bias)


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
    return loss, delta.to(dtype=logits.dtype)


def _forward_conv(module, x, cache):
    y = module(x)
    if cache is not None:
        name = _module_log_name(module)
        op_index = len(cache)
        _log_forward_norm("conv", name, op_index, y)
        cache.append(
            dict(kind="conv", module=module, name=name, input=x.detach(), op_index=op_index)
        )
    return y


def _forward_relu(module, x, cache):
    y = module(x)
    if cache is not None:
        name = _module_log_name(module)
        op_index = len(cache)
        _log_forward_norm("relu", name, op_index, y)
        cache.append(dict(kind="relu", name=name, input=x.detach(), op_index=op_index))
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
    name = _module_log_name(module)
    op_index = len(cache)
    _log_forward_norm("pool", name, op_index, y)
    cache.append(
        dict(
            kind="pool",
            module=module,
            name=name,
            input=x.detach(),
            indices=indices,
            op_index=op_index,
        )
    )
    return y


def _forward_batchnorm(module, x, cache):
    if cache is None:
        return module(x)
    if not module.training:
        y = module(x)
        name = _module_log_name(module)
        op_index = len(cache)
        _log_forward_norm("identity", name, op_index, y)
        cache.append(dict(kind="identity", name=name, input=x.detach(), op_index=op_index))
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
            name=_module_log_name(module),
            input=x.detach(),
            x_hat=x_hat.detach(),
            invstd=invstd.detach(),
            op_index=len(cache),
        )
    )
    _log_forward_norm("batchnorm", cache[-1]["name"], cache[-1]["op_index"], y)
    return y


def _forward_linear(module, x, cache, scale=1):
    y = module(x) / scale
    if cache is not None:
        name = _module_log_name(module)
        op_index = len(cache)
        _log_forward_norm("linear", name, op_index, y)
        cache.append(
            dict(
                kind="linear",
                module=module,
                name=name,
                input=x.detach(),
                scale=scale,
                op_index=op_index,
            )
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
        op_index = len(cache)
        cache.append(
            dict(
                kind="flatten",
                name="flatten",
                input=x.detach(),
                shape=x.shape,
                op_index=op_index,
            )
        )
    x = x.view(len(x), -1)
    if cache is not None:
        _log_forward_norm("flatten", "flatten", op_index, x)
    x = _bound_sample_rms(x)
    return _forward_linear(model.head, x, cache, scale=x.size(-1)), cache


def forward_no_cache(model, x):
    outputs, _ = _forward_impl(model, x, None)
    return outputs


def forward_with_cache(model, x):
    return _forward_impl(model, x, [])


def _linear_target_backward(module, x, dy, scale):
    x_mat = _bound_sample_rms(x).reshape(-1, x.shape[-1])
    dy_mat = dy.reshape(-1, dy.shape[-1])

    if module.weight.requires_grad:
        xtx = x_mat.float().T @ x_mat.float()
        xtdy = x_mat.float().T @ dy_mat.float()
        delta_eff = _ridge_delta(xtx, xtdy, TARGET_HEAD_LAMBDA)
        weight_update = (delta_eff * scale).T.to(dtype=module.weight.dtype)
        module.weight.data.add_(weight_update)

    if module.bias is not None and module.bias.requires_grad:
        bias_update = dy_mat.mean(dim=0).mul(scale).to(module.bias.dtype)
        module.bias.data.add_(bias_update)
        _project_bias_(module.bias)

    weight_eff = module.weight.data.T.float() / scale
    dx = _ridge_input_delta(dy_mat, weight_eff)
    dx = dx.to(dtype=x.dtype).reshape_as(x)
    return _bound_delta_to_sample_rms(x, dx)


def _conv_weight_eff(module, group, in_per_group, out_per_group):
    kh, kw = _pair(module.kernel_size)
    patch_size = in_per_group * kh * kw
    out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
    return module.weight.data[out_slice].reshape(out_per_group, patch_size).T


def _conv_target_backward(module, x, dy):
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
        delta_weight = torch.zeros_like(module.weight.data)
        for group in range(groups):
            patch_slice = slice(group * patch_size, (group + 1) * patch_size)
            out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
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
                xtx.add_(x_group.T @ x_group)
                xtdy.add_(x_group.T @ dy_group)
            delta_eff = _ridge_delta(xtx, xtdy)
            delta_weight[out_slice] = delta_eff.T.reshape(
                out_per_group, in_per_group, kh, kw
            ).to(dtype=delta_weight.dtype)
        module.weight.data.add_(delta_weight)
        _project_weight_(module.weight)

    if module.bias is not None and module.bias.requires_grad:
        bias_update = dy.mean(dim=(0, 2, 3)).to(module.bias.dtype)
        module.bias.data.add_(bias_update)
        _project_bias_(module.bias)

    weight_effs = [
        _conv_weight_eff(module, group, in_per_group, out_per_group)
        for group in range(groups)
    ]
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
            )
            patch_delta = (
                patch_delta.reshape(end - start, -1, patch_size)
                .transpose(1, 2)
                .to(dtype=x.dtype)
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
    return _bound_delta_to_sample_rms(x, dx)


def _batchnorm_target_backward(module, x, x_hat, invstd, dy):
    axes = (0, 2, 3)
    dtype = dy.dtype
    bias_delta = torch.zeros(1, x.size(1), 1, 1, device=x.device, dtype=dtype)
    if module.bias is not None and module.bias.requires_grad:
        old_bias = module.bias.data.detach().clone()
        module.bias.data.add_(dy.mean(dim=axes).to(module.bias.dtype))
        _project_bias_(module.bias)
        bias_delta = (module.bias.data - old_bias).to(dtype=dtype).view(1, -1, 1, 1)

    target_hat = x_hat.to(dtype=dtype)
    dy_after_bias = dy - bias_delta
    if module.weight is None:
        target_hat = target_hat + dy_after_bias
    else:
        gamma = module.weight.data.to(dtype=dtype).view(1, -1, 1, 1)
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

    target_hat = target_hat - target_hat.mean(dim=axes, keepdim=True)
    target_rms = target_hat.square().mean(dim=axes, keepdim=True).sqrt()
    target_hat = torch.where(
        target_rms > 1e-12,
        target_hat / target_rms.clamp_min(1e-12),
        x_hat.to(dtype=dtype),
    )

    mean = x.float().mean(dim=axes, keepdim=True).to(dtype=dtype)
    std = invstd.to(dtype=dtype).reciprocal()
    x_target = mean + target_hat * std
    dx = x_target.to(dtype=x.dtype) - x
    return _bound_delta_to_sample_rms(x, dx)


def _relu_target_backward(x, dy):
    active = x > 0
    inactive_positive_target = (~active) & (dy > 0)
    jump = torch.minimum(torch.ones_like(dy), dy - x)
    dx = torch.where(active, dy, torch.zeros_like(dy))
    dx = torch.where(inactive_positive_target, jump, dx)
    return _bound_delta_to_sample_rms(x, dx)


def _pool_target_backward(module, x, indices, dy):
    dx = F.max_unpool2d(
        dy,
        indices,
        module.kernel_size,
        module.stride,
        module.padding,
        output_size=x.shape,
    )
    return _bound_delta_to_sample_rms(x, dx)


def target_backward(cache, delta, model=None, train_inputs=None, train_labels=None):
    dy = delta
    for reverse_index, record in enumerate(reversed(cache)):
        kind = record["kind"]
        name = record.get("name", kind)
        op_index = record.get("op_index", len(cache) - reverse_index - 1)
        dy_in = dy
        module = record.get("module")
        param_snapshot = _module_param_snapshot(module)
        train_loss_before = _operation_train_loss(model, train_inputs, train_labels)
        if kind == "linear":
            dy = _linear_target_backward(
                record["module"], record["input"], dy, record["scale"]
            )
        elif kind == "flatten":
            x = record["input"]
            dy = _bound_delta_to_sample_rms(x, dy.reshape(record["shape"]))
        elif kind == "pool":
            dy = _pool_target_backward(
                record["module"], record["input"], record["indices"], dy
            )
        elif kind == "relu":
            dy = _relu_target_backward(record["input"], dy)
        elif kind == "batchnorm":
            dy = _batchnorm_target_backward(
                record["module"],
                record["input"],
                record["x_hat"],
                record["invstd"],
                dy,
            )
        elif kind == "conv":
            dy = _conv_target_backward(record["module"], record["input"], dy)
        elif kind == "identity":
            dy = _bound_delta_to_sample_rms(record["input"], dy)
        else:
            raise TypeError(f"Unsupported cached op: {kind}")
        dw_norm = _module_param_delta_norm(module, param_snapshot)
        train_loss_after = _operation_train_loss(model, train_inputs, train_labels)
        _log_backward_norm(
            kind,
            name,
            op_index,
            dy_in,
            dy,
            dw_norm,
            train_loss_before,
            train_loss_after,
        )
    return dy


def target_train_step(model, inputs, labels, sweeps=1, step=None):
    model.train()
    with torch.no_grad():
        first_loss = None
        old_context = dict(_LOG_CONTEXT)
        for sweep in range(sweeps):
            _LOG_CONTEXT.update(step=step, sweep=sweep + 1, phase="forward")
            outputs, cache = forward_with_cache(model, inputs)
            loss, delta = cross_entropy_loss_and_delta(outputs, labels)
            if first_loss is None:
                first_loss = loss
            if torch.isfinite(loss):
                _LOG_CONTEXT.update(step=step, sweep=sweep + 1, phase="backward")
                target_backward(
                    cache,
                    delta,
                    model=model,
                    train_inputs=inputs,
                    train_labels=labels,
                )
        _LOG_CONTEXT.clear()
        _LOG_CONTEXT.update(old_context)
    return first_loss.item()


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
    register_operation_names(model)
    _LOG_CONTEXT.update(step=0, sweep=None, phase="setup")
    model.reset()
    model.init_whiten(train_images)
    project_trainable_parameters(model)

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

    _LOG_CONTEXT.update(step=0, sweep=None, phase="initial_eval")
    initial_loss = first_batch_loss(model, inputs, labels)
    final_loss = float("inf")
    last_step_loss = float("inf")
    completed_steps = 0

    start_timer()
    for step in range(OVERFIT_STEPS):
        _LOG_CONTEXT.update(step=step + 1, sweep=None, phase="step_start")
        target_train_step(
            model,
            inputs,
            labels,
            sweeps=INNER_TARGET_SWEEPS,
            step=step + 1,
        )
        _LOG_CONTEXT.update(step=step + 1, sweep=None, phase="post_eval")
        last_step_loss = first_batch_loss(model, inputs, labels)
        print(
            "loss_step %02d/%d loss=%.6f"
            % (step + 1, OVERFIT_STEPS, last_step_loss),
            flush=True,
        )
        if not isfinite(last_step_loss):
            break
        completed_steps = step + 1
    stop_timer()

    if completed_steps == OVERFIT_STEPS:
        _LOG_CONTEXT.update(step=completed_steps, sweep=None, phase="final_eval")
        final_loss = first_batch_loss(model, inputs, labels)

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
