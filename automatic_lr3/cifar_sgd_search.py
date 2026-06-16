"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
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
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
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
NORMAL_EQUATION_PINV_RTOL = 1e-5


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


#############################################
#               Muon optimizer              #
#############################################


def normalize_rows(G):
    assert len(G.shape) == 2
    eps = torch.finfo(G.dtype).eps
    row_normalized = G / G.norm(dim=1, keepdim=True).clamp_min(eps)
    row_normalized = torch.nan_to_num(row_normalized)
    return row_normalized * (min(G.shape) / G.size(0)) ** 0.5


def normalize_rows_max(G):
    assert len(G.shape) == 2
    row_normalized = G / G.norm(dim=1, keepdim=True)
    return row_normalized * (max(G.shape) / G.size(0)) ** 0.5


def normalize_matrix(G):
    assert len(G.shape) == 2
    return G * (min(G.shape) ** 0.5 / G.norm())


def normalize_matrix_max(G):
    assert len(G.shape) == 2
    return G * (max(G.shape) ** 0.5 / G.norm())


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
    X_norm = X.norm()
    if not torch.isfinite(X_norm) or X_norm == 0:
        return torch.zeros_like(G)
    X /= X_norm + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def zeropower_via_newtonschulz5_max(G, steps=3, eps=0):
    assert len(G.shape) == 2
    scale = (max(G.shape) / min(G.shape)) ** 0.5
    return zeropower_via_newtonschulz5(G, steps=steps, eps=eps) * scale


if USE_COMPILED_MUON:
    normalize_rows = torch.compile(normalize_rows)
    normalize_rows_max = torch.compile(normalize_rows_max)
    normalize_matrix = torch.compile(normalize_matrix)
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    normalize_matrix_max = torch.compile(normalize_matrix_max)
    zeropower_via_newtonschulz5_max = torch.compile(zeropower_via_newtonschulz5_max)


class Muon(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, momentum=0, nesterov=False, update_fn=normalize_rows
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, update_fn=update_fn
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            update_fn = group["update_fn"]
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
                update = update_fn(g.reshape(len(g), -1)).view(g.shape)
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
class InverseBatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mean, var, eps):
        std = torch.sqrt(var + eps)
        x_hat = (x - mean) / std
        ctx.save_for_backward(x_hat, std, weight)
        y = x_hat * weight.view(1, -1, 1, 1)
        return y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, std, weight = ctx.saved_tensors
        reduce_dims = (0, 2, 3)
        grad_x_hat = grad_output * weight.view(1, -1, 1, 1)
        tangent = grad_x_hat - grad_x_hat.mean(
            dim=reduce_dims, keepdim=True
        ) - x_hat * (grad_x_hat * x_hat).mean(dim=reduce_dims, keepdim=True)
        grad_x = tangent * std
        grad_weight = (grad_output * x_hat).sum(dim=reduce_dims)
        grad_bias = grad_output.sum(dim=reduce_dims)
        return grad_x, grad_weight, grad_bias, None, None, None


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12, norm_inverse=False):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.norm_inverse = norm_inverse
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

    def forward(self, x):
        if not self.norm_inverse or not self.training:
            return super().forward(x)

        self._check_input_dim(x)
        reduce_dims = (0, 2, 3)
        mean = x.mean(dim=reduce_dims, keepdim=True)
        var = x.var(dim=reduce_dims, keepdim=True, unbiased=False)

        if self.track_running_stats:
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                momentum = self.momentum
                if momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
                count = x.numel() // x.size(1)
                unbiased_var = var.view(-1) * (count / max(count - 1, 1))
                self.running_mean.lerp_(mean.view(-1), momentum)
                self.running_var.lerp_(unbiased_var, momentum)

        return InverseBatchNorm2dFunction.apply(
            x,
            self.weight,
            self.bias,
            mean.detach(),
            var.detach(),
            self.eps,
        )


def normal_equation_conv2d_weight_grad(
    x, grad_output, weight_shape, stride, padding, dilation
):
    cols = F.unfold(
        x,
        kernel_size=weight_shape[2:],
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    x_matrix = cols.transpose(1, 2).reshape(-1, cols.size(1))
    y_matrix = grad_output.flatten(2).transpose(1, 2).reshape(-1, grad_output.size(1))

    solve_dtype = torch.float32 if x_matrix.dtype != torch.float64 else torch.float64
    x_solve = x_matrix.to(solve_dtype)
    y_solve = y_matrix.to(solve_dtype)
    xtx = x_solve.T @ x_solve
    xty = x_solve.T @ y_solve
    solution, info = torch.linalg.solve_ex(xtx, xty)
    if torch.any(info != 0):
        solution = torch.linalg.pinv(xtx, rtol=NORMAL_EQUATION_PINV_RTOL) @ xty
    return solution.T.reshape(weight_shape).to(grad_output.dtype)


def pseudo_inverse_conv2d_input_grad(
    input_shape, weight, grad_output, stride, padding, dilation
):
    weight_matrix = weight.reshape(weight.size(0), -1)
    solve_dtype = torch.float32 if weight_matrix.dtype != torch.float64 else torch.float64
    weight_solve = weight_matrix.to(solve_dtype)
    try:
        svd_kwargs = dict(full_matrices=False)
        if weight_solve.is_cuda:
            svd_kwargs["driver"] = "gesvd"
        U, S, Vh = torch.linalg.svd(weight_solve, **svd_kwargs)
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(weight_solve.cpu(), full_matrices=False)
        U = U.to(weight_solve.device)
        S = S.to(weight_solve.device)
        Vh = Vh.to(weight_solve.device)

    cutoff = max(weight_solve.shape) * torch.finfo(S.dtype).eps * S.max()
    S_inv = torch.where(S > cutoff, S.reciprocal(), torch.zeros_like(S))
    weight_pinv = ((Vh.mH * S_inv.unsqueeze(0)) @ U.mH).to(grad_output.dtype)
    grad_output_cols = grad_output.flatten(2)
    grad_input_cols = torch.einsum("ko,nol->nkl", weight_pinv, grad_output_cols)
    return F.fold(
        grad_input_cols,
        output_size=input_shape[2:],
        kernel_size=weight.shape[2:],
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


class MuonConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        normal_equation_dw,
        pseudo_inverse_dx,
    ):
        if groups != 1:
            raise ValueError("MuonConv2dFunction only supports groups=1")
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.normal_equation_dw = normal_equation_dw
        ctx.pseudo_inverse_dx = pseudo_inverse_dx
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            if ctx.pseudo_inverse_dx:
                grad_x = pseudo_inverse_conv2d_input_grad(
                    x.shape,
                    weight,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                )
            else:
                grad_x = torch.nn.grad.conv2d_input(
                    x.shape,
                    weight,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                    ctx.groups,
                )

        if ctx.needs_input_grad[1]:
            if ctx.normal_equation_dw:
                grad_weight = normal_equation_conv2d_weight_grad(
                    x,
                    grad_output,
                    weight.shape,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                )
            else:
                grad_weight = torch.nn.grad.conv2d_weight(
                    x,
                    weight.shape,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                    ctx.groups,
                )

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None


class Conv(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        normal_equation_dw=False,
        pseudo_inverse_dx=False,
    ):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.normal_equation_dw = normal_equation_dw
        self.pseudo_inverse_dx = pseudo_inverse_dx

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])

    def forward(self, x):
        if not self.normal_equation_dw and not self.pseudo_inverse_dx:
            return super().forward(x)
        return MuonConv2dFunction.apply(
            x,
            self.weight,
            self.bias,
            _pair(self.stride),
            _pair(self.padding),
            _pair(self.dilation),
            self.groups,
            self.normal_equation_dw,
            self.pseudo_inverse_dx,
        )


class ConvGroup(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        normal_equation_dw=False,
        pseudo_inverse_dx=False,
        norm_inverse=False,
        remove_norm=False,
    ):
        super().__init__()
        self.conv1 = Conv(
            channels_in,
            channels_out,
            normal_equation_dw=normal_equation_dw,
            pseudo_inverse_dx=pseudo_inverse_dx,
        )
        self.pool = nn.MaxPool2d(2)
        self.norm1 = (
            nn.Identity()
            if remove_norm
            else BatchNorm(channels_out, norm_inverse=norm_inverse)
        )
        self.conv2 = Conv(
            channels_out,
            channels_out,
            normal_equation_dw=normal_equation_dw,
            pseudo_inverse_dx=pseudo_inverse_dx,
        )
        self.norm2 = (
            nn.Identity()
            if remove_norm
            else BatchNorm(channels_out, norm_inverse=norm_inverse)
        )
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
    def __init__(
        self,
        normal_equation_dw=False,
        pseudo_inverse_dx=False,
        norm_inverse=False,
        remove_norm=False,
    ):
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
            ConvGroup(
                whiten_width,
                widths["block1"],
                normal_equation_dw=normal_equation_dw,
                pseudo_inverse_dx=pseudo_inverse_dx,
                norm_inverse=norm_inverse,
                remove_norm=remove_norm,
            ),
            ConvGroup(
                widths["block1"],
                widths["block2"],
                normal_equation_dw=normal_equation_dw,
                pseudo_inverse_dx=pseudo_inverse_dx,
                norm_inverse=norm_inverse,
                remove_norm=remove_norm,
            ),
            ConvGroup(
                widths["block2"],
                widths["block3"],
                normal_equation_dw=normal_equation_dw,
                pseudo_inverse_dx=pseudo_inverse_dx,
                norm_inverse=norm_inverse,
                remove_norm=remove_norm,
            ),
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


def log_step(epoch, step, total_steps, loss, head_lr, muon_lr):
    print(
        f"step={step}/{total_steps} epoch={epoch} "
        f"loss={loss:.4f} head_lr={head_lr:.6g} muon_lr={muon_lr:.6g}",
        flush=True,
    )


def log_eval(run, epoch, val_acc, time_seconds):
    run_info = f" run={run}" if run is not None else ""
    print(
        f"eval{run_info} epoch={epoch} val_acc={val_acc:.4f} "
        f"time_seconds={time_seconds:.4f}",
        flush=True,
    )


def log_final_eval(train25_loss, val_acc, tta_val_acc, time_seconds):
    print(
        f"eval epoch=final 25batch_train_loss={train25_loss:.4f} "
        f"val_acc={val_acc:.4f} tta_val_acc={tta_val_acc:.4f} "
        f"time_seconds={time_seconds:.4f}",
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

TRAIN_EVAL_BATCHES = 25
LR_SEARCH_FACTOR = 0.8
LR_SEARCH_SIG_FIGS = 2
BATCH_CONFIGS = [
    dict(batch_size=2000),
]
UPDATE_CONFIGS = [
    dict(update_name="row_norm", update_fn=normalize_rows),
    dict(
        update_name="zeropower_via_newtonschulz5",
        update_fn=zeropower_via_newtonschulz5,
    ),
]
MANUAL_MUON_LRS = {
    "row_norm": {
        2000: 0.2,
    },
    "zeropower_via_newtonschulz5": {
        2000: 0.2,
    },
}
HPARAM_CONFIGS = [
    dict(
        normal_equation_dw=normal_equation_dw,
        pseudo_inverse_dx=pseudo_inverse_dx,
        norm_inverse=norm_inverse,
        remove_norm=remove_norm,
    )
    for normal_equation_dw in (False, True)
    for pseudo_inverse_dx in (False, True)
    for remove_norm in (False, True)
    for norm_inverse in ((False,) if remove_norm else (False, True))
]
RUN_CONFIGS = [
    dict(
        **batch_config,
        **update_config,
        **hparam_config,
        muon_lr=MANUAL_MUON_LRS[update_config["update_name"]][
            batch_config["batch_size"]
        ],
    )
    for batch_config in BATCH_CONFIGS
    for update_config in UPDATE_CONFIGS
    for hparam_config in HPARAM_CONFIGS
]


def rounded_lr(value):
    return float(f"{value:.{LR_SEARCH_SIG_FIGS}g}")


def lr_key(value):
    return f"{rounded_lr(value):.{LR_SEARCH_SIG_FIGS}g}"


def lr_at_offset(initial_lr, offset):
    return rounded_lr(initial_lr * LR_SEARCH_FACTOR**offset)


def hparam_label(config):
    parts = []
    for name in (
        "normal_equation_dw",
        "pseudo_inverse_dx",
        "norm_inverse",
        "remove_norm",
    ):
        parts.append("%s_%d" % (name, config[name]))
    return "_".join(parts)


def best_offset(offsets, results_by_offset):
    return max(
        offsets,
        key=lambda offset: (
            results_by_offset[offset]["tta_val_acc"],
            -abs(offset),
        ),
    )


def find_integer_middle(evaluate):
    offsets = [-1.0, 0.0, 1.0]
    for offset in offsets:
        evaluate(offset)

    middle = best_offset(offsets, evaluate.results_by_offset)
    if middle == 0.0:
        return middle

    direction = 1.0 if middle > 0.0 else -1.0
    while True:
        next_offset = middle + direction
        evaluate(next_offset)
        candidates = [middle - direction, middle, next_offset]
        new_middle = best_offset(candidates, evaluate.results_by_offset)
        if new_middle == middle:
            return middle
        middle = new_middle


def search_initial_lr(search_index, model, config):
    cache = {}
    results_by_offset = {}
    evaluations = []

    def evaluate(offset):
        lr = lr_at_offset(config["muon_lr"], offset)
        key = lr_key(lr)
        if key not in cache:
            hparams = hparam_label(config)
            run_name = (
                "search%d_bs%d_update_%s_%s_k%g_lr%s"
                % (
                    search_index,
                    config["batch_size"],
                    config["update_name"],
                    hparams,
                    offset,
                    key,
                )
            )
            print(
                "lr_search_eval %s initial_lr=%.6g rounded_lr=%s"
                % (run_name, config["muon_lr"], key),
                flush=True,
            )
            cache[key] = main(
                run_name,
                model,
                config["batch_size"],
                lr,
                config["update_name"],
                config["update_fn"],
                config["normal_equation_dw"],
                config["pseudo_inverse_dx"],
                config["norm_inverse"],
                config["remove_norm"],
            )
            cache[key]["search_offset"] = offset
            cache[key]["rounded_muon_lr"] = lr
            evaluations.append(cache[key])
        else:
            print(
                "lr_search_cache_hit search=%d batch_size=%d update=%s %s "
                "k=%g rounded_lr=%s"
                % (
                    search_index,
                    config["batch_size"],
                    config["update_name"],
                    hparam_label(config),
                    offset,
                    key,
                ),
                flush=True,
            )
        results_by_offset[offset] = cache[key]
        return cache[key]

    evaluate.results_by_offset = results_by_offset

    best_final = find_integer_middle(evaluate)
    best_result = results_by_offset[best_final]
    evaluation_snapshots = [dict(result) for result in evaluations]
    best_result["best_search_offset"] = best_final
    best_result["initial_muon_lr"] = config["muon_lr"]
    best_result["search_evaluations"] = evaluation_snapshots

    print(
        "lr_search_complete search=%d batch_size=%d update=%s %s "
        "initial_lr=%.6g best_k=%g best_lr=%.6g tta_val_acc=%.4f "
        "evaluated_lrs=%d"
        % (
            search_index,
            config["batch_size"],
            config["update_name"],
            hparam_label(config),
            config["muon_lr"],
            best_final,
            best_result["muon_lr"],
            best_result["tta_val_acc"],
            len(cache),
        ),
        flush=True,
    )
    return best_result


def main(
    run,
    model,
    batch_size,
    muon_lr,
    update_name,
    update_fn,
    normal_equation_dw,
    pseudo_inverse_dx,
    norm_inverse,
    remove_norm,
):
    set_training_seed()

    SGD_LR_MULT = batch_size / 2000
    bias_lr = 104 * SGD_LR_MULT
    head_lr = 1340 * SGD_LR_MULT

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
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
        filter_params, lr=muon_lr, momentum=0.6, nesterov=True, update_fn=update_fn
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

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

    model.reset()
    step = 0
    train_eval_batches = []
    failed = False
    val_acc = 0.0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalized_images()[:5000]
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            train_eval_batches.append((inputs.detach(), labels.detach()))
            train_eval_batches = train_eval_batches[-TRAIN_EVAL_BATCHES:]
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss = F.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="mean"
            )
            if not torch.isfinite(loss):
                print(
                    "nonfinite_loss run=%s epoch=%d step=%d loss=%s"
                    % (run, epoch, step + 1, loss.item()),
                    flush=True,
                )
                failed = True
                break
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            log_step(
                epoch=epoch,
                step=step,
                total_steps=total_train_steps,
                loss=loss.item(),
                head_lr=optimizer1.param_groups[2]["lr"],
                muon_lr=optimizer2.param_groups[0]["lr"],
            )
            if step >= total_train_steps:
                break
        stop_timer()

        if failed:
            break

        ####################
        #    Evaluation    #
        ####################

        val_acc = evaluate(model, test_loader, tta_level=0)
        log_eval(run, epoch, val_acc, time_seconds)
        run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    if failed:
        train25_loss = float("inf")
        tta_val_acc = 0.0
        log_final_eval(train25_loss, val_acc, tta_val_acc, time_seconds)
    else:
        start_timer()
        train25_loss = evaluate_train_loss(model, train_eval_batches)
        tta_val_acc = evaluate(model, test_loader, tta_level=2)
        stop_timer()
        log_final_eval(train25_loss, val_acc, tta_val_acc, time_seconds)

    return dict(
        train25_loss=train25_loss,
        **{"25batch_train_loss": train25_loss},
        val_acc=val_acc,
        tta_val_acc=tta_val_acc,
        batch_size=batch_size,
        muon_lr=muon_lr,
        sgd_lr_mult=SGD_LR_MULT,
        update_name=update_name,
        normal_equation_dw=normal_equation_dw,
        pseudo_inverse_dx=pseudo_inverse_dx,
        norm_inverse=norm_inverse,
        remove_norm=remove_norm,
    )


if __name__ == "__main__":
    results = []
    for run, config in enumerate(RUN_CONFIGS):
        set_training_seed()
        model = CifarNet(
            normal_equation_dw=config["normal_equation_dw"],
            pseudo_inverse_dx=config["pseudo_inverse_dx"],
            norm_inverse=config["norm_inverse"],
            remove_norm=config["remove_norm"],
        ).cuda().to(memory_format=torch.channels_last)
        # model.compile(mode="max-autotune")

        print(
            "cifar_baseline2_lr_search search=%d batch_size=%d "
            "initial_muon_lr=%.6g update=%s %s"
            % (
                run,
                config["batch_size"],
                config["muon_lr"],
                config["update_name"],
                hparam_label(config),
            ),
            flush=True,
        )
        result = search_initial_lr(run, model, config)
        results.append(result)
        print("Batch size:          %d" % result["batch_size"])
        print("Initial Muon lr:     %.6g" % result["initial_muon_lr"])
        print("Best Muon lr:        %.6g" % result["muon_lr"])
        print("Best search k:       %.6g" % result["best_search_offset"])
        print("SGD lr mult:         %.6g" % result["sgd_lr_mult"])
        print("Update:              %s" % result["update_name"])
        print("Normal equation dw:  %s" % result["normal_equation_dw"])
        print("Pseudo inverse dx:   %s" % result["pseudo_inverse_dx"])
        print("Norm inverse:        %s" % result["norm_inverse"])
        print("Remove norm:         %s" % result["remove_norm"])
        print("25batch train loss:  %.4f" % result["train25_loss"])
        print("Val acc:             %.4f" % result["val_acc"])
        print("TTA val:             %.4f" % result["tta_val_acc"])

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results), log_path)
    print(os.path.abspath(log_path))
