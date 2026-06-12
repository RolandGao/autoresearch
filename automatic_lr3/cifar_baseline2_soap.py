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
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 1
SOAP_DENOM_POWER = 0.50
SOAP_TARGET_UW = 0.3825
RADIAL_OUTWARD_SCALE = 0.5
RADIAL_INWARD_SCALE = 1.0
PR321_MU_MIN = 0.85
PR321_MU_MAX = 0.95
PR321_MU_WARMUP_FRACTION = 300 / 2900
PR321_MU_COOLDOWN_FRACTION = 200 / 2900


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


#############################################
#               Muon optimizer              #
#############################################


def normalize_rows(G):
    assert len(G.shape) == 2
    row_normalized = G / G.norm(dim=1, keepdim=True)
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


def gram_frobenius_norm_estimate(G, keepdim=False, eps=1e-10):
    X = G.float()
    gram = X.T @ X if X.size(0) > X.size(1) else X @ X.T
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)


def _pr321_ns_inner(X):
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def pr321_zeropower(G):
    assert len(G.shape) == 2
    X = G.to(MUON_DTYPE)
    if G.size(0) > G.size(1):
        X = X.T
    X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    X = _pr321_ns_inner(X)
    if G.size(0) > G.size(1):
        X = X.T
    return X


def soap_eigenbasis(mat):
    eye = torch.eye(mat.size(0), dtype=mat.dtype, device=mat.device)
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * eye)
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * eye.double())
        q = q.float()
    return torch.flip(q, [1])


def soap_basis_qr(row_gg, col_gg, q_row, q_col, exp_avg_sq):
    row_eig = torch.diag(q_row.T @ row_gg @ q_row)
    row_sort = torch.argsort(row_eig, descending=True)
    q_row = q_row[:, row_sort]
    exp_avg_sq = exp_avg_sq.index_select(0, row_sort)
    q_row, _ = torch.linalg.qr(row_gg @ q_row)

    col_eig = torch.diag(q_col.T @ col_gg @ q_col)
    col_sort = torch.argsort(col_eig, descending=True)
    q_col = q_col[:, col_sort]
    exp_avg_sq = exp_avg_sq.index_select(1, col_sort)
    q_col, _ = torch.linalg.qr(col_gg @ q_col)
    return q_row, q_col, exp_avg_sq


def soap_precondition_momentum(update, state, beta2=SOAP_BETA2, eps=1e-8):
    update_f = update.float()
    if state["q_row"] is None:
        return update
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update_f @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(SOAP_DENOM_POWER)
    precond = q_row @ (projected / denom) @ q_col.T
    precond.mul_(
        gram_frobenius_norm_estimate(update_f, eps=eps)
        / gram_frobenius_norm_estimate(precond, eps=eps)
    )
    return precond.to(update.dtype)


def soap_update_preconditioner(
    grad, state, beta2=SOAP_BETA2, frequency=SOAP_PRECONDITION_FREQUENCY
):
    grad_f = grad.float()
    state["row_gg"].lerp_(grad_f @ grad_f.T, 1 - beta2)
    state["col_gg"].lerp_(grad_f.T @ grad_f, 1 - beta2)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"],
            state["col_gg"],
            state["q_row"],
            state["q_col"],
            state["exp_avg_sq"],
        )
    state["soap_step"] += 1


def pr321_muon_update(update):
    update = pr321_zeropower(update)
    update *= max(1, update.size(0) / update.size(1)) ** 0.5
    return update


def scale_radial_update(update, param, eps=1e-12):
    update_f = update.float()
    param_f = param.float()
    denom = param_f.square().sum().clamp_min(eps)
    coeff = (update_f * param_f).sum() / denom
    radial = coeff * param_f
    tangential = update_f - radial
    radial_scale = torch.where(
        coeff < 0,
        update_f.new_tensor(RADIAL_OUTWARD_SCALE),
        update_f.new_tensor(RADIAL_INWARD_SCALE),
    )
    return (tangential + radial_scale * radial).to(update.dtype)


def target_radius_after_update(param, update, lr, eps=1e-8):
    param_f = param.float()
    update_f = update.float()
    before_norm = param_f.norm().clamp_min(eps)
    radial_delta = -lr * (update_f * param_f).sum() / before_norm
    return (before_norm + radial_delta).clamp_min(eps)


def rescale_to_radius(param, target_norm, eps=1e-8):
    after_norm = param.float().norm().clamp_min(eps)
    param.mul_((target_norm / after_norm).to(param.dtype))


def pr321_mu_at_step(step, total_steps):
    warmup_steps = max(1, ceil(total_steps * PR321_MU_WARMUP_FRACTION))
    cooldown_steps = max(1, ceil(total_steps * PR321_MU_COOLDOWN_FRACTION))
    cooldown_start = total_steps - cooldown_steps
    if step < warmup_steps:
        frac = step / warmup_steps
        return PR321_MU_MIN + frac * (PR321_MU_MAX - PR321_MU_MIN)
    if step > cooldown_start:
        frac = (step - cooldown_start) / cooldown_steps
        return PR321_MU_MAX - frac * (PR321_MU_MAX - PR321_MU_MIN)
    return PR321_MU_MAX


class PR321SOAPMuon(torch.optim.Optimizer):
    """PR #321 SOAP-Muon adapted to CIFAR conv filters.

    The PR applies SOAP to all hidden 2-D matrices. CIFAR filters are flattened as
    out_channels x (in_channels * kernel_h * kernel_w), so every trainable conv
    filter passed to this optimizer is treated as a hidden matrix.
    """

    def __init__(self, named_params, lr=1e-3, mu=0.95):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum value: {mu}")
        named_params = list(named_params)
        if not named_params:
            raise ValueError("PR321SOAPMuon requires at least one parameter")
        self.param_names = {p: n for n, p in named_params}
        params = [p for _, p in named_params]
        super().__init__(params, dict(lr=lr, mu=mu))

    @staticmethod
    def _matrix_view(tensor):
        return tensor.reshape(tensor.size(0), -1)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                grad_matrix = self._matrix_view(grad)
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad_matrix, dtype=torch.float32
                    )
                    state["row_gg"] = torch.zeros(
                        grad_matrix.size(0),
                        grad_matrix.size(0),
                        dtype=torch.float32,
                        device=p.device,
                    )
                    state["col_gg"] = torch.zeros(
                        grad_matrix.size(1),
                        grad_matrix.size(1),
                        dtype=torch.float32,
                        device=p.device,
                    )
                    state["q_row"] = None
                    state["q_col"] = None
                    state["soap_step"] = 0

                state["momentum"].lerp_(grad, 1 - mu)
                momentum_update = grad.lerp(state["momentum"], mu)
                update_matrix = self._matrix_view(momentum_update)
                update_matrix = soap_precondition_momentum(update_matrix, state)
                update_matrix = pr321_muon_update(update_matrix)
                update = update_matrix.view_as(p)
                update = scale_radial_update(update, p)

                p_norm = p.float().norm().clamp_min(1e-8)
                update_norm = update.float().norm().clamp_min(1e-8)
                cur_uw = update_norm / p_norm
                scale = torch.where(
                    cur_uw < SOAP_TARGET_UW,
                    SOAP_TARGET_UW * p_norm / update_norm,
                    torch.ones_like(p_norm),
                )
                update = update * scale.to(update.dtype)
                target_radius = target_radius_after_update(p, update, lr)
                p.add_(update, alpha=-lr)
                rescale_to_radius(p, target_radius)
                soap_update_preconditioner(grad_matrix, state)


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
    dict(batch_size=125),
    dict(batch_size=500),
    dict(batch_size=2000),
]
UPDATE_CONFIGS = [
    dict(update_name="pr321_soap_muon", update_fn=None),
    dict(update_name="row_norm", update_fn=normalize_rows),
    dict(update_name="row_norm_max", update_fn=normalize_rows_max),
    dict(update_name="matrix_norm", update_fn=normalize_matrix),
    dict(update_name="matrix_norm_max", update_fn=normalize_matrix_max),
    dict(
        update_name="zeropower_via_newtonschulz5",
        update_fn=zeropower_via_newtonschulz5,
    ),
    dict(
        update_name="zeropower_via_newtonschulz5_max",
        update_fn=zeropower_via_newtonschulz5_max,
    ),
]
MANUAL_MUON_LRS = {
    "pr321_soap_muon": {
        125: 0.062,
        500: 0.099,
        2000: 0.22,
    },
    "row_norm": {
        125: 0.04,
        500: 0.084,
        2000: 0.061,
    },
    "row_norm_max": {
        125: 0.04,
        500: 0.084,
        2000: 0.061,
    },
    "matrix_norm": {
        125: 0.04,
        500: 0.084,
        2000: 0.061,
    },
    "matrix_norm_max": {
        125: 0.04,
        500: 0.084,
        2000: 0.061,
    },
    "zeropower_via_newtonschulz5": {
        125: 0.062,
        500: 0.099,
        2000: 0.22,
    },
    "zeropower_via_newtonschulz5_max": {
        125: 0.062,
        500: 0.099,
        2000: 0.22,
    },
}
RUN_CONFIGS = [
    dict(
        **batch_config,
        **update_config,
        muon_lr=MANUAL_MUON_LRS[update_config["update_name"]][
            batch_config["batch_size"]
        ],
    )
    for batch_config in BATCH_CONFIGS
    for update_config in UPDATE_CONFIGS
]


def rounded_lr(value):
    return float(f"{value:.{LR_SEARCH_SIG_FIGS}g}")


def lr_key(value):
    return f"{rounded_lr(value):.{LR_SEARCH_SIG_FIGS}g}"


def lr_at_offset(initial_lr, offset):
    return rounded_lr(initial_lr * LR_SEARCH_FACTOR**offset)


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
            run_name = (
                "search%d_bs%d_update_%s_k%g_lr%s"
                % (
                    search_index,
                    config["batch_size"],
                    config["update_name"],
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
            )
            cache[key]["search_offset"] = offset
            cache[key]["rounded_muon_lr"] = lr
            evaluations.append(cache[key])
        else:
            print(
                "lr_search_cache_hit search=%d batch_size=%d update=%s "
                "k=%g rounded_lr=%s"
                % (
                    search_index,
                    config["batch_size"],
                    config["update_name"],
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
        "lr_search_complete search=%d batch_size=%d update=%s "
        "initial_lr=%.6g best_k=%g best_lr=%.6g tta_val_acc=%.4f "
        "evaluated_lrs=%d"
        % (
            search_index,
            config["batch_size"],
            config["update_name"],
            config["muon_lr"],
            best_final,
            best_result["muon_lr"],
            best_result["tta_val_acc"],
            len(cache),
        ),
        flush=True,
    )
    return best_result


def main(run, model, batch_size, muon_lr, update_name, update_fn):
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
    filter_named_params = [
        (n, p)
        for n, p in model.named_parameters()
        if len(p.shape) == 4 and p.requires_grad
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
    if update_name == "pr321_soap_muon":
        optimizer2 = PR321SOAPMuon(filter_named_params, lr=muon_lr, mu=0.95)
    else:
        optimizer2 = Muon(
            filter_params,
            lr=muon_lr,
            momentum=0.6,
            nesterov=True,
            update_fn=update_fn,
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
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            if update_name == "pr321_soap_muon":
                for group in optimizer2.param_groups:
                    group["mu"] = pr321_mu_at_step(step, total_train_steps)
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

        ####################
        #    Evaluation    #
        ####################

        val_acc = evaluate(model, test_loader, tta_level=0)
        log_eval(run, epoch, val_acc, time_seconds)
        run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

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
    )


if __name__ == "__main__":
    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    # main("warmup", model, RUN_CONFIGS[0]["batch_size"], RUN_CONFIGS[0]["muon_lr"])
    results = []
    for run, config in enumerate(RUN_CONFIGS):
        print(
            "cifar_baseline2_lr_search search=%d batch_size=%d "
            "initial_muon_lr=%.6g update=%s"
            % (
                run,
                config["batch_size"],
                config["muon_lr"],
                config["update_name"],
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
        print("25batch train loss:  %.4f" % result["train25_loss"])
        print("Val acc:             %.4f" % result["val_acc"])
        print("TTA val:             %.4f" % result["tta_val_acc"])

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results), log_path)
    print(os.path.abspath(log_path))
