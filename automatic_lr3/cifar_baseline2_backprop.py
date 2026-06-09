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

    def _step_one(self, group, p, g):
        lr = group["lr"]
        momentum = group["momentum"]
        with torch.no_grad():
            g = g.detach()
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

    def step_param(self, p, g):
        for group in self.param_groups:
            for group_param in group["params"]:
                if group_param is p:
                    self._step_one(group, p, g)
                    return
        raise ValueError("Parameter is not owned by this Muon optimizer")

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                self._step_one(group, p, g)


def sgd_step_param(optimizer, p, g):
    for group in optimizer.param_groups:
        for group_param in group["params"]:
            if group_param is p:
                sgd_step_one(group, optimizer.state[p], p, g)
                return
    raise ValueError("Parameter is not owned by this SGD optimizer")


def sgd_step_one(group, state, p, g):
    with torch.no_grad():
        g = g.detach()
        if group.get("maximize", False):
            g = -g

        weight_decay = group["weight_decay"]
        if weight_decay != 0:
            g = g.add(p, alpha=weight_decay)

        momentum = group["momentum"]
        if momentum != 0:
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.clone(g).detach()
                state["momentum_buffer"] = buf
            else:
                buf.mul_(momentum).add_(g, alpha=1 - group["dampening"])
            g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

        p.data.add_(g, alpha=-group["lr"])


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


def _conv_grad_padding(padding, kernel_size, dilation):
    if padding == "same":
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)
        effective_kernel = tuple(
            d * (k - 1) + 1 for k, d in zip(kernel_size, dilation)
        )
        if any(k % 2 == 0 for k in effective_kernel):
            raise ValueError(
                '"same" padding grad helper only supports odd effective kernels'
            )
        return tuple(k // 2 for k in effective_kernel)
    if padding == "valid":
        return (0, 0)
    return padding


def _weight_for_input_grad(module, old_weight):
    blend = module.input_grad_weight_blend
    if blend == 1.0:
        return module.weight
    if blend == 0.0:
        return old_weight
    return old_weight.add(module.weight.detach() - old_weight, alpha=blend)


class MuonConv2dUpdateBeforeInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, module):
        ctx.module = module
        ctx.input_size = x.shape
        ctx.save_for_backward(x)
        return F.conv2d(
            x,
            weight,
            None,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        module = ctx.module
        padding = _conv_grad_padding(
            module.padding, module.kernel_size, module.dilation
        )
        old_weight = module.weight.detach().clone()

        grad_weight = torch.nn.grad.conv2d_weight(
            x,
            module.weight.shape,
            grad_output,
            stride=module.stride,
            padding=padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        module.muon_optimizer.step_param(module.weight, grad_weight)

        grad_input = None
        if ctx.needs_input_grad[0]:
            weight_for_input_grad = _weight_for_input_grad(module, old_weight)
            grad_input = torch.nn.grad.conv2d_input(
                ctx.input_size,
                weight_for_input_grad,
                grad_output,
                stride=module.stride,
                padding=padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        return grad_input, None, None


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )
        self.muon_optimizer = None
        self.input_grad_weight_blend = 1.0

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])

    def forward(self, x):
        if self.training and torch.is_grad_enabled() and self.weight.requires_grad:
            if self.muon_optimizer is None:
                raise RuntimeError(
                    "Train-time Conv backward requires an attached Muon optimizer"
                )
            return MuonConv2dUpdateBeforeInputGrad.apply(x, self.weight, self)
        return super().forward(x)


class SgdLinearUpdateBeforeInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, module):
        ctx.module = module
        ctx.save_for_backward(x)
        return F.linear(x, weight, None)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        module = ctx.module
        old_weight = module.weight.detach().clone()

        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).T @ x.reshape(
            -1, x.shape[-1]
        )
        sgd_step_param(module.sgd_optimizer, module.weight, grad_weight)

        grad_input = None
        if ctx.needs_input_grad[0]:
            weight_for_input_grad = _weight_for_input_grad(module, old_weight)
            grad_input = grad_output @ weight_for_input_grad
        return grad_input, None, None


class SgdLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.sgd_optimizer = None
        self.input_grad_weight_blend = 1.0

    def forward(self, x):
        if self.training and torch.is_grad_enabled() and self.weight.requires_grad:
            if self.sgd_optimizer is None:
                raise RuntimeError(
                    "Train-time Linear backward requires an attached SGD optimizer"
                )
            return SgdLinearUpdateBeforeInputGrad.apply(x, self.weight, self)
        return super().forward(x)


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
        self.head = SgdLinear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear, SgdLinear):
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
UPDATE_NAME = "update_before_backprop"
CONV_INPUT_GRAD_WEIGHT_BLEND = 0.1
HEAD_INPUT_GRAD_WEIGHT_BLEND = 0.0
RUN_CONFIGS = [
    dict(
        batch_size=125,
        muon_lr=0.04,
        update_name=f"{UPDATE_NAME}_conv0p1_head0_bs125",
        conv_input_grad_weight_blend=CONV_INPUT_GRAD_WEIGHT_BLEND,
        head_input_grad_weight_blend=HEAD_INPUT_GRAD_WEIGHT_BLEND,
    ),
    dict(
        batch_size=500,
        muon_lr=0.079,
        update_name=f"{UPDATE_NAME}_conv0p1_head0_bs500",
        conv_input_grad_weight_blend=CONV_INPUT_GRAD_WEIGHT_BLEND,
        head_input_grad_weight_blend=HEAD_INPUT_GRAD_WEIGHT_BLEND,
    ),
    dict(
        batch_size=2000,
        muon_lr=0.24,
        update_name=f"{UPDATE_NAME}_conv0p1_head0_bs2000",
        conv_input_grad_weight_blend=CONV_INPUT_GRAD_WEIGHT_BLEND,
        head_input_grad_weight_blend=HEAD_INPUT_GRAD_WEIGHT_BLEND,
    ),
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
                config["conv_input_grad_weight_blend"],
                config["head_input_grad_weight_blend"],
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
        "evaluated_lrs=%d conv_blend=%.3g head_blend=%.3g"
        % (
            search_index,
            config["batch_size"],
            config["update_name"],
            config["muon_lr"],
            best_final,
            best_result["muon_lr"],
            best_result["tta_val_acc"],
            len(cache),
            config["conv_input_grad_weight_blend"],
            config["head_input_grad_weight_blend"],
        ),
        flush=True,
    )
    return best_result


def main(
    run,
    model,
    batch_size,
    muon_lr,
    update_name=UPDATE_NAME,
    conv_input_grad_weight_blend=CONV_INPUT_GRAD_WEIGHT_BLEND,
    head_input_grad_weight_blend=HEAD_INPUT_GRAD_WEIGHT_BLEND,
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
    optimizer2 = Muon(filter_params, lr=muon_lr, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    for module in model.modules():
        if isinstance(module, Conv):
            module.muon_optimizer = optimizer2
            module.input_grad_weight_blend = conv_input_grad_weight_blend
        if isinstance(module, SgdLinear):
            module.sgd_optimizer = optimizer1
            module.input_grad_weight_blend = head_input_grad_weight_blend

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
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (
                    1 - step / whiten_bias_train_steps
                )
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            loss.backward()
            optimizer1.step()
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
        conv_input_grad_weight_blend=conv_input_grad_weight_blend,
        head_input_grad_weight_blend=head_input_grad_weight_blend,
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
        print("Conv dx blend:       %.6g" % result["conv_input_grad_weight_blend"])
        print("Head dx blend:       %.6g" % result["head_input_grad_weight_blend"])
        print("25batch train loss:  %.4f" % result["train25_loss"])
        print("Val acc:             %.4f" % result["val_acc"])
        print("TTA val:             %.4f" % result["tta_val_acc"])

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results), log_path)
    print(os.path.abspath(log_path))
