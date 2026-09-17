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
from math import ceil, log10

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
#             Angular optimizer             #
#############################################


def angular_fallback_tangent(p):
    fallback = torch.roll(p, shifts=1, dims=-1)
    p_norm_sq = (p * p).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-24)
    tangent = fallback - (fallback * p).sum(dim=(-2, -1), keepdim=True) / p_norm_sq * p
    second = torch.roll(p, shifts=1, dims=-2)
    second = second - (second * p).sum(dim=(-2, -1), keepdim=True) / p_norm_sq * p
    tangent_norm = tangent.norm(dim=(-2, -1), keepdim=True)
    return torch.where(tangent_norm > 1e-12, tangent, second)


def angular_move_on_frobenius_sphere(p, direction, lr):
    p = p.float()
    direction = direction.float()
    p_norm = p.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    inner = (direction * p).sum(dim=(-2, -1), keepdim=True)
    tangent = direction - inner / p_norm.square() * p
    tangent_norm = tangent.norm(dim=(-2, -1), keepdim=True)
    fallback = angular_fallback_tangent(p)
    tangent = torch.where(tangent_norm > 1e-12, tangent, fallback)
    tangent_unit = tangent / tangent.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    lr = torch.as_tensor(lr, device=p.device, dtype=p.dtype)
    rotated = p * torch.cos(lr) - tangent_unit * (p_norm * torch.sin(lr))
    return rotated / rotated.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12) * p_norm


class AngularSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.copy_(
                    angular_move_on_frobenius_sphere(p, p.grad, lr).to(dtype=p.dtype)
                )


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
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.bfloat16() / 255)
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
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.bfloat16()

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


def compute_weight_angles(previous_weights, current_params):
    angle_tensors = []
    for (name, previous), (current_name, current) in zip(
        previous_weights, current_params
    ):
        assert name == current_name
        previous_flat = previous.reshape(-1)
        current_flat = current.detach().float().reshape(-1)
        denom = previous_flat.norm() * current_flat.norm()
        cos_angle = (previous_flat @ current_flat / denom.clamp_min(1e-30)).clamp(
            -1.0, 1.0
        )
        angle_tensors.append(torch.acos(cos_angle))
    return [
        (name, angle)
        for (name, _), angle in zip(
            previous_weights, torch.stack(angle_tensors).cpu().tolist()
        )
    ]


def log_step(epoch, step, total_steps, loss, head_lr, muon_lr, weight_angles):
    angle_info = " ".join(
        f"{name}_angle_rad={angle:.6g}" for name, angle in weight_angles
    )
    print(
        f"step={step}/{total_steps} epoch={epoch} "
        f"loss={loss:.4f} head_lr={head_lr:.6g} muon_lr={muon_lr:.6g} "
        f"{angle_info}",
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
HEAD_WARMUP_STEPS = 5
HEAD_WARMUP_LR = 1000.0


def decayed_angular_lr(step, total_steps, initial_lr):
    angular_step = step - HEAD_WARMUP_STEPS
    angular_steps = max(1, total_steps - HEAD_WARMUP_STEPS - 1)
    return initial_lr * max(0.0, 1.0 - angular_step / angular_steps)


def main(run, model, angular_lr):
    set_training_seed()

    batch_size = 2000
    SGD_LR_MULT = 1.0
    MUON_LR_MULT = 1.0
    bias_lr = 104 * SGD_LR_MULT
    angular_lr = float(angular_lr)

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
    muon_params = [
        (n, p)
        for n, p in model.named_parameters()
        if len(p.shape) == 4 and p.requires_grad
    ]
    filter_params = [p for _, p in muon_params]
    angle_params = [("head", model.head.weight)] + [
        (f"muon.{name}", param) for name, param in muon_params
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=0),
        dict(params=norm_biases, lr=bias_lr, weight_decay=0),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    head_warmup_optimizer = torch.optim.SGD(
        [dict(params=[model.head.weight], lr=HEAD_WARMUP_LR, weight_decay=0)],
        momentum=0.85,
        nesterov=True,
        fused=True,
    )
    head_angular_optimizer = AngularSGD([model.head.weight], lr=angular_lr)
    optimizer2 = Muon(
        filter_params, lr=0.24 * MUON_LR_MULT, momentum=0.6, nesterov=True
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
            previous_weights = [
                (name, param.detach().float().clone()) for name, param in angle_params
            ]
            use_head_warmup = step < HEAD_WARMUP_STEPS
            optimizer1.step()
            if use_head_warmup:
                head_lr = HEAD_WARMUP_LR
                head_warmup_optimizer.step()
            else:
                head_lr = decayed_angular_lr(step, total_train_steps, angular_lr)
                head_angular_optimizer.param_groups[0]["lr"] = head_lr
                head_angular_optimizer.step()
            optimizer2.step()
            weight_angles = compute_weight_angles(previous_weights, angle_params)
            model.zero_grad(set_to_none=True)
            step += 1
            log_step(
                epoch=epoch,
                step=step,
                total_steps=total_train_steps,
                loss=loss.item(),
                head_lr=head_lr,
                muon_lr=optimizer2.param_groups[0]["lr"],
                weight_angles=weight_angles,
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
    )


if __name__ == "__main__":
    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    angular_lrs = torch.logspace(log10(0.01), log10(0.3), steps=40).tolist()
    # main("warmup", model, angular_lrs[0])
    results = []
    for run, angular_lr in enumerate(angular_lrs):
        print(f"angular_lr_search run={run} angular_lr={angular_lr:.6g}", flush=True)
        result = main(run, model, angular_lr)
        result["angular_lr"] = angular_lr
        results.append(result)
        print("angular lr:         %.6g" % result["angular_lr"])
        print("25batch train loss: %.4f" % result["train25_loss"])
        print("Val acc:            %.4f" % result["val_acc"])
        print("TTA val:            %.4f" % result["tta_val_acc"])

    best_result = max(results, key=lambda result: result["tta_val_acc"])
    print("Best angular lr:    %.6g" % best_result["angular_lr"])
    print("Best val acc:       %.4f" % best_result["val_acc"])
    print("Best TTA val:       %.4f" % best_result["tta_val_acc"])

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results, best_result=best_result), log_path)
    print(os.path.abspath(log_path))
