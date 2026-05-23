# A variant of airbench optimized for time-to-96%.
# 27.3s runtime on an A100; 3.1 PFLOPs.
# Evidence: 96.00 average accuracy in n=200 runs.
#
# We recorded the above runtime on an NVIDIA A100-SXM4-40GB with the following nvidia-smi:
# NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7
# torch.__version__ == '2.4.0+cu121'

#############################################
#            Setup/Hyperparameters          #
#############################################

import json
import os
import random
import sys
import uuid
from collections import Counter
from math import ceil, isfinite

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

with open(sys.argv[0]) as f:
    code = f.read()

TRAINING_SEED = 42
DATA_SEED = 1_000_003
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(SCRIPT_DIR, "cifar10")
USE_COMPILE = os.environ.get("CIFAR96_COMPILE", "0") == "1"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def setup_reproducibility(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_model(model):
    return getattr(model, "_orig_mod", model)


def maybe_compile(model):
    if USE_COMPILE:
        return torch.compile(model, mode="reduce-overhead")
    return model


hyp = {
    "opt": {
        "train_epochs": 45.0,
        "batch_size": 1024,
        "batch_size_masked": 512,
        "lr": 9.0,  # learning rate per 1024 examples
        "momentum": 0.85,
        "weight_decay": 0.012,  # weight decay per 1024 examples (decoupled from learning rate)
        "bias_scaler": 64.0,  # scales up learning rate (but not weight decay) for BatchNorm biases
        "label_smoothing": 0.2,
        "whiten_bias_epochs": 3,  # how many epochs to train the whitening layer bias before freezing
    },
    "aug": {
        "flip": True,
        "translate": 4,
        "cutout": 12,
    },
    "proxy": {
        "widths": {
            "block1": 32,
            "block2": 64,
            "block3": 64,
        },
        "depth": 2,
        "scaling_factor": 1 / 9,
    },
    "net": {
        "widths": {
            "block1": 128,
            "block2": 384,
            "block3": 512,
        },
        "depth": 3,
        "scaling_factor": 1 / 9,
        "tta_level": 2,  # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

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
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype,
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n, c, h, w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h - size + 1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w - size + 1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(
        1, 1, h, 1
    ) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(
        1, 1, 1, w
    ) - corner_x.view(-1, 1, 1, 1)

    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask


def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)


def set_random_state(seed, state):
    if seed is None:
        # If we don't get a data seed, then make sure to randomize the state using independent generator, since
        # it might have already been set by the model seed.
        import random

        torch.manual_seed(random.randint(0, 2**63))
    else:
        seed1 = (
            1000 * seed + state
        )  # just don't do more than 1000 epochs or else there will be overlap
        torch.manual_seed(seed1)


class InfiniteCifarLoader:
    """
    CIFAR-10 loader which constructs every input to be used during training during the call to __iter__.
    The purpose is to support cross-epoch batches (in case the batch size does not divide the number of train examples),
    and support stochastic iteration counts in order to preserve perfect linearity/independence.
    """

    def __init__(
        self,
        path,
        train=True,
        batch_size=500,
        aug=None,
        altflip=True,
        subset_mask=None,
        aug_seed=None,
        order_seed=None,
    ):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )

        data = torch.load(data_path, map_location="cuda")
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate", "cutout"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.altflip = altflip
        self.subset_mask = (
            subset_mask
            if subset_mask is not None
            else torch.tensor([True] * len(self.images)).cuda()
        )
        self.train = train
        self.aug_seed = aug_seed
        self.order_seed = order_seed

    def __iter__(self):

        # Preprocess
        images0 = self.normalize(self.images)
        # Pre-randomly flip images in order to do alternating flip later.
        if self.aug.get("flip", False) and self.altflip:
            set_random_state(self.aug_seed, 0)
            images0 = batch_flip_lr(images0)
        # Pre-pad images to save time when doing random translation
        pad = self.aug.get("translate", 0)
        if pad > 0:
            images0 = F.pad(images0, (pad,) * 4, "reflect")
        labels0 = self.labels

        # Iterate forever
        epoch = 0
        batch_size = self.batch_size

        # In the below while-loop, we will repeatedly build a batch and then yield it.
        num_examples = self.subset_mask.sum().item()
        current_pointer = num_examples
        batch_images = torch.empty(
            0, 3, 32, 32, dtype=images0.dtype, device=images0.device
        )
        batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
        batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

        while True:
            # Assume we need to generate more data to add to the batch.
            assert len(batch_images) < batch_size

            # If we have already exhausted the current epoch, then begin a new one.
            if current_pointer >= num_examples:
                # If we already reached the end of the last epoch then we need to generate
                # a new augmented epoch of data (using random crop and alternating flip).
                epoch += 1

                set_random_state(self.aug_seed, epoch)
                if pad > 0:
                    images1 = batch_crop(images0, 32)
                if self.aug.get("flip", False):
                    if self.altflip:
                        images1 = images1 if epoch % 2 == 0 else images1.flip(-1)
                    else:
                        images1 = batch_flip_lr(images1)
                if self.aug.get("cutout", 0) > 0:
                    images1 = batch_cutout(images1, self.aug["cutout"])

                set_random_state(self.order_seed, epoch)
                indices = (torch.randperm if self.train else torch.arange)(
                    len(self.images), device=images0.device
                )

                # The effect of doing subsetting in this manner is as follows. If the permutation wants to show us
                # our four examples in order [3, 2, 0, 1], and the subset mask is [True, False, True, False],
                # then we will be shown the examples [2, 0]. It is the subset of the ordering.
                # The purpose is to minimize the interaction between the subset mask and the randomness.
                # So that the subset causes not only a subset of the total examples to be shown, but also a subset of
                # the actual sequence of examples which is shown during training.
                indices_subset = indices[self.subset_mask[indices]]
                current_pointer = 0

            # Now we are sure to have more data in this epoch remaining.
            # This epoch's remaining data is given by (images1[current_pointer:], labels0[current_pointer:])
            # We add more data to the batch, up to whatever is needed to make a full batch (but it might not be enough).
            remaining_size = batch_size - len(batch_images)

            # Given that we want `remaining_size` more training examples, we construct them here, using
            # the remaining available examples in the epoch.

            extra_indices = indices_subset[
                current_pointer : current_pointer + remaining_size
            ]
            extra_images = images1[extra_indices]
            extra_labels = labels0[extra_indices]
            current_pointer += remaining_size
            batch_indices = torch.cat([batch_indices, extra_indices])
            batch_images = torch.cat([batch_images, extra_images])
            batch_labels = torch.cat([batch_labels, extra_labels])

            # If we have a full batch ready then yield it and reset.
            if len(batch_images) == batch_size:
                assert len(batch_images) == len(batch_labels)
                yield (batch_indices, batch_images, batch_labels)
                batch_images = torch.empty(
                    0, 3, 32, 32, dtype=images0.dtype, device=images0.device
                )
                batch_labels = torch.empty(
                    0, dtype=labels0.dtype, device=labels0.device
                )
                batch_indices = torch.empty(
                    0, dtype=labels0.dtype, device=labels0.device
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
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat(
            [infer_fn(inputs, model) for inputs in test_images.split(2000)]
        )


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


#############################################
#            Network Components             #
#############################################


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding="same", bias=False
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth):
        super().__init__()
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        if depth == 3:
            self.conv3 = Conv(channels_out, channels_out)
            self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.depth == 3:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        if self.depth == 3:
            x = self.conv3(x)
            x = self.norm3(x)
            x = x + x0
            x = self.activ(x)
        return x


#############################################
#            Network Definition             #
#############################################


def make_net(hyp):
    widths = hyp["widths"]
    scaling_factor = hyp["scaling_factor"]
    depth = hyp["depth"]
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width, widths["block1"], depth),
        ConvGroup(widths["block1"], widths["block2"], depth),
        ConvGroup(widths["block2"], widths["block3"], depth),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths["block3"], 10, bias=False),
        Mul(scaling_factor),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net


def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()


#############################################
#       Whitening Conv Initialization       #
#############################################


def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return (
        x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    )


def get_whitening_parameters(patches):
    n, c, h, w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(
        c * h * w, c, h, w
    ).flip(0)


def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))


############################################
#                Lookahead                 #
############################################


class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(
            self.net_ema.values(), net.state_dict().values()
        ):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1 - decay)
                net_param.copy_(ema_param)


logging_columns_list = [
    "run",
    "epoch",
    "train_loss",
    "train_acc",
    "val_acc",
    "tta_val_acc",
    "time_seconds",
]


def print_training_details(variables, is_final_entry):
    fields = []
    for col in logging_columns_list:
        var = variables.get(col, None)
        if type(var) in (int, str):
            fields.append(f"{col}={var}")
        elif type(var) is float:
            fields.append(f"{col}={var:0.4f}")
        else:
            assert var is None
    print("epoch_summary " + " ".join(fields), flush=True)


############################################
#              Line Search                 #
############################################

LINE_SEARCH_FACTOR = 0.8
LINE_SEARCH_LEFT_STEPS = 30
LINE_SEARCH_RIGHT_STEPS = 20


def clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state(v) for v in value)
    return value


def clone_grads(model):
    return [
        None
        if p.grad is None
        else p.grad.detach().clone(memory_format=torch.preserve_format)
        for p in model.parameters()
    ]


def restore_grads(model, grads):
    for p, grad in zip(model.parameters(), grads):
        p.grad = (
            None if grad is None else grad.clone(memory_format=torch.preserve_format)
        )


def capture_training_state(model, optimizers):
    return (
        clone_state(model.state_dict()),
        [clone_state(opt.state_dict()) for opt in optimizers],
        clone_grads(model),
        model.training,
    )


def restore_training_state(model, optimizers, state):
    model_state, optimizer_states, grads, was_training = state
    model.load_state_dict(model_state)
    for opt, opt_state in zip(optimizers, optimizer_states):
        opt.load_state_dict(clone_state(opt_state))
    restore_grads(model, grads)
    model.train(was_training)


def lr_from_exponent(base_lr, exponent):
    return base_lr * (LINE_SEARCH_FACTOR**exponent)


def configure_matrix_lr_multipliers(optimizer, initial_matrix_lr):
    for group in optimizer.param_groups:
        group_initial_lr = group.get("initial_lr", group["lr"])
        group["matrix_lr_multiplier"] = group_initial_lr / initial_matrix_lr


def matrix_lr_from_optimizer(optimizer):
    group = optimizer.param_groups[-1]
    return group["lr"] / group.get("matrix_lr_multiplier", 1.0)


def set_matrix_lr(optimizer, matrix_lr):
    for group in optimizer.param_groups:
        group["lr"] = matrix_lr * group.get("matrix_lr_multiplier", 1.0)


def compute_line_search_losses(model, batches):
    losses = []
    model.train()
    for inputs, labels in batches:
        outputs = model(inputs)
        batch_losses = F.cross_entropy(
            outputs.float(),
            labels,
            label_smoothing=hyp["opt"]["label_smoothing"],
            reduction="none",
        )
        losses.append(batch_losses)
    return torch.cat(losses)


def finite_loss_item(loss):
    return loss.item() if torch.isfinite(loss) else float("inf")


def finite_loss_sum_item(losses):
    return finite_loss_item(losses.sum())


def finite_float_or_none(value):
    value = float(value)
    return value if isfinite(value) else None


def sample_lr_loss_curve_payload(sample_losses_by_matrix_lr):
    matrix_lrs = sorted(sample_losses_by_matrix_lr)
    if not matrix_lrs:
        return dict(lrs=[], optimal_lr_counts={}, curves=[])
    losses = torch.stack(
        [
            torch.where(
                torch.isfinite(sample_losses_by_matrix_lr[matrix_lr]),
                sample_losses_by_matrix_lr[matrix_lr],
                torch.full_like(sample_losses_by_matrix_lr[matrix_lr], float("inf")),
            )
            for matrix_lr in matrix_lrs
        ]
    )
    best_indices = losses.argmin(dim=0).detach().cpu().tolist()
    best_lrs = [matrix_lrs[index] for index in best_indices]
    optimal_lr_counts = Counter(best_lrs)

    selected_indices = []
    seen_best_indices = set()
    for sample_index, best_index in enumerate(best_indices):
        if best_index in seen_best_indices:
            continue
        selected_indices.append(sample_index)
        seen_best_indices.add(best_index)

    losses = losses.detach().cpu()
    return dict(
        lrs=format_matrix_lrs(matrix_lrs),
        optimal_lr_counts={
            "%.8g" % matrix_lr: count
            for matrix_lr, count in sorted(optimal_lr_counts.items())
        },
        curves=[
            dict(
                sample_index=sample_index,
                optimal_lr=float("%.8g" % best_lrs[sample_index]),
                losses=[
                    finite_float_or_none(losses[lr_index, sample_index].item())
                    for lr_index in range(len(matrix_lrs))
                ],
            )
            for sample_index in selected_indices
        ],
    )


def evaluate_matrix_lr_grid(
    model,
    optimizers,
    matrix_optimizer,
    validation_sets,
    matrix_lrs,
):
    search_state = capture_training_state(model, optimizers)
    losses_by_name = {name: {} for name, _ in validation_sets}
    sample_losses_by_name = {name: {} for name, _ in validation_sets}

    def restore_search_state():
        restore_training_state(model, optimizers, search_state)

    for matrix_lr in matrix_lrs:
        if all(matrix_lr in losses for losses in losses_by_name.values()):
            continue
        for name, batches in validation_sets:
            if matrix_lr in losses_by_name[name]:
                continue
            restore_search_state()
            set_matrix_lr(matrix_optimizer, matrix_lr)
            for opt in optimizers:
                opt.step()
            with torch.no_grad():
                losses = compute_line_search_losses(model, batches)
            losses_by_name[name][matrix_lr] = finite_loss_sum_item(losses)
            if name == "current_batch":
                sample_losses_by_name[name][matrix_lr] = losses.detach()

    restore_search_state()
    results = {}
    for name, losses in losses_by_name.items():
        results[name] = dict(
            best_matrix_lr=min(losses.items(), key=lambda item: item[1])[0],
            best_loss=min(losses.values()),
            losses_by_matrix_lr=losses,
        )
        sample_losses = sample_losses_by_name.get(name, {})
        if sample_losses:
            results[name]["sample_lr_loss_curves"] = sample_lr_loss_curve_payload(
                sample_losses
            )
    return results


def initial_ground_truth_matrix_lr_sweep(
    initial_ground_truth_matrix_lr, ground_truth_matrix_lr
):
    sweep = [0.0, initial_ground_truth_matrix_lr]
    for i in range(1, LINE_SEARCH_LEFT_STEPS + 1):
        sweep.append(lr_from_exponent(initial_ground_truth_matrix_lr, i))
    for i in range(1, LINE_SEARCH_RIGHT_STEPS + 1):
        sweep.append(lr_from_exponent(initial_ground_truth_matrix_lr, -i))
    sweep.append(ground_truth_matrix_lr)
    return sweep


def format_matrix_lr_loss_map(losses_by_matrix_lr):
    return {
        "%.8g" % matrix_lr: loss
        for matrix_lr, loss in sorted(losses_by_matrix_lr.items())
    }


def format_matrix_lrs(matrix_lrs):
    return [float("%.8g" % matrix_lr) for matrix_lr in matrix_lrs]


def line_search_validation_sets(epoch_batches, batch_index):
    return [
        ("current_batch", [epoch_batches[batch_index]]),
    ]


############################################
#                Training                  #
############################################


def train_proxy(hyp, model, data_seed):

    batch_size = hyp["opt"]["batch_size"]
    epochs = hyp["opt"]["train_epochs"]
    momentum = hyp["opt"]["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["opt"]["lr"] / kilostep_scale  # un-decoupled learning rate for PyTorch SGD
    wd = hyp["opt"]["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["opt"]["bias_scaler"]

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=hyp["opt"]["label_smoothing"], reduction="none"
    )
    train_loader = InfiniteCifarLoader(
        CIFAR10_DIR,
        train=True,
        batch_size=batch_size,
        aug=hyp["aug"],
        aug_seed=data_seed,
        order_seed=data_seed,
    )
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(TRAINING_SEED, 1)
    reinit_net(model)
    print("Proxy parameters:", sum(p.numel() for p in model.parameters()))
    current_steps = 0

    norm_biases = [
        p for k, p in model.named_parameters() if "norm" in k and p.requires_grad
    ]
    other_params = [
        p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad
    ]
    param_configs = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other_params, lr=lr, weight_decay=wd / lr),
    ]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Initialize the whitening layer using training images
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(base_model(model)[0], train_images)

    masks = []

    for indices, inputs, labels in train_loader:
        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            model.train()

        # Skip every other backward pass
        if current_steps % 4 == 0:
            outputs = model(inputs)
            loss1 = loss_fn(outputs, labels)
            mask = torch.zeros(len(inputs)).cuda().bool()
            mask[loss1.argsort()[-hyp["opt"]["batch_size_masked"] :]] = True
            masks.append(mask)
            loss = (loss1 * mask.float()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss1 = loss_fn(outputs, labels)
                mask = torch.zeros(len(inputs)).cuda().bool()
                mask[loss1.argsort()[-hyp["opt"]["batch_size_masked"] :]] = True
                masks.append(mask)

        scheduler.step()

        current_steps += 1
        if current_steps == total_train_steps:
            break

    return masks


def main(run, hyp, model_proxy, model_trainbias, model_freezebias):

    batch_size = hyp["opt"]["batch_size"]
    epochs = hyp["opt"]["train_epochs"]
    momentum = hyp["opt"]["momentum"]
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["opt"]["lr"] / kilostep_scale  # un-decoupled learning rate for PyTorch SGD
    wd = hyp["opt"]["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["opt"]["bias_scaler"]

    set_random_state(TRAINING_SEED, 0)
    data_seed = DATA_SEED

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=hyp["opt"]["label_smoothing"], reduction="none"
    )
    test_loader = InfiniteCifarLoader(CIFAR10_DIR, train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader(
        CIFAR10_DIR,
        train=True,
        batch_size=batch_size,
        aug=hyp["aug"],
        aug_seed=data_seed,
        order_seed=data_seed,
    )
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(TRAINING_SEED, 2)
    reinit_net(model_trainbias)
    print(
        "Main model parameters:", sum(p.numel() for p in model_trainbias.parameters())
    )
    current_steps = 0

    norm_biases = [p for k, p in model_trainbias.named_parameters() if "norm" in k]
    other_params = [p for k, p in model_trainbias.named_parameters() if "norm" not in k]
    param_configs = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other_params, lr=lr, weight_decay=wd / lr),
    ]
    optimizer_trainbias = torch.optim.SGD(
        param_configs, momentum=momentum, nesterov=True
    )

    norm_biases = [p for k, p in model_freezebias.named_parameters() if "norm" in k]
    other_params = [
        p for k, p in model_freezebias.named_parameters() if "norm" not in k
    ]
    param_configs = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=other_params, lr=lr, weight_decay=wd / lr),
    ]
    optimizer_freezebias = torch.optim.SGD(
        param_configs, momentum=momentum, nesterov=True
    )

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac

    scheduler_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer_trainbias, get_lr)
    scheduler_freezebias = torch.optim.lr_scheduler.LambdaLR(
        optimizer_freezebias, get_lr
    )
    configure_matrix_lr_multipliers(optimizer_trainbias, lr)
    configure_matrix_lr_multipliers(optimizer_freezebias, lr)

    alpha_schedule = (
        0.95**5 * (torch.arange(total_train_steps + 1) / total_train_steps) ** 3
    )
    lookahead_state = LookaheadState(model_trainbias)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(base_model(model_trainbias)[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    time_seconds += 1e-3 * starter.elapsed_time(ender)

    # Do a small proxy run to collect masks for use in fullsize run
    print("Training small proxy...")
    starter.record()
    masks = iter(train_proxy(hyp, model_proxy, data_seed))
    ender.record()
    torch.cuda.synchronize()
    time_seconds += 1e-3 * starter.elapsed_time(ender)

    for indices, inputs, labels in train_loader:
        # After training the whiten bias for some epochs, swap in the compiled model with frozen bias
        if current_steps == 0:
            model = model_trainbias
            optimizer = optimizer_trainbias
            scheduler = scheduler_trainbias
        elif epoch == hyp["opt"]["whiten_bias_epochs"] * steps_per_epoch:
            model = model_freezebias
            optimizer = optimizer_freezebias
            scheduler = scheduler_freezebias
            model.load_state_dict(model_trainbias.state_dict())
            optimizer.load_state_dict(optimizer_trainbias.state_dict())
            scheduler.load_state_dict(scheduler_trainbias.state_dict())

        ####################
        #     Training     #
        ####################

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            starter.record()
            model.train()

        mask = next(masks)
        inputs = inputs[mask]
        labels = labels[mask]
        outputs = model(inputs)
        loss = loss_fn(outputs, labels).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ground_truth_matrix_lr = matrix_lr_from_optimizer(optimizer)
        line_search_results = evaluate_matrix_lr_grid(
            model,
            [optimizer],
            optimizer,
            line_search_validation_sets([(inputs, labels)], 0),
            initial_ground_truth_matrix_lr_sweep(lr, ground_truth_matrix_lr),
        )
        current_batch_results = line_search_results["current_batch"]
        print(
            "\nline_search step=%d "
            "line_search_matrix_lr=%.8g ground_truth_matrix_lr=%.8g "
            "pre_update_train_loss=%.6f line_search_loss=%.6f "
            "current_batch_line_search_matrix_lr=%.8g "
            "current_batch_line_search_loss=%.6f "
            "current_batch_matrix_lr_train_losses=%s "
            "current_batch_sample_lr_loss_curves=%s"
            % (
                current_steps,
                current_batch_results["best_matrix_lr"],
                ground_truth_matrix_lr,
                finite_loss_item(loss),
                current_batch_results["best_loss"],
                current_batch_results["best_matrix_lr"],
                current_batch_results["best_loss"],
                json.dumps(
                    format_matrix_lr_loss_map(
                        current_batch_results["losses_by_matrix_lr"]
                    ),
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                json.dumps(
                    current_batch_results["sample_lr_loss_curves"],
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            ),
            flush=True,
        )
        optimizer.step()
        scheduler.step()
        model.zero_grad(set_to_none=True)

        current_steps += 1
        if current_steps % 5 == 0:
            lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
        if current_steps == total_train_steps:
            if lookahead_state is not None:
                lookahead_state.update(model, decay=1.0)

        if (current_steps % steps_per_epoch == 0) or (
            current_steps == total_train_steps
        ):
            ender.record()
            torch.cuda.synchronize()
            time_seconds += 1e-3 * starter.elapsed_time(ender)

            ####################
            #    Evaluation    #
            ####################

            # Save the accuracy and loss from the last training batch of the epoch
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = loss.item() / batch_size
            val_acc = evaluate(model, test_loader, tta_level=0)
            print_training_details(locals(), is_final_entry=False)
            run = None  # Only print the run number once

        if current_steps == total_train_steps:
            break

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp["net"]["tta_level"])
    ender.record()
    torch.cuda.synchronize()
    time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc


if __name__ == "__main__":
    setup_reproducibility(TRAINING_SEED)

    model_proxy = make_net(hyp["proxy"])
    model_proxy[0].bias.requires_grad = False
    model_trainbias = make_net(hyp["net"])
    model_freezebias = make_net(hyp["net"])
    model_freezebias[0].bias.requires_grad = False
    model_proxy = maybe_compile(model_proxy)
    model_trainbias = maybe_compile(model_trainbias)
    model_freezebias = maybe_compile(model_freezebias)

    accs = torch.tensor(
        [
            main(
                0,
                hyp,
                model_proxy,
                model_trainbias,
                model_freezebias,
            )
        ]
    )
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs), log_path)
    print(os.path.abspath(log_path))
