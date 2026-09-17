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


_UPDATE_EPS = 1e-12


def _target_frobenius(G, use_max=False):
    size = max(G.shape) if use_max else min(G.shape)
    return size**0.5


def _match_frobenius(G, target):
    return G * (target / G.norm().clamp_min(_UPDATE_EPS))


def _unit_rows(G):
    return G / G.norm(dim=1, keepdim=True).clamp_min(_UPDATE_EPS)


def _unit_columns(G):
    return G / G.norm(dim=0, keepdim=True).clamp_min(_UPDATE_EPS)


def normalize_columns(G):
    assert len(G.shape) == 2
    return _unit_columns(G) * (min(G.shape) / G.size(1)) ** 0.5


def normalize_columns_max(G):
    assert len(G.shape) == 2
    return _unit_columns(G) * (max(G.shape) / G.size(1)) ** 0.5


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


def sinkhorn_rows_first_norm(G):
    assert len(G.shape) == 2
    X = _unit_rows(G)
    X = _unit_columns(X)
    X = _unit_rows(X)
    return _match_frobenius(X, _target_frobenius(G))


def sinkhorn_columns_first_norm(G):
    assert len(G.shape) == 2
    X = _unit_columns(G)
    X = _unit_rows(X)
    X = _unit_columns(X)
    return _match_frobenius(X, _target_frobenius(G))


def row_centered_row_norm(G):
    assert len(G.shape) == 2
    X = G - G.mean(dim=1, keepdim=True)
    return _match_frobenius(_unit_rows(X), _target_frobenius(G))


def column_centered_column_norm(G):
    assert len(G.shape) == 2
    X = G - G.mean(dim=0, keepdim=True)
    return _match_frobenius(_unit_columns(X), _target_frobenius(G))


def double_centered_matrix_norm(G):
    assert len(G.shape) == 2
    X = G - G.mean(dim=1, keepdim=True) - G.mean(dim=0, keepdim=True) + G.mean()
    return _match_frobenius(X, _target_frobenius(G))


def _signed_power_matrix_norm(G, power):
    X = G.sign() * G.abs().pow(power)
    return _match_frobenius(X, _target_frobenius(G))


def signed_sqrt_matrix_norm(G):
    assert len(G.shape) == 2
    return _signed_power_matrix_norm(G, 0.5)


def signed_cuberoot_matrix_norm(G):
    assert len(G.shape) == 2
    return _signed_power_matrix_norm(G, 1 / 3)


def signed_square_matrix_norm(G):
    assert len(G.shape) == 2
    return _signed_power_matrix_norm(G, 2.0)


def softsign_matrix_norm(G):
    assert len(G.shape) == 2
    scale = G.abs().mean().clamp_min(_UPDATE_EPS)
    X = G / (G.abs() + scale)
    return _match_frobenius(X, _target_frobenius(G))


def tanh_matrix_norm(G):
    assert len(G.shape) == 2
    scale = G.abs().mean().clamp_min(_UPDATE_EPS)
    return _match_frobenius(torch.tanh(G / scale), _target_frobenius(G))


def _factorized_rms(G):
    square = G.square()
    row_ms = square.mean(dim=1, keepdim=True)
    column_ms = square.mean(dim=0, keepdim=True)
    global_ms = square.mean().clamp_min(_UPDATE_EPS)
    return (row_ms * column_ms / global_ms).clamp_min(_UPDATE_EPS).sqrt()


def factorized_rms_norm(G):
    assert len(G.shape) == 2
    X = G / _factorized_rms(G)
    return _match_frobenius(X, _target_frobenius(G))


def inverse_factorized_rms_norm(G):
    assert len(G.shape) == 2
    X = G * _factorized_rms(G)
    return _match_frobenius(X, _target_frobenius(G))


def row_norm_sqrt_weighted(G):
    assert len(G.shape) == 2
    row_norms = G.norm(dim=1, keepdim=True).clamp_min(_UPDATE_EPS)
    weights = (row_norms / row_norms.mean().clamp_min(_UPDATE_EPS)).sqrt()
    return _match_frobenius(_unit_rows(G) * weights, _target_frobenius(G))


def row_norm_inv_sqrt_weighted(G):
    assert len(G.shape) == 2
    row_norms = G.norm(dim=1, keepdim=True).clamp_min(_UPDATE_EPS)
    weights = (row_norms.mean().clamp_min(_UPDATE_EPS) / row_norms).sqrt()
    return _match_frobenius(_unit_rows(G) * weights, _target_frobenius(G))


def qr_row_orthogonal(G):
    assert len(G.shape) == 2
    if G.size(0) <= G.size(1):
        q, r = torch.linalg.qr(G.T, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        X = (q * signs).T
    else:
        q, r = torch.linalg.qr(G, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        X = q * signs
    return _match_frobenius(X, _target_frobenius(G))


def zeropower_via_newtonschulz5_steps1(G):
    assert len(G.shape) == 2
    return zeropower_via_newtonschulz5(G, steps=1, eps=_UPDATE_EPS)


def zeropower_via_newtonschulz5_steps4(G):
    assert len(G.shape) == 2
    return zeropower_via_newtonschulz5(G, steps=4, eps=_UPDATE_EPS)


def zeropower_double_centered(G):
    assert len(G.shape) == 2
    X = G - G.mean(dim=1, keepdim=True) - G.mean(dim=0, keepdim=True) + G.mean()
    return zeropower_via_newtonschulz5(X, steps=3, eps=_UPDATE_EPS)


def context_update(fn):
    fn._accepts_update_context = True
    return fn


def apply_update_fn(update_fn, G, state, step, total_steps, param, lr):
    if getattr(update_fn, "_accepts_update_context", False):
        return update_fn(
            G, state=state, step=step, total_steps=total_steps, param=param, lr=lr
        )
    return update_fn(G)


def _schedule_fraction(step, total_steps):
    if total_steps is None or total_steps <= 1:
        return 0.0
    return min(1.0, max(0.0, step / (total_steps - 1)))


def _linear_ramp(progress, start, end):
    if end <= start:
        return 1.0 if progress >= end else 0.0
    return min(1.0, max(0.0, (progress - start) / (end - start)))


def _gram_frobenius_norm_estimate(G, keepdim=False, eps=1e-10):
    assert len(G.shape) == 2
    G = G.float()
    gram = G @ G.T if G.size(0) <= G.size(1) else G.T @ G
    norm = gram.norm().sqrt().clamp_min(eps)
    if keepdim:
        return norm.view(1, 1)
    return norm


def _match_gram_frobenius_norm(G, reference, eps=1e-10):
    return G * (
        _gram_frobenius_norm_estimate(reference, eps=eps)
        / _gram_frobenius_norm_estimate(G, eps=eps)
    ).to(G.dtype)


def zeropower_track3(G, steps=12, eps=1e-7):
    assert len(G.shape) == 2
    X = G.to(MUON_DTYPE if G.is_cuda else torch.float32)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / _gram_frobenius_norm_estimate(X, keepdim=True, eps=eps).to(X.dtype)
    for _ in range(steps):
        A = X @ X.T
        B = -1.5 * A + 0.5 * A @ A
        X = 2.0 * X + B @ X
    if transpose:
        X = X.T
    return X


def _soft_coefficients(p):
    if p == 0.1:
        return 0.0, (
            0.1091613623,
            0.07085664498,
            0.05210528973,
            0.05457295795,
            0.05011334061,
            0.03334622198,
            0.05022104481,
            0.1053727358,
            0.1187323776,
            0.1185061091,
            0.1185059576,
            0.1185059576,
        )
    raise ValueError(f"unsupported soft-muon singular-value power: {p}")


def soft_track3(G, p=0.1):
    assert len(G.shape) == 2
    X = G.to(MUON_DTYPE if G.is_cuda else torch.float32)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / _gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    constant, coeffs = _soft_coefficients(p)
    basis = [X]
    for _ in range(len(coeffs)):
        A = X @ X.T
        B = -1.5 * A + 0.5 * A @ A
        X = 2.0 * X + B @ X
        basis.append(X)
    out = constant * basis[-1]
    for coeff, basis_term in zip(coeffs, basis[:-1]):
        out = out + coeff * basis_term
    if transpose:
        out = out.T
    return out


def _track3_scaled_polar(G):
    update = zeropower_track3(G)
    update *= max(1.0, G.size(0) / G.size(1)) ** 0.5
    return _match_frobenius(update.float(), _target_frobenius(G)).to(G.dtype)


def _track3_soft_scaled(G):
    polar = zeropower_track3(G)
    soft = soft_track3(G)
    soft = _match_gram_frobenius_norm(soft, polar)
    soft *= max(1.0, G.size(0) / G.size(1)) ** 0.5
    return _match_frobenius(soft.float(), _target_frobenius(G)).to(G.dtype)


def _long_axis_second_moment(update, state, key, beta2=0.95):
    if update.size(0) >= update.size(1):
        moment = update.float().square().mean(dim=1, keepdim=True)
    else:
        moment = update.float().square().mean(dim=0, keepdim=True)
    if key not in state or state[key].shape != moment.shape:
        state[key] = torch.zeros_like(moment)
    state[key].lerp_(moment, 1 - beta2)
    return state[key].clamp_min(1e-10)


def _normon_postcondition(update, state, key="nor_second_moment", beta2=0.95):
    reference = update
    second_moment = _long_axis_second_moment(update, state, key, beta2=beta2)
    update = update * second_moment.rsqrt().to(update.dtype)
    return _match_gram_frobenius_norm(update, reference).to(reference.dtype)


def _contra_soft_blend(G, step, total_steps, use_contra=True, use_soft=True):
    polar = zeropower_track3(G)
    reference_norm = _gram_frobenius_norm_estimate(polar)
    normalized_grad = G / _gram_frobenius_norm_estimate(G, keepdim=True).to(G.dtype)

    progress = _schedule_fraction(step, total_steps)
    contra_coeff = -0.2 * (1.0 - _linear_ramp(progress, 0.0, 0.65)) if use_contra else 0
    update = polar + contra_coeff * normalized_grad
    update = update * (reference_norm / _gram_frobenius_norm_estimate(update)).to(
        update.dtype
    )

    if use_soft:
        soft = soft_track3(G)
        soft = soft * (reference_norm / _gram_frobenius_norm_estimate(soft)).to(
            soft.dtype
        )
        soft_blend = min(0.8, _linear_ramp(progress, 0.80, 0.98))
        update = update + (soft - update) * soft_blend
        update = update * (reference_norm / _gram_frobenius_norm_estimate(update)).to(
            update.dtype
        )
    update *= max(1.0, G.size(0) / G.size(1)) ** 0.5
    return _match_frobenius(update.float(), _target_frobenius(G)).to(G.dtype)


def _row_balanced_gradient(G, power=1.0):
    row_rms = G.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(_UPDATE_EPS)
    row_rms = row_rms / row_rms.mean().clamp_min(_UPDATE_EPS)
    return G / row_rms.to(G.dtype).pow(power)


def _muown_row_norm_control(update, state, key="muown_row_norm", beta=0.9):
    row_norm = update.float().norm(dim=1, keepdim=True).clamp_min(_UPDATE_EPS)
    if key not in state or state[key].shape != row_norm.shape:
        state[key] = row_norm.detach().clone()
    state[key].lerp_(row_norm, 1 - beta)
    target = state[key].mean().clamp_min(_UPDATE_EPS)
    scale = (target / state[key]).sqrt().to(update.dtype)
    controlled = update * scale
    return _match_gram_frobenius_norm(controlled, update).to(update.dtype)


def _soap_eigenbasis(mat):
    eye = torch.eye(mat.size(0), dtype=mat.dtype, device=mat.device)
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * eye)
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * eye.double())
        q = q.float()
    return torch.flip(q, [1])


def _soap_precondition(
    G,
    state,
    prefix,
    beta2=0.90,
    denom_power=0.5,
    frequency=10,
    blend=1.0,
    denom_floor_ratio=0.0,
):
    Gf = G.float()
    row_key, col_key = f"{prefix}_row_gg", f"{prefix}_col_gg"
    q_row_key, q_col_key = f"{prefix}_q_row", f"{prefix}_q_col"
    exp_key, step_key = f"{prefix}_exp_avg_sq", f"{prefix}_step"
    if row_key not in state or state[row_key].shape != (G.size(0), G.size(0)):
        state[row_key] = torch.zeros(
            G.size(0), G.size(0), dtype=torch.float32, device=G.device
        )
        state[col_key] = torch.zeros(
            G.size(1), G.size(1), dtype=torch.float32, device=G.device
        )
        state[q_row_key] = None
        state[q_col_key] = None
        state[exp_key] = torch.zeros_like(Gf)
        state[step_key] = 0
    state[row_key].lerp_(Gf @ Gf.T, 1 - beta2)
    state[col_key].lerp_(Gf.T @ Gf, 1 - beta2)
    if state[q_row_key] is None or state[step_key] % frequency == 0:
        state[q_row_key] = _soap_eigenbasis(state[row_key])
        state[q_col_key] = _soap_eigenbasis(state[col_key])
    state[step_key] += 1

    q_row, q_col = state[q_row_key], state[q_col_key]
    projected = q_row.T @ Gf @ q_col
    state[exp_key].lerp_(projected.square(), 1 - beta2)
    denom = state[exp_key].clamp_min(1e-16).pow(denom_power)
    if denom_floor_ratio > 0:
        floor = denom.float().square().mean().sqrt().mul(denom_floor_ratio)
        denom = denom.clamp_min(floor.clamp_min(1e-8))
    preconditioned = q_row @ (projected / denom) @ q_col.T
    if blend != 1.0:
        preconditioned = preconditioned * blend + Gf * (1 - blend)
    return _match_gram_frobenius_norm(preconditioned, Gf).to(G.dtype)


def _shampoo_precondition(G, state, prefix, beta2=0.90, power=0.25):
    Gf = G.float()
    row_key, col_key = f"{prefix}_row_gg", f"{prefix}_col_gg"
    if row_key not in state or state[row_key].shape != (G.size(0), G.size(0)):
        state[row_key] = torch.zeros(
            G.size(0), G.size(0), dtype=torch.float32, device=G.device
        )
        state[col_key] = torch.zeros(
            G.size(1), G.size(1), dtype=torch.float32, device=G.device
        )
    state[row_key].lerp_(Gf @ Gf.T, 1 - beta2)
    state[col_key].lerp_(Gf.T @ Gf, 1 - beta2)
    evals_r, q_r = torch.linalg.eigh(
        state[row_key] + 1e-8 * torch.eye(G.size(0), device=G.device)
    )
    evals_c, q_c = torch.linalg.eigh(
        state[col_key] + 1e-8 * torch.eye(G.size(1), device=G.device)
    )
    left = q_r @ torch.diag(evals_r.clamp_min(1e-8).pow(-power)) @ q_r.T
    right = q_c @ torch.diag(evals_c.clamp_min(1e-8).pow(-power)) @ q_c.T
    preconditioned = left @ Gf @ right
    return _match_gram_frobenius_norm(preconditioned, Gf).to(G.dtype)


def _sinkhorn_precondition(G, rounds=3):
    X = G.float()
    for _ in range(rounds):
        X = X / X.square().mean(dim=1, keepdim=True).sqrt().clamp_min(_UPDATE_EPS)
        X = X / X.square().mean(dim=0, keepdim=True).sqrt().clamp_min(_UPDATE_EPS)
    return _match_gram_frobenius_norm(X, G).to(G.dtype)


def _radial_brake(update, param, brake=0.75):
    if param is None:
        return update
    Pf = param.float()
    denom = Pf.square().sum().clamp_min(_UPDATE_EPS)
    coeff = (update.float() * Pf).sum() / denom
    outward = coeff.clamp(max=0.0)
    braked = update.float() - brake * outward * Pf
    return _match_gram_frobenius_norm(braked, update).to(update.dtype)


def _soda_anchor_term(update, state, param, lr, step, total_steps, strength=0.03):
    if param is None or lr is None or lr <= 0:
        return update
    if "soda_anchor" not in state or state["soda_anchor"].shape != param.shape:
        state["soda_anchor"] = param.detach().clone().float()
    progress = _schedule_fraction(step, total_steps)
    fade = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
    correction = (param.float() - state["soda_anchor"]) * (strength * fade / lr)
    anchored = update.float() + correction
    return _match_gram_frobenius_norm(anchored, update).to(update.dtype)


@context_update
def track3_muon(G, **kwargs):
    return _track3_scaled_polar(G)


@context_update
def track3_soft_muon_p01(G, **kwargs):
    return _track3_soft_scaled(G)


@context_update
def track3_contra_muon(G, step=0, total_steps=None, **kwargs):
    return _contra_soft_blend(G, step, total_steps, use_contra=True, use_soft=False)


@context_update
def track3_contra_to_soft_muon(G, step=0, total_steps=None, **kwargs):
    return _contra_soft_blend(G, step, total_steps, use_contra=True, use_soft=True)


@context_update
def normon_muon(G, state=None, **kwargs):
    return _normon_postcondition(_track3_scaled_polar(G), state, key="normon")


@context_update
def normon_soft_muon(G, state=None, **kwargs):
    return _normon_postcondition(_track3_soft_scaled(G), state, key="normon_soft")


@context_update
def normon_contra_to_soft_muon(G, state=None, step=0, total_steps=None, **kwargs):
    update = _contra_soft_blend(G, step, total_steps, use_contra=True, use_soft=True)
    return _normon_postcondition(update, state, key="normon_contra_soft")


@context_update
def aurora_row_balanced_muon(G, **kwargs):
    return _track3_scaled_polar(_row_balanced_gradient(G))


@context_update
def aurora_half_balanced_muon(G, **kwargs):
    return _track3_scaled_polar(_row_balanced_gradient(G, power=0.5))


@context_update
def aurora_normon_muon(G, state=None, **kwargs):
    update = _track3_scaled_polar(_row_balanced_gradient(G))
    return _normon_postcondition(update, state, key="aurora_normon")


@context_update
def muown_row_control_muon(G, state=None, **kwargs):
    return _muown_row_norm_control(_track3_scaled_polar(G), state, key="muown")


@context_update
def muown_normon_muon(G, state=None, **kwargs):
    update = _muown_row_norm_control(_track3_scaled_polar(G), state, key="muown_nor")
    return _normon_postcondition(update, state, key="muown_normon")


@context_update
def soap_muon(G, state=None, **kwargs):
    return _track3_scaled_polar(_soap_precondition(G, state, prefix="soap"))


@context_update
def soap_normon_muon(G, state=None, **kwargs):
    update = _track3_scaled_polar(_soap_precondition(G, state, prefix="soap_nor"))
    return _normon_postcondition(update, state, key="soap_normon")


@context_update
def soap_contra_soft_normon(G, state=None, step=0, total_steps=None, **kwargs):
    preconditioned = _soap_precondition(
        G, state, prefix="soap_contra_soft", blend=1.0, denom_floor_ratio=0.55
    )
    update = _contra_soft_blend(
        preconditioned, step, total_steps, use_contra=True, use_soft=True
    )
    return _normon_postcondition(update, state, key="soap_contra_soft_normon")


@context_update
def sinksoap_normon_muon(G, state=None, **kwargs):
    preconditioned = _sinkhorn_precondition(
        _soap_precondition(G, state, prefix="sinksoap", blend=0.8)
    )
    update = _track3_scaled_polar(preconditioned)
    return _normon_postcondition(update, state, key="sinksoap_normon")


@context_update
def kl_soap_muon(G, state=None, **kwargs):
    preconditioned = _soap_precondition(
        G, state, prefix="kl_soap", beta2=0.95, denom_power=0.35, denom_floor_ratio=0.2
    )
    return _track3_scaled_polar(preconditioned)


@context_update
def shampoo_muon(G, state=None, **kwargs):
    return _track3_scaled_polar(_shampoo_precondition(G, state, prefix="shampoo"))


@context_update
def radial_brake_soft_normon(G, state=None, param=None, **kwargs):
    update = _normon_postcondition(_track3_soft_scaled(G), state, key="radial_soft")
    return _radial_brake(update, param)


@context_update
def soda_contra_soft_normon(G, state=None, step=0, total_steps=None, param=None, lr=None):
    update = normon_contra_to_soft_muon(
        G, state=state, step=step, total_steps=total_steps
    )
    return _soda_anchor_term(update, state, param, lr, step, total_steps)


if USE_COMPILED_MUON:
    normalize_rows = torch.compile(normalize_rows)
    normalize_rows_max = torch.compile(normalize_rows_max)
    normalize_matrix = torch.compile(normalize_matrix)
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    normalize_matrix_max = torch.compile(normalize_matrix_max)
    zeropower_via_newtonschulz5_max = torch.compile(zeropower_via_newtonschulz5_max)
    normalize_columns = torch.compile(normalize_columns)
    normalize_columns_max = torch.compile(normalize_columns_max)
    sinkhorn_rows_first_norm = torch.compile(sinkhorn_rows_first_norm)
    sinkhorn_columns_first_norm = torch.compile(sinkhorn_columns_first_norm)
    row_centered_row_norm = torch.compile(row_centered_row_norm)
    column_centered_column_norm = torch.compile(column_centered_column_norm)
    double_centered_matrix_norm = torch.compile(double_centered_matrix_norm)
    signed_sqrt_matrix_norm = torch.compile(signed_sqrt_matrix_norm)
    signed_cuberoot_matrix_norm = torch.compile(signed_cuberoot_matrix_norm)
    signed_square_matrix_norm = torch.compile(signed_square_matrix_norm)
    softsign_matrix_norm = torch.compile(softsign_matrix_norm)
    tanh_matrix_norm = torch.compile(tanh_matrix_norm)
    factorized_rms_norm = torch.compile(factorized_rms_norm)
    inverse_factorized_rms_norm = torch.compile(inverse_factorized_rms_norm)
    row_norm_sqrt_weighted = torch.compile(row_norm_sqrt_weighted)
    row_norm_inv_sqrt_weighted = torch.compile(row_norm_inv_sqrt_weighted)
    qr_row_orthogonal = torch.compile(qr_row_orthogonal)
    zeropower_via_newtonschulz5_steps1 = torch.compile(
        zeropower_via_newtonschulz5_steps1
    )
    zeropower_via_newtonschulz5_steps4 = torch.compile(
        zeropower_via_newtonschulz5_steps4
    )
    zeropower_double_centered = torch.compile(zeropower_double_centered)


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
        self.step_count = 0
        self.total_steps = None

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
                update = apply_update_fn(
                    update_fn=update_fn,
                    G=g.reshape(len(g), -1),
                    state=state,
                    step=self.step_count,
                    total_steps=self.total_steps,
                    param=p.data.reshape(len(p.data), -1),
                    lr=lr,
                ).view(g.shape)
                p.data.add_(update, alpha=-lr)  # take a step
        self.step_count += 1


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
    dict(batch_size=2000),
]


def update_config(update_name, update_fn, muon_lr=0.22):
    return dict(update_name=update_name, update_fn=update_fn, muon_lr=muon_lr)


# Leaderboard-inspired Muon-family candidates from Track 3 optimization.
UPDATE_CONFIGS = [
    update_config("track3_muon", track3_muon),
    update_config("track3_soft_muon_p01", track3_soft_muon_p01),
    update_config("track3_contra_muon", track3_contra_muon),
    update_config("track3_contra_to_soft_muon", track3_contra_to_soft_muon),
    update_config("normon_muon", normon_muon),
    update_config("normon_soft_muon", normon_soft_muon),
    update_config("normon_contra_to_soft_muon", normon_contra_to_soft_muon),
    update_config("aurora_row_balanced_muon", aurora_row_balanced_muon),
    update_config("aurora_half_balanced_muon", aurora_half_balanced_muon),
    update_config("aurora_normon_muon", aurora_normon_muon),
    update_config("muown_row_control_muon", muown_row_control_muon),
    update_config("muown_normon_muon", muown_normon_muon),
    update_config("soap_muon", soap_muon),
    update_config("soap_normon_muon", soap_normon_muon),
    update_config("soap_contra_soft_normon", soap_contra_soft_normon),
    update_config("sinksoap_normon_muon", sinksoap_normon_muon),
    update_config("kl_soap_muon", kl_soap_muon),
    update_config("shampoo_muon", shampoo_muon),
    update_config("radial_brake_soft_normon", radial_brake_soft_normon),
    update_config("soda_contra_soft_normon", soda_contra_soft_normon),
]
assert len(UPDATE_CONFIGS) == 20
RUN_CONFIGS = [
    dict(
        **batch_config,
        **update_config,
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
    optimizer2.total_steps = total_train_steps
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
