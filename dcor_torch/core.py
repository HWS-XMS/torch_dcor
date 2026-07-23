"""Torch-batched distance correlation (Szekely-Rizzo 2007) and partial distance
correlation (Szekely-Rizzo 2014).

Public API
----------
- ``dcor(x, y)`` -- biased scalar dCor in ``[0, 1]``.
- ``pdcor(x, y, z)`` -- signed scalar pdCor.
- ``dcor_per_sample(X, y, chunk_size=None)`` -- ``(S,)`` biased dCor per sample.
- ``pdcor_per_sample(X, y, Z, chunk_size=None)`` -- ``(S,)`` signed pdCor per sample.

Inputs may be 1-D ``(N,)`` scalar-per-observation or 2-D ``(N, d)`` vector-per-
observation (e.g. ``d=2`` for (Re, Im) complex).  Batched inputs stack an extra
leading ``S`` axis.

fp64 is used throughout for numerical stability against the reference
``dcor`` package.
"""
from __future__ import annotations

import torch
from torch import Tensor


# ---------- pairwise distance ----------------------------------------------

def _to_2d(x: Tensor) -> Tensor:
    x = x.to(torch.float64)
    return x[..., None] if x.ndim == 1 else x


def _to_3d(X: Tensor) -> Tensor:
    X = X.to(torch.float64)
    return X[..., None] if X.ndim == 2 else X


def _pdist(x: Tensor) -> Tensor:
    """(N,) or (N, d) -> (N, N) Euclidean pairwise distance."""
    x2 = _to_2d(x)
    return torch.cdist(x2, x2)


def _pdist_batched(X: Tensor) -> Tensor:
    """(S, N) or (S, N, d) -> (S, N, N)."""
    X3 = _to_3d(X)
    return torch.cdist(X3, X3)


# ---------- centering ------------------------------------------------------

def _double_center(D: Tensor) -> Tensor:
    """Biased (SR2007) double-centering.  ``D: (..., N, N) -> (..., N, N)``.
    Mutates D in place to keep peak memory at one (N, N) tensor per batch."""
    row = D.mean(dim=-1, keepdim=True)
    col = D.mean(dim=-2, keepdim=True)
    tot = D.mean(dim=(-1, -2), keepdim=True)
    D.sub_(row).sub_(col).add_(tot)
    return D


def _u_center(D: Tensor) -> Tensor:
    """Unbiased (SR2014) U-centering.  ``D: (..., N, N) -> (..., N, N)`` with a
    zeroed diagonal (paper convention).  Mutates D in place."""
    N = D.shape[-1]
    if N < 4:
        raise ValueError(f"U-centering requires N >= 4; got N={N}")
    row_sum = D.sum(dim=-1, keepdim=True)
    col_sum = D.sum(dim=-2, keepdim=True)
    tot_sum = D.sum(dim=(-1, -2), keepdim=True)
    row_sum.div_(N - 2)
    col_sum.div_(N - 2)
    tot_sum.div_((N - 1) * (N - 2))
    D.sub_(row_sum).sub_(col_sum).add_(tot_sum)
    eye = torch.eye(N, dtype=torch.bool, device=D.device)
    D.masked_fill_(eye, 0.0)
    return D


# ---------- inner products -------------------------------------------------

def _inner_biased(A: Tensor, B: Tensor) -> Tensor:
    """``(1/N^2) sum_ij A_ij B_ij`` along the last two axes."""
    N = A.shape[-1]
    return (A * B).sum(dim=(-1, -2)) / (N * N)


def _inner_u(A: Tensor, B: Tensor) -> Tensor:
    """SR2014 U-inner product ``(1/(N(N-3))) sum_{i!=j} A_ij B_ij``.
    Diagonals are already zero after ``_u_center``, so the plain sum suffices."""
    N = A.shape[-1]
    return (A * B).sum(dim=(-1, -2)) / (N * (N - 3))


# ---------- scalar dCor / pdCor -------------------------------------------

def dcor(x: Tensor, y: Tensor) -> Tensor:
    """Biased distance correlation (SR2007).  Returns a scalar ``[0, 1]``."""
    A = _double_center(_pdist(x))
    B = _double_center(_pdist(y))
    dcov2 = _inner_biased(A, B)
    dvar_x = _inner_biased(A, A)
    dvar_y = _inner_biased(B, B)
    denom = torch.sqrt(dvar_x * dvar_y)
    if denom.item() <= 0.0:
        return torch.tensor(0.0, dtype=torch.float64, device=A.device)
    return torch.sqrt(dcov2.clamp(min=0.0) / denom)


def pdcor(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Signed partial distance correlation (SR2014).  Returns a scalar
    (can be negative)."""
    A_x = _u_center(_pdist(x))
    A_y = _u_center(_pdist(y))
    A_z = _u_center(_pdist(z))
    return _pdcor_from_ucentered(A_x, A_y, A_z)


def _pdcor_from_ucentered(A_x: Tensor, A_y: Tensor, A_z: Tensor) -> Tensor:
    """Projection form: pdCor = ((AX,AY) - (AX,AZ)(AY,AZ)/(AZ,AZ))
    / sqrt(||AX - PZ AX||^2 * ||AY - PZ AY||^2)."""
    xy = _inner_u(A_x, A_y)
    xz = _inner_u(A_x, A_z)
    yz = _inner_u(A_y, A_z)
    xx = _inner_u(A_x, A_x)
    yy = _inner_u(A_y, A_y)
    zz = _inner_u(A_z, A_z)
    eps = torch.tensor(1e-15, dtype=xy.dtype, device=xy.device)
    # If Z has no distance variance, projection is undefined -> pdCor := dCor(X, Y)
    z_ok = zz.abs() > eps
    beta_x = torch.where(z_ok, xz / zz.clamp(min=eps), torch.zeros_like(xz))
    beta_y = torch.where(z_ok, yz / zz.clamp(min=eps), torch.zeros_like(yz))
    pdcov = xy - beta_x * yz
    dx = (xx - beta_x * xz).clamp(min=0.0)
    dy = (yy - beta_y * yz).clamp(min=0.0)
    denom = torch.sqrt(dx * dy)
    return torch.where(denom > eps, pdcov / denom.clamp(min=eps),
                       torch.zeros_like(pdcov))


# ---------- chunking -------------------------------------------------------

def _auto_chunk(N: int, device: torch.device, matrices: int) -> int:
    """Pick a chunk size that keeps the concurrent (K, N, N) fp64 tensors under
    25% of free VRAM on CUDA (headroom for cdist scratch + centering intermediates
    even with the in-place path); a fixed small value on CPU."""
    if device.type != "cuda":
        return 32
    free, _ = torch.cuda.mem_get_info(device)
    budget = int(free * 0.25)
    per_sample = matrices * N * N * 8  # fp64 bytes
    return max(1, min(512, budget // per_sample))


# ---------- batched per-sample --------------------------------------------

def dcor_per_sample(X: Tensor, y: Tensor,
                    chunk_size: int | None = None) -> Tensor:
    """Per-sample biased dCor.

    Parameters
    ----------
    X : ``(S, N)`` or ``(S, N, d)`` -- per-sample observations.
    y : ``(N,)`` or ``(N, d)`` -- shared labels/targets.
    chunk_size : optional int; auto-picked from VRAM when ``None``.

    Returns
    -------
    ``(S,)`` fp64 ``[0, 1]``.
    """
    X3 = _to_3d(X)
    S, N, _ = X3.shape
    device = X3.device
    A_y = _double_center(_pdist(y))
    dvar_y = _inner_biased(A_y, A_y)
    if chunk_size is None:
        chunk_size = _auto_chunk(N, device, matrices=2)
    out = torch.empty(S, dtype=torch.float64, device=device)
    A_y_b = A_y[None]
    for i in range(0, S, chunk_size):
        j = min(i + chunk_size, S)
        A_x = _double_center(_pdist_batched(X3[i:j]))
        dcov2 = _inner_biased(A_x, A_y_b)
        dvar_x = _inner_biased(A_x, A_x)
        denom = torch.sqrt(dvar_x * dvar_y)
        val = torch.where(denom > 0,
                          torch.sqrt(dcov2.clamp(min=0.0) / denom.clamp(min=1e-300)),
                          torch.zeros_like(denom))
        out[i:j] = val
    return out


def pdcor_per_sample(X: Tensor, y: Tensor, Z: Tensor,
                     chunk_size: int | None = None) -> Tensor:
    """Per-sample signed pdCor ``(X, y | Z)``.

    Parameters
    ----------
    X, Z : ``(S, N)`` or ``(S, N, d)`` -- per-sample observations and per-sample
        conditioning.
    y : ``(N,)`` or ``(N, d)`` -- shared labels/targets.
    chunk_size : optional int; auto-picked from VRAM when ``None``.

    Returns
    -------
    ``(S,)`` fp64.  Can be negative.
    """
    X3 = _to_3d(X)
    Z3 = _to_3d(Z)
    if X3.shape[:2] != Z3.shape[:2]:
        raise ValueError(f"X and Z must share (S, N); got X {X3.shape}, Z {Z3.shape}")
    S, N, _ = X3.shape
    device = X3.device
    A_y = _u_center(_pdist(y))
    if chunk_size is None:
        chunk_size = _auto_chunk(N, device, matrices=4)  # X, Z + intermediates
    out = torch.empty(S, dtype=torch.float64, device=device)
    A_y_b = A_y[None]
    for i in range(0, S, chunk_size):
        j = min(i + chunk_size, S)
        A_x = _u_center(_pdist_batched(X3[i:j]))
        A_z = _u_center(_pdist_batched(Z3[i:j]))
        out[i:j] = _pdcor_from_ucentered(A_x, A_y_b, A_z)
    return out
