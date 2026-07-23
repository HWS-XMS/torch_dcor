"""Sanity edge cases: deterministic dependence, independence, and pdCor
collapsing to zero when the target is entirely explained by the conditioning."""
import numpy as np
import torch
from dcor_torch import dcor, pdcor


def test_deterministic_y_is_dcor_one():
    """Y = affine(X) -> dCor ~ 1."""
    rng = np.random.default_rng(10)
    N = 500
    x = rng.standard_normal(N)
    y = 2.5 * x + 7.0
    v = dcor(torch.from_numpy(x), torch.from_numpy(y)).item()
    assert v > 0.999, f"dCor should be ~1 for affine dependence; got {v}"


def test_independent_dcor_small():
    """Independent samples -> dCor should be near zero at large N."""
    rng = np.random.default_rng(11)
    N = 5000
    x = rng.standard_normal(N)
    y = rng.standard_normal(N)
    v = dcor(torch.from_numpy(x), torch.from_numpy(y)).item()
    assert v < 0.05, f"dCor of independent should be small; got {v}"


def test_pdcor_zero_when_y_is_function_of_z_alone():
    """Y = f(Z) and X independent of everything -> pdCor(X, Y | Z) ~ 0."""
    rng = np.random.default_rng(12)
    N = 1500
    z = rng.standard_normal(N)
    y = z ** 2 + 0.01 * rng.standard_normal(N)  # near-deterministic in Z
    x = rng.standard_normal(N)                   # independent
    v = pdcor(torch.from_numpy(x), torch.from_numpy(y),
              torch.from_numpy(z)).item()
    assert abs(v) < 0.15, f"pdCor should be near zero; got {v}"


def test_pdcor_nonzero_when_extra_dependence_exists():
    """Y depends on both Z and X -> pdCor should be substantial."""
    rng = np.random.default_rng(13)
    N = 1500
    z = rng.standard_normal(N)
    x = rng.standard_normal(N)
    y = 0.5 * z + 0.8 * x + 0.2 * rng.standard_normal(N)
    v = pdcor(torch.from_numpy(x), torch.from_numpy(y),
              torch.from_numpy(z)).item()
    assert v > 0.3, f"pdCor should be well above zero; got {v}"
