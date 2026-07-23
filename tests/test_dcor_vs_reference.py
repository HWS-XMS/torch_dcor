"""Scalar and batched dCor against the reference `dcor` package."""
import numpy as np
import pytest
import torch
import dcor as ref
from dcor_torch import dcor, dcor_per_sample

TOL = 1e-6


@pytest.mark.parametrize("N", [100, 500, 2000])
def test_scalar_dcor_gaussian(N):
    rng = np.random.default_rng(0xC0FFEE + N)
    x = rng.standard_normal(N)
    y = 0.4 * x + 0.6 * rng.standard_normal(N)
    ours = dcor(torch.from_numpy(x), torch.from_numpy(y)).item()
    theirs = ref.distance_correlation(x, y)
    assert abs(ours - theirs) < TOL, f"N={N}: ours={ours} theirs={theirs}"


@pytest.mark.parametrize("N", [100, 500])
def test_scalar_dcor_vector_valued(N):
    """d=2 per observation (Re/Im-like)."""
    rng = np.random.default_rng(N)
    x = rng.standard_normal((N, 2))
    y = rng.standard_normal((N, 2))
    ours = dcor(torch.from_numpy(x), torch.from_numpy(y)).item()
    theirs = ref.distance_correlation(x, y)
    assert abs(ours - theirs) < TOL


def test_batched_matches_scalar():
    """Row-by-row equivalence between dcor_per_sample and scalar dcor."""
    rng = np.random.default_rng(1)
    S, N = 25, 300
    X = rng.standard_normal((S, N))
    y = rng.standard_normal(N)
    batched = dcor_per_sample(torch.from_numpy(X), torch.from_numpy(y)).cpu().numpy()
    scalar = np.array([
        dcor(torch.from_numpy(X[s]), torch.from_numpy(y)).item()
        for s in range(S)
    ])
    np.testing.assert_allclose(batched, scalar, atol=TOL)


def test_batched_matches_reference():
    rng = np.random.default_rng(2)
    S, N = 40, 500
    X = rng.standard_normal((S, N))
    y = rng.standard_normal(N)
    ours = dcor_per_sample(torch.from_numpy(X), torch.from_numpy(y)).cpu().numpy()
    theirs = np.array([ref.distance_correlation(X[s], y) for s in range(S)])
    np.testing.assert_allclose(ours, theirs, atol=TOL)


def test_chunking_is_exact():
    """Chunked and unchunked results must be bit-close (deterministic reduction)."""
    rng = np.random.default_rng(3)
    S, N = 60, 400
    X = torch.from_numpy(rng.standard_normal((S, N)))
    y = torch.from_numpy(rng.standard_normal(N))
    a = dcor_per_sample(X, y, chunk_size=10).cpu().numpy()
    b = dcor_per_sample(X, y, chunk_size=60).cpu().numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)
