"""Scalar and batched pdCor against the reference `dcor.partial_distance_correlation`."""
import numpy as np
import pytest
import torch
import dcor as ref
from dcor_torch import pdcor, pdcor_per_sample

TOL = 1e-6


@pytest.mark.parametrize("N", [100, 500, 2000])
def test_scalar_pdcor_gaussian(N):
    rng = np.random.default_rng(0xDEC0DE + N)
    z = rng.standard_normal(N)
    x = 0.6 * z + rng.standard_normal(N)
    y = 0.4 * z + rng.standard_normal(N)
    ours = pdcor(torch.from_numpy(x), torch.from_numpy(y),
                 torch.from_numpy(z)).item()
    theirs = ref.partial_distance_correlation(x, y, z)
    assert abs(ours - theirs) < TOL, f"N={N}: ours={ours} theirs={theirs}"


def test_scalar_pdcor_can_be_negative():
    """pdCor is signed; verify sign parity with reference on an antipodal case."""
    rng = np.random.default_rng(4)
    N = 300
    z = rng.standard_normal(N)
    x = 0.7 * z + 0.3 * rng.standard_normal(N)
    y = -0.7 * z + 0.3 * rng.standard_normal(N)
    ours = pdcor(torch.from_numpy(x), torch.from_numpy(y),
                 torch.from_numpy(z)).item()
    theirs = ref.partial_distance_correlation(x, y, z)
    assert abs(ours - theirs) < TOL
    assert np.sign(ours) == np.sign(theirs)


def test_batched_matches_scalar():
    rng = np.random.default_rng(5)
    S, N = 20, 300
    X = rng.standard_normal((S, N))
    Z = rng.standard_normal((S, N))
    y = rng.standard_normal(N)
    batched = pdcor_per_sample(torch.from_numpy(X), torch.from_numpy(y),
                                torch.from_numpy(Z)).cpu().numpy()
    scalar = np.array([
        pdcor(torch.from_numpy(X[s]), torch.from_numpy(y),
              torch.from_numpy(Z[s])).item()
        for s in range(S)
    ])
    np.testing.assert_allclose(batched, scalar, atol=TOL)


def test_batched_matches_reference():
    rng = np.random.default_rng(6)
    S, N = 30, 400
    X = rng.standard_normal((S, N))
    Z = rng.standard_normal((S, N))
    y = rng.standard_normal(N)
    ours = pdcor_per_sample(torch.from_numpy(X), torch.from_numpy(y),
                             torch.from_numpy(Z)).cpu().numpy()
    theirs = np.array([ref.partial_distance_correlation(X[s], y, Z[s])
                       for s in range(S)])
    np.testing.assert_allclose(ours, theirs, atol=TOL)


def test_chunking_is_exact():
    rng = np.random.default_rng(7)
    S, N = 40, 300
    X = torch.from_numpy(rng.standard_normal((S, N)))
    Z = torch.from_numpy(rng.standard_normal((S, N)))
    y = torch.from_numpy(rng.standard_normal(N))
    a = pdcor_per_sample(X, y, Z, chunk_size=7).cpu().numpy()
    b = pdcor_per_sample(X, y, Z, chunk_size=40).cpu().numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)
