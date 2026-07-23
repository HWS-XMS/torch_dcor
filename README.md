# dcor_torch

GPU-batched **distance correlation** (Székely–Rizzo 2007) and **partial distance
correlation** (Székely–Rizzo 2014) in PyTorch, verified against the reference
[`dcor`](https://dcor.readthedocs.io) package to `< 1e-6`.

The batched API computes a dependence value for each of `S` signals in one GPU
pass, so large sweeps run in seconds to minutes instead of hours.

## Install

```bash
pip install .            # or: pip install -e .
pip install ".[test]"    # optional: adds dcor, numpy, pytest for cross-checks
```

Requires `torch >= 2.0` and Python `>= 3.10`. Uses CUDA automatically when the
inputs are on a CUDA device; otherwise runs on CPU.

## API

```python
import torch
from dcor_torch import dcor, pdcor, dcor_per_sample, pdcor_per_sample

# scalars
dcor(x, y)          # biased distance correlation in [0, 1]   (SR2007)
pdcor(x, y, z)      # signed partial distance correlation      (SR2014)

# batched over a leading S axis
dcor_per_sample(X, y)        # X:(S,N) or (S,N,d), y:(N,) or (N,d)  -> (S,)
pdcor_per_sample(X, y, Z)    # X,Z:(S,N[,d]), y shared              -> (S,)
```

- Inputs may be 1-D `(N,)` scalar-per-observation or 2-D `(N, d)`
  vector-per-observation (e.g. `d=2` for complex `(Re, Im)`).
- `pdcor` uses the SR2014 projection form and **can be negative**.
- Computation is in **fp64**; memory is bounded by VRAM-aware chunking
  (`chunk_size=` to override), which is numerically exact.
- `pdcor(x, y, z) = 0` does **not** imply `x ⟂ y | z` — it is a projection, not
  a conditional-independence test (a property of the SR2014 statistic).

## Tests

```bash
pytest -q    # edge cases need only numpy+torch; *_vs_reference tests need dcor
```

## References

- G. J. Székely, M. L. Rizzo, N. K. Bakirov. *Measuring and testing dependence
  by correlation of distances.* Ann. Statist. 35(6):2769–2794, 2007.
- G. J. Székely, M. L. Rizzo. *Partial distance correlation with methods for
  dissimilarities.* Ann. Statist. 42(6):2382–2412, 2014.

## License

MIT — see [LICENSE](LICENSE).
