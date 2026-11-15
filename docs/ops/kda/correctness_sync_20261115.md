# Correctness sync validation (2026-11-15)

Pinned internal main: `18ea6891f0e8d9affe9e81815267f5251bdb0b3b`.
See `internal_main_migration_20261115.md` for scope, semantic adaptations and
separate compact-CP feature ownership.

## #1688 forward

Native TPU7x, four chips / eight logical devices; Python 3.12, JAX/jaxlib 0.12.0,
libtpu 0.0.50.1. Large second-minor layout enabled. No interpreter.

```
python -m pytest -q -x \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_padding_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_fwd_fused_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_packed_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_compact_forward_test.py
```

Result: 66 passed in 213.62 s. Original tolerances remain unchanged. New padding
cases use the existing fused-forward/XLA numerical bounds; invalid rows are
checked against exact zero. The unused BF16-state derivative is checked with
exact equality to a dtype-matched cotangent (stronger than the former allclose).

The first experimental run and its failures are retained separately. No benchmark
from the preceding head establishes performance of this new correctness patch.
The PR remains Draft; full upstream GPU/TPU CI and final performance review are
separate acceptance steps.
