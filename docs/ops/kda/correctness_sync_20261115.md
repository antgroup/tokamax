# Backward correctness synchronization (2026-11-15)

Internal main: `18ea6891`. The scope ledger distinguishes the existing backward
migration from the separate compact all-gather CP Draft.

Native TPU7x, four chips/eight logical devices; JAX/jaxlib 0.12.0,
libtpu 0.0.50.1; large second-minor layout enabled, no interpretation.

```
python -m pytest -q -x tokamax/_src/ops/kda/pallas_mosaic_tpu_padding_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_compact_forward_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_compact_backward_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_bwd_fused_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_packed_bwd_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_packed_gradients_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_remat_test.py
```

Full suite: 161 passed (640.34 s). Two white-box tests explicitly request the raw
aligned tape they mutate, preserving every original gradient/NaN/zero assertion.

The final original-position/native carry changes then passed this separate suite:
```
python -m pytest -q -x tokamax/_src/ops/kda/pallas_mosaic_tpu_compact_forward_test.py tokamax/_src/ops/kda/pallas_mosaic_tpu_compact_backward_test.py tokamax/_src/ops/kda/native_padding_carry_test.py tokamax/_src/ops/kda/packing_test.py
```
58 passed (230.02 s): 24 forward, 32 backward, one scratch-poison and one capacity
case. The later analytic short-convolution addition also passes CPU exact checks.
Counts overlap the full suite and must not be summed. CPU/JAX 0.11.1 padding
compatibility: 16 passed (124.12 s). Original numerical tolerances are unchanged;
new native boundary cases reuse the existing 2% relative-RMS/elementwise bounds.

Backward metadata keys on first/last valid rows, routes non-tail padding through
per-token metadata, and consumes defined zero h on empty chunks. Cotangents match
primal dtype, including BF16 initial states. Grouped native records keep physical
token positions; resumed/interleaved labels retain the per-label grouping fallback.
The unchanged independent recurrence remains the correctness oracle.

Same-input internal-main comparison uses frozen seed-619 H64/B1/T128/K=V128 data
and all seven gradients; outputs, inputs and per-leaf errors are retained. A separate
H2 leading-padding scratch-poison control confirms an existing internal-main carry
initialization gap: 16,384 nonfinite outputs versus zero after the upstream fix.
Internal main was not edited. This is not a production-loss reproduction claim.

Keep Draft. Old benchmark results do not establish performance of the new source.

## Raw packed-tape CI fixture followup

The TPU6e packed-forward comparison exposed opaque pullback function identities
in `return_residuals=True` mode. That white-box mode now explicitly requests the
raw aligned tape; it still compares every array leaf with its original bounds.
Ordinary output mode retains the public padding wrapper. No numerical tolerance
or reference change. Directed CPU/Pallas checks: 4 passed (13.41 s). This is a
test-only change; the benchmark calculation files are unchanged.
