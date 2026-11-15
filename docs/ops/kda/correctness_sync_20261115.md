# Inference correctness synchronization (2026-11-15)

The standalone state-safe forward kernel is retained. It does not import or register
a training VJP. The common parent padding correctness fixes are synchronized without
claiming backward/CP optimizations as inference work.

Native TPU7x, four chips/eight logical devices; JAX/jaxlib 0.12.0,
libtpu 0.0.50.1, large second-minor layout enabled, no interpretation.

```
python -m pytest -q -x tokamax/_src/ops/kda/inference_test.py tokamax/_src/ops/kda/inference_lowering_test.py
```

Final source: 28 passed (67.56 s). Covers initial/final and unused state slots,
leading/interior padding, repeated labels, 128 one-token fragments, shared maps,
B1/B3, and 64-head grouped input tiles. Existing elementwise and 2% relative-norm
bounds are unchanged. Invalid outputs remain zero; the reference was not changed.

Grouped records keep original token positions. Interleaved labels are stably grouped,
with initial-state slots gathered and final states restored to their logical slots.
The chain is initialized even when its first chunk is all padding, and segment 0
cannot be saved as a final state. First/last-valid metadata and physical head-tile
layout fixes are ported into this state-safe core rather than substituting the
training-only compact implementation.

Common parent validation: 16 CPU/JAX 0.11.1 padding cases passed (146.27 s), plus
exact logical-capacity and short-history convolution checks. These are correctness
checks, not inference speed measurements. Formatting preserves the Python AST.

The migration ledger pins internal main at `18ea6891`, records all recent deltas
and the deterministic scratch-poison control. Historical inference performance
numbers predate this commit; no new speedup is claimed here. Keep Draft pending
full upstream CI and performance review.
