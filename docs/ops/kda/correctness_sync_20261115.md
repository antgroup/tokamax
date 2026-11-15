# CP correctness synchronization (2026-11-15)

Internal source main is frozen at `18ea6891f0e8d9affe9e81815267f5251bdb0b3b`.
The separate compact all-gather feature is excluded from this PR.

## Validation

Native TPU7x, four chips/eight logical devices; JAX/jaxlib 0.12.0,
libtpu 0.0.50.1, large second-minor layout enabled, no interpretation.

```
python -m pytest -q -x \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_padding_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_workspace_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_workspace_tpu_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_cp_fused_test.py \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_cp_megakernel_test.py
```

106 cases passed before a white-box initial-state test encountered the new
packing tape instead of its required raw kernel tape. That test explicitly
requests the raw tape now, retaining its original gradient assertions and bounds.
The four affected parameter cases then all passed (21.85 s):

```
python -m pytest -q -x \
  tokamax/_src/ops/kda/pallas_mosaic_tpu_cp_megakernel_test.py \
  -k initial_state_grad
```

These runs cover all 110 parameter cases in the listed suite, including CP4/CP8,
B1/B2, capacity 262144, empty ranks, rematerialization and workspace scheduling.
A separate exact CPU/JAX test ensures default CP preserves logical capacity 512
with global label 500 and T_local=4 (`packing_test.py`, 1 passed). Only the explicit
workspace adapter narrows local physical capacity. Public default capacity is not
silently truncated. All numerical tolerances and reference arithmetic are unchanged.

The new padding wrapper preserves original-position convolution and Tokamax's
label-indexed state semantics. Cross-rank branch selection is collective-uniform.
BF16 cotangents match primal dtype; the unused-state derivative uses exact equality
to a dtype-matched cotangent. Pure formatting changes have identical Python ASTs.

The prior official ArgSpec sweep at head `35c7c850` is historical only. A fresh
15-ArgSpec comparison against main `0d3e82b4` is required after this sync. Keep Draft
until final performance review and upstream CI are complete.
