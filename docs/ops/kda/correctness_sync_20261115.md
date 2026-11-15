# Forward correctness synchronization (2026-11-15)

Internal main is pinned at `18ea6891`; see the complete migration ledger and source manifest.

Native TPU7x, four chips/eight logical devices; JAX/jaxlib 0.12.0, libtpu 0.0.50.1,
large second-minor layout enabled, no interpretation.

- Default/fused/packed/padding/compact-forward suite: 66 passed (213.62 s).
- Final original-position native followup: 24 forward cases plus one deterministic
  scratch-poison case passed within the shared 58-case native suite (230.02 s).
  The companion backward cases in that suite belong to #1689 and are not a forward
  performance claim. New leading/interior padding cases retain the existing
  2% relative-RMS and elementwise bounds; invalid outputs remain exactly zero.
- Common padding compatibility: 16 CPU Pallas cases passed on JAX 0.11.1.
- Exact capacity and short-convolution contract checks: 2 CPU tests passed.

Reproduce the forward followup with:
```
python -m pytest -q -x tokamax/_src/ops/kda/pallas_mosaic_tpu_compact_forward_test.py tokamax/_src/ops/kda/native_padding_carry_test.py tokamax/_src/ops/kda/packing_test.py
```

Grouped native records retain token positions; resumed/interleaved labels use the
stable grouping fallback. The first all-padding chunk now initializes recurrent
carry, all-padding saved h is defined, and nonexistent segment 0 is not saved as a
final state. The scratch-poison control reproduces an existing issue in immutable
internal main (16,384 nonfinite outputs) while the corrected core has zero.
This is a scratch-safety control, not a production-failure reproduction claim.

No reference arithmetic or numerical tolerances were weakened. The BF16 unused-state
analytic check is exact equality to its dtype-matched cotangent. Public CP logical
capacity is preserved; only the workspace adapter narrows physical capacity.
One/two-token convolution uses only existing history. Formatting has unchanged ASTs.

Compact backward, CP and standalone inference remain in their corresponding PRs.
Historical performance tables predate these correctness commits. Keep Draft pending
complete upstream CI and the final performance review.
