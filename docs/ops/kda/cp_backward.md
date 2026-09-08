# CP local backward fusion

This stacked change adds the opt-in Mosaic Config `fuse_cp_backward=False`.
With it enabled, `fuse_backward=True`, saved forward states, and K/V dimensions
that are multiples of 128, CP uses the existing fused local reverse kernel
**after** its state-gradient collective. With `rematerialize_for_backward=True`,
add `fuse_rematerialization=True` to enable the same local reverse fusion.
Without that additional opt-in, CP rematerialization retains its staged path.
Other widths also retain the staged path. Public inputs, residual contracts, and collective
ordering are unchanged.

## Boundaries

The CP preprocessing still needs materialized w/qg/kg and dv before communication.
The pre-collective attention-gradient kernel therefore returns only dv; dAqk is
recomputed inside the final reverse kernel. That reverse kernel also recomputes
w/qg/kg and consumes the original v_new. Saved-state mode retains its existing
v_new dtype; rematerialization retains the FP32 running-state v_new without
reconstructing it from rounded hidden-state snapshots. The change trades local
recomputation for fewer local HBM reads and removes the dAqk HBM intermediate.
It does not fuse collectives or eliminate all CP intermediates, and is not the
complete CP megakernel from the source implementation.

The migration source remains primatrix/pallas-kernel commit
`c82625a2bdadd5881b12f1ab37ec3fa2004c9780`. This incremental implementation reuses
Tokamax's preceding saved-state fusion rather than introducing a second CP
communication protocol.

## Rematerialization boundary

CP rematerialization still uses `_recompute_w_u_fwd` and
`chunk_gated_delta_rule_fwd_h` to rebuild its local states, seeded with the
forward residual's CP initial state. These outputs feed the existing CP
preprocessing. This change only extends post-collective local fusion and the
dv-only pre-collective call to that path. State reconstruction, its HBM buffers,
and all collective ordering remain unchanged. This does not enable the non-CP
fused state reconstruction kernel for CP.

## Validation

The CP equivalence test runs actual shard_map collectives on two devices and
compares all five input gradients against the staged CP path in BF16 and FP32.
It covers continuous sequences and boundaries within either context shard for
saved states, fused rematerialization, and rematerialization fallback. Tests
also verify which reverse kernel path is selected and that rematerialized
v_new stays FP32.
The dv-only test checks exact equality against the full attention/value-gradient
kernel. On CPU, Pallas interpretation is enabled; run with
`XLA_FLAGS=--xla_force_host_platform_device_count=2` to exercise CP tests.

Local validation: 14 CP/dv-only tests and 39 non-CP rematerialization/saved-state
backward tests passed. CI shard tests: 33 passed; shard consistency passed.
Eight TPU benchmark cases collected without timing.

TPU lowering, VMEM capacity and performance remain unverified. Keep the flag off
by default until those checks pass. `python -m tokamax.benchmarks.kda_cp` compares
staged and fused forward+VJP using the existing device timer on 2/4 TPU devices
with H=32, B=1, T=8192 and K=V=128. Record hardware and median times before
claiming a speedup. Benchmarks cover saved-state and rematerialized modes
separately. Fusing CP state reconstruction remains a separate follow-up.
