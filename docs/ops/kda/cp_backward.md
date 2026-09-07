# CP saved-state local backward fusion

This stacked change adds the opt-in Mosaic Config `fuse_cp_backward=False`.
With it enabled, `fuse_backward=True`, saved forward states, and K/V dimensions
that are multiples of 128, CP uses the existing fused local reverse kernel
**after** its state-gradient collective. CP rematerialization and other widths
retain the staged path. Public inputs, residual contracts, and collective
ordering are unchanged.

## Boundaries

The CP preprocessing still needs materialized w/qg/kg and dv before communication.
The pre-collective attention-gradient kernel therefore returns only dv; dAqk is
recomputed inside the final reverse kernel. That reverse kernel also recomputes
w/qg/kg and consumes the original saved-path v_new. The change trades local
recomputation for fewer local HBM reads and removes the dAqk HBM intermediate.
It does not fuse collectives or eliminate all CP intermediates, and is not the
complete CP megakernel from the source implementation.

The migration source remains primatrix/pallas-kernel commit
`c82625a2bdadd5881b12f1ab37ec3fa2004c9780`. This incremental implementation reuses
Tokamax's preceding saved-state fusion rather than introducing a second CP
communication protocol.

## Validation

The CP equivalence test runs actual shard_map collectives on two devices and
compares all five input gradients against the staged CP path in BF16 and FP32.
It covers continuous sequences and boundaries within either context shard.
The dv-only test checks exact equality against the full attention/value-gradient
kernel. On CPU, Pallas interpretation is enabled; run with
`XLA_FLAGS=--xla_force_host_platform_device_count=2` to exercise CP tests.

Local results: 72 tests and 9 subtests passed across the CP, dv-only, saved-state
backward, base and Mosaic adapter suites. CI shard tests: 33 passed; shard
consistency passed; four TPU benchmark cases collected without timing.

TPU lowering, VMEM capacity and performance remain unverified. Keep the flag off
by default until those checks pass. `python -m tokamax.benchmarks.kda_cp` compares
staged and fused forward+VJP using the existing device timer on 2/4 TPU devices
with H=32, B=1, T=8192 and K=V=128. Record hardware and median times before
claiming a speedup. CP rematerialization fusion remains a separate follow-up.
