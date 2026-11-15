# Workspace input-placement investigation (2026-11-15)

The November 15 comparison of `1f4d897e` with `49007216` reported a
2.6–5.8% workspace host-time regression. That harness placed every input on
logical device 0, then repeatedly called a CP executable. The historical
transport-inclusive measurements remain valid for that caller scenario,
but they do not establish a resident-input kernel regression.

## Direct evidence and correction

Official XProf 2.24.2 raw XPlane queries show seven `jit__multi_slice`
modules before each target call in the affected CP4 trace: 140 slice
modules for 20 calls on rank 0. Each input shard operation and its host
dispatch were included in the host stopwatch. The actual target-module
20-call median, using the maximum duration across CP ranks per call,
is 503.629 -> 473.806 microseconds for CP4 and 2733.038 -> 2600.293
microseconds for the 8K CP8 case. This does not support a device-module
regression at the published operator head.

The corrected harness uses `compiled.input_formats[0]` to prepare each
input with `jax.device_put` once and synchronizes this preparation before
warmup/timing. This keeps numerical inputs, shapes, dtypes, cotangents,
CP partitioning and operator flags unchanged. It excludes input-layout and
resharding work from the resident-input scenario. It does not claim to
speed up the old transport-inclusive scenario or whole-model training.

Internal Pallas main `18ea6891` is an independent same-input control,
not the before/after baseline. The before baseline remains #1690
`1f4d897e`; the published operator is `49007216`. Internal workspace tests
also place inputs onto NamedSharding before their timing loop. Internal
M5 and upstream M4 are different implementations, so conclusions are
case-specific and numerical outputs/gradients must also be checked.

## Reproduction

Use the exact source archives and manifest named in the final evidence:
`/workspace/before`, `/workspace/candidate`, `/workspace/internal`, and
`/workspace/revisions.json`. Pin JAX/JAXLIB 0.12.0 and libtpu 0.0.50.1.
Place the benchmark script and `cp_workspace_matrix_20261115.py` together.
For each variant (`before_workspace`, `candidate_workspace`,
`internal_workspace`), run:

```sh
python cp_workspace_resident_20261115.py --suite cp \
  --variant candidate_workspace --round 0 --output-root /workspace/prepared
```

Run a second fresh-process sweep in reverse variant order. Each case uses
five warmups and 20 synchronized samples per sweep. Keep every sample.
`--include-input-reshard` reproduces the historical caller scenario;
`--profile-only --round 10` records an independent 20-call trace.
The `cp4_mega` case is selected only with the explicit mega variant.

Final resident-input results, raw trace checks and complete-array numerical
validation are recorded alongside this document once the paired experiment
finishes. Both PRs remain Draft; no operator arithmetic or threshold is
changed by this benchmark correction.
