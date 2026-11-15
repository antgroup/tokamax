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

## Final validation and same-input results

Same exclusive TPU7x Pod (`5a90e09f-b81d-4ec7-b0c7-f463a1d301a1`),
JAX/JAXLIB 0.12.0, libtpu 0.0.50.1, Python 3.12.15. Seed 619, BF16
activations, FP32 A_log/bias, K=V=128, seven operand gradients and the
identical cotangent. Every logical input hash and compiled physical input
Format matches across all three implementations. The source-level interface
transpose for the internal batch-first API is included in compilation.

Five warmups then 20 synchronized host samples in each of two reversed-order
fresh-process sweeps give **40 samples per cell, 840 timings**. All samples,
including process-to-process variation, are retained:

| Case | Before `1f4d897e` ms | Published `49007216` ms | Change | Internal `18ea6891` ms |
|---|---:|---:|---:|---:|
| cp4_normal | 0.890128 | 0.899660 | +1.07% | 0.787774 |
| cp8_normal | 1.408825 | 1.395174 | -0.97% | 1.240048 |
| cp4_remat | 0.903992 | 0.900811 | -0.35% | 0.807858 |
| cp8_long | 3.041740 | 2.990976 | -1.67% | 2.135919 |
| cp8_long_remat | 3.191619 | 3.091252 | -3.14% | 2.541995 |
| cp8_capacity | 1.347046 | 1.326178 | -1.55% | 1.153929 |
| cp8_overflow | 23.652161 | 21.279632 | -10.03% | 354.296334 |

The +1.07% CP4 fresh-process difference is retained above; it is not silently
rounded to a pass. To distinguish kernel behavior from process/backend
variation at this ~0.9 ms scale, all three executables were compiled in one
process and invoked on the **same actual prepared input tuple**. After 20
warmups per implementation, 100 randomized palindromic blocks retained
**200 calls per implementation**. Medians: before **0.887107 ms**,
published **0.876889 ms**, internal **0.780830 ms**.
The paired mean published-minus-before difference is
**-12.589 microseconds**, with a paired-block
bootstrap 95% interval of **[-17.473,
-8.375] microseconds**. No outlier was removed.
This additional control supports a small improvement, rather than the
previously inferred 5.8% kernel regression. It does not erase the separate
fresh-process or historical transport-inclusive scenarios.

Independent resident-input 20-call XPlane captures, using each call's slowest
CP-rank target-module duration, give:

| Case | Before device module us | Published device module us | Internal device module us |
|---|---:|---:|---:|
| CP4 normal | 450.348 | 438.373 | 330.996 |
| CP8 8K | 2455.269 | 2337.455 | 1640.893 |

All six corrected captures contain **zero** `jit__multi_slice` events on the
checked rank, compared with 140 slice events for 20 calls in the original
CP4 capture. Raw event output from every participating rank, source captures,
flags and exact XProf 2.24.2 commands are archived. Profiles enable LLO debug
registration and custom-call region tracing consistently for the controls;
they are independent evidence, not replacements for the unprofiled host samples.
No low-level spill/roofline claim is inferred from these module timings.

## Ablations and decisions

The screening campaign retained 40 transport-inclusive samples for each of
seven variants across the original three regression cases. Variants moved
normalization outside M4 with FP32 Q/K gradient stores, restored the M4
head-batch cap to 16, combined those changes, or replaced the inactive-chunk
conditional with a compute-and-mask diagnostic. None was adopted. The
compute-and-mask diagnostic gave no gain and is unsuitable as a production
fix for undefined inactive states. Moving normalization outside increased
HBM traffic and its captured full-module timing was worse than the published
implementation. A head cap of 16 is a structural no-op at H=8 yet the host
medians changed between fresh processes, reinforcing the need for paired and
device controls. The measurements do not justify rolling back canonical
arithmetic, inactive-chunk safety, or the normalization precision fix.

The original three regression inputs all fit the capacity-2 **all-heads fast
branch**; they do not execute the grouped fallback. Internal main uses its
M5 core with outer normalization, while #1690 uses shared M4 arithmetic.
Internal normal cases are faster here, but its large-capacity overflow
fallback is much slower (354.30 ms versus 21.28 ms). This is a different
implementation/workspace cost, not proof of a historical internal regression
or a universal ranking across workloads.

## Numerical policy and publication

All **21 variant/case combinations, 168 complete-array leaf comparisons**
pass against the generation-pinned independent FP32 recurrence from the
previous run, after matching every input/cotangent hash. Output plus all
seven gradients are checked, with exact shape/dtype and finite requirements.
Internal CP token/output rules use atol=rtol=0.05, A_log 0.05/0.05, and bias
**0.008/0.008**, with the existing one-BF16-ULP fallback. RMS diagnostics are
retained but no extra 5% gate is introduced. No coordinate is removed.

The calculation files under `tokamax/_src/ops/kda/` remain byte-identical to
`49007216`. This change fixes benchmark preparation and the interpretation of
its results; it does not claim a new kernel optimization. The old measurements
are retained under their transport-inclusive scope. Both PRs remain Draft at
the author's request. Full JSON, all 1,440 new resident/paired host samples,
raw traces, ablations and source snapshots are archived under
`gs://data_ant/diagnostics/kda-cp-ablation-20261115/5a90e09f/`.
