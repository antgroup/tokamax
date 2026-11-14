# KDA backward performance and numerical follow-up (2026-11-15)

Frozen upstream main: `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18`.
Starting PR head: `ac0cae118ae59d4a90f9fef1b3b4cf365c69cbc3`.
The candidate is the operator patch in this PR; the companion source manifest
pins the published files and verifies that subsequent description-only edits
leave the measured non-doc AST unchanged. The PR remains **Draft** per author
instruction; a green CI matrix alone does not change its review state.

## Full operator plus VJP and runtime-residual VJP

Times are synchronized host-wall medians, in milliseconds. Negative differences
mean less elapsed time. Every cell uses all 40 samples: five warmups, 20 calls
per fresh process, two sweeps in reversed variant order. Compilation and input
creation are excluded. The default and same-head forward-only controls are
preserved in the raw data, including their unfavorable measurements.

| Case | Main full | Candidate full | Difference | Main VJP | Candidate VJP | Difference |
|---|---:|---:|---:|---:|---:|---:|
| arg_fixed | 3.927215 | 3.618995 | -7.85% | 2.489574 | 2.215297 | -11.02% |
| arg_packed | 7.391404 | 6.699759 | -9.36% | 4.064177 | 4.031045 | -0.82% |
| remat_packed | 9.067555 | 8.150345 | -10.12% | 5.574706 | 5.388874 | -3.33% |

The candidate default full-call control ranged from -0.92% to +0.53% versus
main; VJP ranged from -0.77% to +0.21%. Small differences at this scale should
not be interpreted as universal speedups. The earlier same-environment audit
of `ac0cae11` found full-call regressions of 7.76%, 16.39% and 8.87% for these
three optimized rows, and 3.73%/4.12% for default packed/remat. Those results
and failed experiments remain archived; the October 27 tables are historical.

This is a component benchmark, not a measured training step-time improvement.
Pure VJP receives the prepared pullback as a runtime pytree argument; its
largest closed constant is checked to remain below 1 MiB. No residual arrays
are baked into HLO constants. Compiler temporary-byte counts are in the JSON;
they are not allocator/model peak HBM measurements.

## Changes

- The local reverse body and normalization share canonical staged arithmetic.
  Q/K normalization finishes in VMEM before activation-dtype q/k/v gradient
  stores; FP32 gate reductions keep their existing precision.
- Scalar beta/db use token-minor tiles in both fused passes. Cotangents retain
  activation dtype in HBM, promoting in the kernels. Head-group caps follow
  the existing hardware VMEM policy, not the model batch size.
- Packed forward normalizes raw Q/K inside its mutable VMEM input tiles and
  writes unit Q/K and RSTD to the existing aligned residual fields. Backward
  reuses those fields; normalized Q/K and beta do not need packed-window
  replay. Unnormalized Q/K and packed V retain their window loaders. The
  public residual schema, epsilon, data order, and model configuration stay
  unchanged.
- Parallel Pallas WY reconstruction reuses the canonical state recurrence,
  including FP32 varlen state snapshots/value corrections. The previous
  producer-inside-state loop is removed. No independent reference budget is
  enlarged to accept it.
- Inactive backward chunks skip their matmuls and write zero VMEM output
  tiles. This replaces complete q/k/v/db clear scans. Per-token gate masks
  and convolution-specific protections remain.

## Numerical validation and limits

The independent reference is the unchanged token recurrence in FP32 with
highest matmul precision; per-head checkpointing only bounds its memory.
All eight output/gradient leaves and all coordinates participate in the
comparisons. No first token, spike, or unfavorable metric is removed.

For rematerialized packed execution, dQ relative RMS falls from about 3.31%
to **0.4090%**, dGate from about 4.59% to **1.6362%**, and A_log from about
10.13% to **4.9823%**, returning to the staged reference level. The bias
relative RMS remains **5.2036%**, comparable to main's existing 5.2038% floor
and still above the independent 5% parameter budget. It is explicitly retained
as an inherited reference failure; this report does not claim every leaf
passes that independent budget.

The author-authorized BF16 pairwise rule remains atol=0.005, rtol=0.05 with
one BF16 ULP. FP32, exact analytic tests, finite checks, and independent 2%
input/5% parameter RMS budgets are unchanged. Inactive intermediate slots of
the canonical state launcher are undefined; live snapshots are compared, and
a separate poisoned-state test requires finite, exactly zero inactive final
gradient outputs.

The final-source dedicated TPU checks passed 6 guard/offset/state cases and
16 saved-state cases (overlapping broader runs). The packed producer and
retained-residual tests also passed native TPU and CPU checks. The published
head's complete H100/TPU6e/TPU7x results are tracked on
[#1689 checks](https://github.com/openxla/tokamax/pull/1689/checks).

## Environment and reproduction

Dedicated TPU7x four-chip/eight-logical-device Pod; the timed operator uses
device 0. B1/T8192/H32/K=V128, chunk64, BF16 q/k/v/g/beta/dO; raw bounded
gate lower=-5, FP32 A_log/bias, Q/K normalization, no initial/final state.
Packed rows contain 25 segments; the fixed row has one sequence. The full
call returns output plus seven operand/parameter gradients. NumPy seed619,
identical input hashes across every variant and round, no donation, and
large-second-minor layout enabled. No host-side analysis or compression
runs overlap timed samples.

JAX 0.12.0, JAXLIB 0.12.0, libtpu 0.0.50.1; 3.12.16 (main, Oct 28 2026, 04:34:44) [GCC 14.2.0].

[Raw samples, configs, hashes and memory JSON](benchmarks/backward_performance_followup_20261115.json),
[CSV](benchmarks/backward_performance_followup_20261115.csv),
[reference errors](benchmarks/backward_reference_followup_20261115.json), and
[source manifest](benchmarks/backward_source_followup_20261115.json).

Source snapshots, raw tensors, HLO, logs and immutable GCS generations are
archived under `gs://data_ant/diagnostics/kda-bwd-perf-20261115/2d528bf8/`;
the verified archive manifests and harness are also retained in the local
`D:/dev/kda-bwd-perf-1115` evidence directory. Earlier rejected raw archives
have generation-pinned URIs with server CRC32C and SHA256 metadata.
