# Complete official ArgSpec comparison after correctness synchronization

Follow-up: [single-core packing refactor and complete rerun](packing_single_call_validation_20261116.md). The measurements below retain their historical source revisions.

Status: Ready for Review authorized with the two measured Tokamax default-path regressions disclosed as follow-up work. Current MaxText KDA training uses TOPS; the benchmark results and numerical checks are unchanged.

## Revisions and scope

- Baseline: latest upstream main `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18`, verified again before execution.
- PR calculation snapshot: `f20013174fb54b7e7dd7d7093b6d383b2e457f33`; measured integration merge: `d155f97ae8cc25d9470f46dc4b6ebe5739680948`.
- Later raw-tape fixture changes modify tests/documentation only. The measured migrated Mosaic implementation files remain byte-identical to the final PR calculation source.
- Main integration adds six official decode ArgSpecs and the short-history reference-convolution fix. The measured merge contains those latest definitions.
- All 15 official ArgSpecs are included: 15 forward points plus three `forward_and_vjp` points, following the 12 `forward_only` tags. These are the official single-device API regression profiles; CP/workspace measurements and Draft #2009 are separate workloads.

## Reproducible method

- Dedicated TPU7x allocation: four physical chips/eight JAX devices; each benchmark runs on logical device 0. No other test process runs on that allocation during sampling/capture.
- Python `3.12.14`; jax 0.12.0, jaxlib 0.12.0, libtpu 0.0.50.1, numpy 2.5.4, flax 0.12.12, qwix 0.1.10, xprof 2.24.2.
- Observed `LIBTPU_INIT_ARGS`: `--xla_tpu_impure_enable_large_2nd_minor_layout=true --xla_tpu_use_enhanced_launch_barrier=true`. `XLA_FLAGS` and `PALLAS_INTERPRET` are unset. JAX cloud TPU initialization adds the enhanced launch-barrier flag.
- Six fresh worker processes: two sweeps in opposite variant order (`main-default`, `merge-default`, `merge-opt`; then reversed). Per cell: 10 warmups, 30 synchronized `block_until_ready` wall samples per sweep, 60 total. All 3,240 measured wall samples are retained.
- Inputs are serialized once from official ArgSpec shapes/static options using seed 619; beta is fixed to 0.5 consistently across variants. Full array shapes, dtypes, options and logical-input SHA256 hashes are in `manifest.json`.
- `tokamax.standardize_function` supplies the official forward/VJP mode. `forward_and_vjp` times complete public forward plus its pullback with output-derived cotangent. Compilation, input creation and initial placement are outside timing.
- Before warming up, inputs are placed in each executable's compiled `input_formats` and synchronized. All 108 device captures contain zero `jit__multi_slice` input-reshard modules.
- Device time is the complete `jit_invoke` XLA-module event elapsed time on `/device:TPU:0`, including in-module metadata, copies, normalization and reverse stages. Two independent captures per point, five synchronized invocations each: 108 raw XPlanes/540 module events. The table reports the median of ten module-event durations.
- All six worker environments match. All 54 profiling endpoints have the same input hash, input Formats and output hash as their corresponding wall-sweep endpoint. Both wall sweeps reproduce endpoint outputs bitwise.
- The first parallel CLI parse returned a nonzero status. The same immutable 108 captures were successfully reparsed with two workers and CPU thread limits; all captures remain included. The failed command and successful parser are retained.

## Shapes and static options

Layouts: Q/K `[H,B,T,128]`, V `[H,B,T,128]`, gate `[H,B,T,GW]`, beta `[H,B,T]`; B=1 throughout. Tokens/gate/beta use BF16; gate parameters and any initial state use FP32. Q/K normalization is enabled for all cases. Bounded raw gate with lower bound -5 is enabled except the scalar-gate GDN case. All stateful inference cases use convolution width 4 and return final state.

| ID | Official ArgSpec | H | T | Logical N | GW | State / conv | Modes |
|---|---|---:|---:|---:|---:|---|---|
| 00 | `fixed_training_b1_t8192_bf16` | 32 | 8192 | fixed | 128 | none / none | F, F+VJP |
| 01 | `packed_training_b1_t8192_n25_bf16` | 32 | 8192 | 25 | 128 | none / none | F, F+VJP |
| 02 | `gdn_fixed_training_b1_t8192_bf16` | 32 | 8192 | fixed | 1 | none / none | F, F+VJP |
| 03 | `tp32_prefill_b1_t128_bf16` | 3 | 128 | 1 | 128 | FP32 / W4 | F |
| 04 | `tp32_prefill_b1_t512_bf16` | 3 | 512 | 1 | 128 | FP32 / W4 | F |
| 05 | `tp32_prefill_b1_t1024_bf16` | 3 | 1024 | 1 | 128 | FP32 / W4 | F |
| 06 | `tp32_verify_b1_t64_n8_bf16` | 3 | 64 | 8 | 128 | FP32 / W4 | F |
| 07 | `tp16_prefill_b1_t1024_bf16` | 6 | 1024 | 1 | 128 | FP32 / W4 | F |
| 08 | `tp8_prefill_b1_t2048_bf16` | 12 | 2048 | 1 | 128 | FP32 / W4 | F |
| 09 | `tp32_decode_b1_t1_n1_bf16` | 3 | 1 | 1 | 128 | FP32 / W4 | F |
| 10 | `tp32_dspark_decode_b1_t8_n1_bf16` | 3 | 8 | 1 | 128 | FP32 / W4 | F |
| 11 | `tp32_dspark_decode_b1_t64_n16_bf16` | 3 | 64 | 16 | 128 | FP32 / W4 | F |
| 12 | `tp32_decode_b1_t64_n64_bf16` | 3 | 64 | 64 | 128 | FP32 / W4 | F |
| 13 | `tp16_dp2_dspark_decode_b1_t64_n16_bf16` | 6 | 64 | 16 | 128 | FP32 / W4 | F |
| 14 | `tp16_dp2_decode_b1_t64_n64_bf16` | 6 | 64 | 64 | 128 | FP32 / W4 | F |

## Synchronized wall time (ms)

Default uses `Config=None`. Opt-in uses `fuse_forward`, `packed_forward`, `packed_output`, `fuse_backward`, `packed_backward`, `packed_gradients`, `fuse_cp_backward` enabled. The CP selector is dormant on these single-device cases. This opt-in profile measures fused/packed execution, with Q/K normalization enabled; it is distinct from historical compact/no-normalization benchmarks.

| Case / mode | main | Default | Change | Opt-in | Change |
|---|---:|---:|---:|---:|---:|
| 00 F | 1.701955 | 1.710141 | +0.48% | 1.638992 | -3.70% |
| 00 F+VJP | 3.992807 | 4.013056 | +0.51% | 3.706256 | -7.18% |
| 01 F | 4.005913 | 3.816489 | -4.73% | 2.765849 | -30.96% |
| 01 F+VJP | 7.617597 | 8.199981 | +7.65% | 7.671264 | +0.70% |
| 02 F | 1.701758 | 1.710764 | +0.53% | 1.686881 | -0.87% |
| 02 F+VJP | 3.360882 | 3.361707 | +0.02% | 3.334319 | -0.79% |
| 03 F | 0.216713 | 0.227739 | +5.09% | 0.218091 | +0.64% |
| 04 F | 0.234032 | 0.238502 | +1.91% | 0.226315 | -3.30% |
| 05 F | 0.273185 | 0.285392 | +4.47% | 0.250045 | -8.47% |
| 06 F | 0.451521 | 0.461461 | +2.20% | 0.464363 | +2.84% |
| 07 F | 0.290241 | 0.299035 | +3.03% | 0.276025 | -4.90% |
| 08 F | 0.413024 | 0.429806 | +4.06% | 0.419569 | +1.58% |
| 09 F | 0.247616 | 0.253907 | +2.54% | 0.210776 | -14.88% |
| 10 F | 0.257567 | 0.263954 | +2.48% | 0.250171 | -2.87% |
| 11 F | 0.632069 | 0.639170 | +1.12% | 0.620163 | -1.88% |
| 12 F | 1.569377 | 1.596671 | +1.74% | 1.596530 | +1.73% |
| 13 F | 0.642358 | 0.652255 | +1.54% | 0.628665 | -2.13% |
| 14 F | 1.655722 | 1.768533 | +6.81% | 1.684859 | +1.76% |

## Full device-module time (ms)

| Case / mode | main | Default | Change | Opt-in | Change |
|---|---:|---:|---:|---:|---:|
| 00 F | 1.507143 | 1.507171 | +0.00% | 1.462622 | -2.95% |
| 00 F+VJP | 3.726674 | 3.725150 | -0.04% | 3.442132 | -7.64% |
| 01 F | 3.828535 | 3.603132 | -5.89% | 2.583307 | -32.52% |
| 01 F+VJP | 7.221417 | 7.794879 | +7.94% | 7.299652 | +1.08% |
| 02 F | 1.520852 | 1.520983 | +0.01% | 1.519250 | -0.11% |
| 02 F+VJP | 3.093097 | 3.092489 | -0.02% | 3.092431 | -0.02% |
| 03 F | 0.017758 | 0.017673 | -0.47% | 0.017546 | -1.19% |
| 04 F | 0.034439 | 0.034502 | +0.18% | 0.034535 | +0.28% |
| 05 F | 0.055349 | 0.055237 | -0.20% | 0.055466 | +0.21% |
| 06 F | 0.220051 | 0.224870 | +2.19% | 0.226765 | +3.05% |
| 07 F | 0.069622 | 0.069866 | +0.35% | 0.069688 | +0.09% |
| 08 F | 0.185644 | 0.185651 | +0.00% | 0.185411 | -0.13% |
| 09 F | 0.029340 | 0.026538 | -9.55% | 0.027052 | -7.80% |
| 10 F | 0.043130 | 0.042944 | -0.43% | 0.044968 | +4.26% |
| 11 F | 0.381793 | 0.377703 | -1.07% | 0.376912 | -1.28% |
| 12 F | 1.307144 | 1.322723 | +1.19% | 1.351744 | +3.41% |
| 13 F | 0.396881 | 0.404741 | +1.98% | 0.395481 | -0.35% |
| 14 F | 1.405730 | 1.483216 | +5.51% | 1.455887 | +3.57% |

## Default-path review

The default path has unresolved performance regressions. The two largest material regressions are stable across reversed wall sweeps and are also present in full-module device time:

| Workload | Wall main -> default ms | Device main -> default ms | Compiler temporary GiB main -> default |
|---|---:|---:|---:|
| `packed_training_b1_t8192_n25_bf16` / forward_and_vjp | 7.617597 -> 8.199981 | 7.221417 -> 7.794879 | 2.424892 -> 3.117951 |
| `tp16_dp2_decode_b1_t64_n64_bf16` / forward | 1.655722 -> 1.768533 | 1.405730 -> 1.483216 | 0.103240 -> 0.180166 |

Packed full forward/VJP wall medians are 7.604596/7.624214 ms for main and 8.219743/8.175008 ms for default in the two sweeps. The compiler temporary-buffer increase is 744,167,072 bytes. H6/T64/N64 decode increases by 82,599,424 temporary bytes. These are compiler buffer-planning metrics; model/allocator peak memory is outside this measurement.

Smaller default movements remain in the full tables. Host changes for very short prefill kernels do not establish the same device-time change. Packed opt-in forward improves by 30.96% wall and 32.52% device, while its complete forward/VJP is +0.70% wall/+1.08% device versus main. No aggregate or universal speedup is claimed.

The correctness wrapper/conditional tape and added temporary buffers are investigation leads. A causal attribution or a final default-path performance fix is not established here. The author has authorized entering review while retaining this performance follow-up. Upstream maintainers can assess its merge impact; performance acceptance is not claimed.

## Numerical checks

- All measured-cell endpoint leaves are finite; outputs repeat bitwise across both fresh-process wall sweeps. All profile endpoints match their wall counterpart.
- Default is bitwise equal to main on 17/18 execution points. Packed complete forward/VJP has maximum leaf relative RMS `2.419694e-7` and maximum absolute difference `1.907349e-6`.
- Opt-in is bitwise equal on 16/18 points. Packed forward/full-VJP differ with maximum leaf relative RMS `7.173140e-5` and maximum absolute difference `0.0001220703125`.
- Per-leaf shapes/errors are retained in `output-comparison.json`. No reference or numerical tolerance changes were used to accept this sweep. These are operator-output/gradient checks, separate from model LM/MTP loss validation.

## MaxText consumer scope and review decision

The timing baseline is upstream OpenXLA/Tokamax main `0d3e82b4`, not internal
pallas-kernel main. The workloads exercise Tokamax's public Mosaic default API.
The packed full forward/VJP point is a training workload, so direct Tokamax
training consumers can encounter this regression.

Current MaxText main `fd1446d5a1b534f767e237be2a495b66d2411ae3` routes KDA training through
`maxtext.kernels.kda.pallas.pallas_chunk_kda` into TOPS:

- Ordinary/all-to-all route: `tops.ops.kda.chunk_kda`.
- Explicit CP workspace route: `tops.ops.kda.chunk_kda_cp`.
- Compact all-gather route: `tops.ops.kda.cp_compact.chunk_kda_cp_compact`.

[Current pinned consumer routing](https://github.com/primatrix/maxtext/blob/fd1446d5a1b534f767e237be2a495b66d2411ae3/src/maxtext/kernels/kda/pallas.py)
shows these branches. The training layer performs Q/K normalization outside the
kernel and calls that wrapper. Installing Tokamax as another MaxText dependency
does not route its KDA training through this upstream adapter.

Consequently these two measurements do not establish a slowdown in current MaxText
training and are not a current MaxText training-delivery blocker. At the author's
request #1690 may enter review with the numbers retained and performance optimization
deferred. This decision does not claim the generic Tokamax regression is fixed or
that the PR has merge approval.

## Evidence and reproduction

Data and exact harness are in `benchmarks/official_argspec_sync_20261115/`. Public script copies are whitespace-formatted with Python AST identity checked against the executed versions. The harness expects `/workspace/official-sync/main` and `/workspace/official-sync/merge` source snapshots; arrange those paths or consistently substitute a reproduction prefix.

Run `run.py`, `analyze.py`, `check_outputs.py`, `run_profiles.py`, then `analyze_profiles.py`. `generate.py` asserts 15 ArgSpecs. Raw synchronized samples, memory and input Formats are in `comparison.json`; device-event samples and capture checks are in `device-captures.json`.

Immutable raw evidence (internal GCS access) includes source snapshots, serialized inputs, HLO, XPlanes, successful/failed correctness experiments and CI logs:

`gs://data_ant/diagnostics/kda-full-sync-20261115/a447cbfd-6fa3-4092-979f-6e46172d9825/evidence-f2001317-main0d3e82b4.tgz#1794753076907365`

Bytes: `4102941674`; SHA256: `911c350331d50c34576d6590450798b25bc8b8470cf80e42ffa458323cf0368b`; CRC32C: `3oB8kQ==`. Streamed upload size, server CRC32C and stored SHA256 metadata were verified.
