# KDA backward: pinned-source review validation

## Provenance and audit correction

This is a pinned historical measurement record, not live PR status or a new
measurement of an arbitrary future main/head. The audit checked table values
against the retained JSON/CSV, units, shape/static options and baseline scope.
This correction changes explanations and labels only; measured values, raw
samples and numerical thresholds are unchanged. All evidence remains on this
separate branch rather than in the implementation PR.

The restored KDA implementation/tests at `ad9153f73c3255340c8295718d202f3bdb5a1385` match the original report snapshot `921c3554c129577b22c5ed00482cf8dc558ca4ec`. Changes to those files in future require renewed validation.

PR #1689. Tested calculation commit `5adb7a493e918ba27e6180bb9d274a6f0ac3fcfb`; baseline upstream main `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18`. Measurements were collected on 2026-11-16 (Asia/Shanghai).

## Scope

Optimize non-CP saved-state backward, packed/compact gradients and fused recomputation. Consume the original-token forward tape from #1688. Keep initial-state and primal-dtype cotangent contracts. The final M5 fix retains FP32 intra gate cotangents for A_log/bias reductions while preserving the prior token-gradient rounding.

## Correctness and source acceptance

The 149-case native backward/packing suite passed. The expanded 46-case compact suite passed on the fixed kernel (overlapping earlier coverage), then the direct eight-head long-packed regression passed. The old implementation fails that same regression. Existing norm/elementwise thresholds are unchanged.

All prior reference modules and numerical thresholds are unchanged. Default formal endpoints are bitwise equal to main at 15/18 execution points; short multi-state convolution differences have maximum per-leaf relative RMS below 1.4e-5. Explicit fused/packed formal paths retain their original BF16 rounding. Both reversed sweeps reproduce endpoint hashes.

## Formal operator coverage and method

- All 15 official ArgSpecs, comprising 15 forward and three full forward/VJP execution points. Serialized primals are reused byte-for-byte; seed 619, beta explicitly overridden to the legal constant 0.5, with logical-input hashes in the manifest.
- `tokamax.standardize_function(..., seed=None)` supplies the modes. Its full forward/VJP uses the output itself as cotangent, behind an optimization barrier. Cotangents are consequently output-derived; this is not a fixed external-dY experiment.
- Dedicated TPU7x allocation: four physical chips / eight JAX devices; single-device timing on logical device0. JAX/jaxlib 0.12.0, libtpu 0.0.50.1. Complete observed versions and flags are in environments.json.
- Two fresh-process sweeps reverse implementation order. Inputs are prepared once using compiled input Formats. Ten warmups precede thirty blocking host-wall samples per cell per sweep; the table uses the median of all sixty samples.
- Each formal execution point has two independent five-call XPlane captures. Device time is the complete jit_invoke module, including metadata, copies and all forward/reverse stages. Every capture has zero input-reshard modules.

### Shapes

Layout is [H,B,T,K/V]; B=1 and K=V=128. Tokens/gates/beta use BF16; gate parameters and initial/final states use FP32. All formal shapes enable Q/K normalization. Stateful cases include width-4 convolution. Unsegmented prefill has one FP32 state slot (N=1), with segment_ids absent and max_num_segments unset; fixed training has no explicit segment-ID table.

| Official case | H | T | N | Gate width | State | Modes |
|---|---:|---:|---:|---:|---|---|
| fixed_training_b1_t8192_bf16 | 32 | 8192 | fixed | 128 | none | F, F+VJP |
| packed_training_b1_t8192_n25_bf16 | 32 | 8192 | 25 | 128 | none | F, F+VJP |
| gdn_fixed_training_b1_t8192_bf16 | 32 | 8192 | fixed | 1 | none | F, F+VJP |
| tp32_prefill_b1_t128_bf16 | 3 | 128 | 1 | 128 | FP32 | F |
| tp32_prefill_b1_t512_bf16 | 3 | 512 | 1 | 128 | FP32 | F |
| tp32_prefill_b1_t1024_bf16 | 3 | 1024 | 1 | 128 | FP32 | F |
| tp32_verify_b1_t64_n8_bf16 | 3 | 64 | 8 | 128 | FP32 | F |
| tp16_prefill_b1_t1024_bf16 | 6 | 1024 | 1 | 128 | FP32 | F |
| tp8_prefill_b1_t2048_bf16 | 12 | 2048 | 1 | 128 | FP32 | F |
| tp32_decode_b1_t1_n1_bf16 | 3 | 1 | 1 | 128 | FP32 | F |
| tp32_dspark_decode_b1_t8_n1_bf16 | 3 | 8 | 1 | 128 | FP32 | F |
| tp32_dspark_decode_b1_t64_n16_bf16 | 3 | 64 | 16 | 128 | FP32 | F |
| tp32_decode_b1_t64_n64_bf16 | 3 | 64 | 64 | 128 | FP32 | F |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 | 6 | 64 | 16 | 128 | FP32 | F |
| tp16_dp2_decode_b1_t64_n64_bf16 | 6 | 64 | 64 | 128 | FP32 | F |

### Complete formal wall/device matrix

All wall/device values in both performance tables are milliseconds. Change = (PR wall / main wall - 1) * 100%; positive means longer latency. Speedup = main wall / PR wall. Each row explicitly identifies the default or opt-in path; no universal speedup is claimed. Compiler temporary bytes are provided in the CSV/JSON and are not model/allocator peak memory.

| Case / mode | Path | Main wall | PR wall | Change | Main device | PR device |
|---|---|---:|---:|---:|---:|---:|
| fixed_training_b1_t8192_bf16 / forward | backward-default | 1.696256 | 1.697979 | +0.10% | 1.506584 | 1.506717 |
| fixed_training_b1_t8192_bf16 / forward | backward-opt | 1.696256 | 1.655380 | -2.41% | 1.506584 | 1.462061 |
| fixed_training_b1_t8192_bf16 / forward_and_vjp | backward-default | 4.019903 | 4.016398 | -0.09% | 3.725200 | 3.724296 |
| fixed_training_b1_t8192_bf16 / forward_and_vjp | backward-opt | 4.019903 | 3.751553 | -6.68% | 3.725200 | 3.442387 |
| packed_training_b1_t8192_n25_bf16 / forward | backward-default | 4.039549 | 3.807035 | -5.76% | 3.825856 | 3.605230 |
| packed_training_b1_t8192_n25_bf16 / forward | backward-opt | 4.039549 | 2.784619 | -31.07% | 3.825856 | 2.584397 |
| packed_training_b1_t8192_n25_bf16 / forward_and_vjp | backward-default | 7.615922 | 7.731422 | +1.52% | 7.215888 | 7.290010 |
| packed_training_b1_t8192_n25_bf16 / forward_and_vjp | backward-opt | 7.615922 | 7.696257 | +1.05% | 7.215888 | 7.297046 |
| gdn_fixed_training_b1_t8192_bf16 / forward | backward-default | 1.729614 | 1.709042 | -1.19% | 1.520061 | 1.518088 |
| gdn_fixed_training_b1_t8192_bf16 / forward | backward-opt | 1.729614 | 1.711458 | -1.05% | 1.520061 | 1.519385 |
| gdn_fixed_training_b1_t8192_bf16 / forward_and_vjp | backward-default | 3.371878 | 3.364320 | -0.22% | 3.093274 | 3.092064 |
| gdn_fixed_training_b1_t8192_bf16 / forward_and_vjp | backward-opt | 3.371878 | 3.367868 | -0.12% | 3.093274 | 3.091427 |
| tp32_prefill_b1_t128_bf16 / forward | backward-default | 0.231913 | 0.239107 | +3.10% | 0.017728 | 0.017677 |
| tp32_prefill_b1_t128_bf16 / forward | backward-opt | 0.231913 | 0.223620 | -3.58% | 0.017728 | 0.017749 |
| tp32_prefill_b1_t512_bf16 / forward | backward-default | 0.245647 | 0.250045 | +1.79% | 0.034594 | 0.034609 |
| tp32_prefill_b1_t512_bf16 / forward | backward-opt | 0.245647 | 0.241999 | -1.48% | 0.034594 | 0.034816 |
| tp32_prefill_b1_t1024_bf16 / forward | backward-default | 0.292392 | 0.272250 | -6.89% | 0.055379 | 0.055435 |
| tp32_prefill_b1_t1024_bf16 / forward | backward-opt | 0.292392 | 0.282290 | -3.45% | 0.055379 | 0.055407 |
| tp32_verify_b1_t64_n8_bf16 / forward | backward-default | 0.477208 | 0.402164 | -15.73% | 0.219741 | 0.160199 |
| tp32_verify_b1_t64_n8_bf16 / forward | backward-opt | 0.477208 | 0.472757 | -0.93% | 0.219741 | 0.226783 |
| tp16_prefill_b1_t1024_bf16 / forward | backward-default | 0.291921 | 0.295505 | +1.23% | 0.069692 | 0.069786 |
| tp16_prefill_b1_t1024_bf16 / forward | backward-opt | 0.291921 | 0.298492 | +2.25% | 0.069692 | 0.069948 |
| tp8_prefill_b1_t2048_bf16 / forward | backward-default | 0.428630 | 0.427886 | -0.17% | 0.185579 | 0.185463 |
| tp8_prefill_b1_t2048_bf16 / forward | backward-opt | 0.428630 | 0.428007 | -0.15% | 0.185579 | 0.185721 |
| tp32_decode_b1_t1_n1_bf16 / forward | backward-default | 0.255163 | 0.243762 | -4.47% | 0.029384 | 0.026507 |
| tp32_decode_b1_t1_n1_bf16 / forward | backward-opt | 0.255163 | 0.244032 | -4.36% | 0.029384 | 0.027286 |
| tp32_dspark_decode_b1_t8_n1_bf16 / forward | backward-default | 0.259823 | 0.257969 | -0.71% | 0.043058 | 0.043009 |
| tp32_dspark_decode_b1_t8_n1_bf16 / forward | backward-opt | 0.259823 | 0.269979 | +3.91% | 0.043058 | 0.045076 |
| tp32_dspark_decode_b1_t64_n16_bf16 / forward | backward-default | 0.645495 | 0.522173 | -19.10% | 0.381563 | 0.277816 |
| tp32_dspark_decode_b1_t64_n16_bf16 / forward | backward-opt | 0.645495 | 0.625640 | -3.08% | 0.381563 | 0.377469 |
| tp32_decode_b1_t64_n64_bf16 / forward | backward-default | 1.563943 | 1.256189 | -19.68% | 1.304905 | 0.997227 |
| tp32_decode_b1_t64_n64_bf16 / forward | backward-opt | 1.563943 | 1.613322 | +3.16% | 1.304905 | 1.352801 |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 / forward | backward-default | 0.655948 | 0.550248 | -16.11% | 0.396683 | 0.294901 |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 / forward | backward-opt | 0.655948 | 0.642161 | -2.10% | 0.396683 | 0.395256 |
| tp16_dp2_decode_b1_t64_n64_bf16 / forward | backward-default | 1.669431 | 1.320934 | -20.88% | 1.412362 | 1.057489 |
| tp16_dp2_decode_b1_t64_n64_bf16 / forward | backward-opt | 1.669431 | 1.713648 | +2.65% | 1.412362 | 1.449982 |

Formal opt-in flags select fused/packed forward and backward; compact feature rows additionally enable `compact_forward=True` and `compact_sorted_segments=True`. Rematerialization is enabled only in its named profile.

## Feature-specific comparison

These additional profiles use identical Gaussian inputs and a fixed external Gaussian cotangent (where applicable), seed 619. BF16 Q/K/V have std0.05, raw gate std0.1, beta is sigmoid of Gaussian logits with std0.1; A_log is uniform[-1,0] and bias uniform[-4,-3]. Lower bound is -5, scale is 128**-0.5. Both implementations use matching Q/K normalization options. Norm=False means normalization is disabled; the Gaussian Q/K inputs were not separately L2-normalized. Compact and optimized are different flag configurations, not interchangeable speedup labels. Two reversed fresh-process sweeps retain sixty host-wall samples per cell. Five-call complete-device captures in each sweep contain zero input-reshard modules.

Full means forward plus seven-gradient VJP, using a fixed external dO. Pure VJP prepares residuals before timing and supplies them as runtime inputs. The harness rejects a largest closed-over constant of 1MiB or more. Full forward+VJP includes preparation and its seven operand/parameter gradients.

| Case / mode | B/T/H/N | Norm | Path | Main wall | PR wall | Speedup | Main device | PR device |
|---|---|---|---|---:|---:|---:|---:|---:|
| compact_packed / full | 1/8192/32/25 | False | compact | 6.955280 | 3.594767 | 1.93x | 6.672754 | 3.208616 |
| compact_packed / vjp | 1/8192/32/25 | False | compact | 3.981377 | 2.123952 | 1.87x | 3.702135 | 1.823290 |
| fragmented / full | 1/2048/8/512 | False | compact | 19.709358 | 0.674943 | 29.20x | 18.605825 | 0.302973 |
| fragmented / vjp | 1/2048/8/512 | False | compact | 8.816300 | 0.443727 | 19.87x | 8.503744 | 0.175545 |
| compact_packed / full | 1/8192/32/25 | False | optimized | 6.955280 | 8.201833 | 0.85x | 6.672754 | 7.907774 |
| compact_packed / vjp | 1/8192/32/25 | False | optimized | 3.981377 | 4.540866 | 0.88x | 3.702135 | 4.209782 |
| fragmented / full | 1/2048/8/512 | False | optimized | 19.709358 | 24.110770 | 0.82x | 18.605825 | 22.837366 |
| fragmented / vjp | 1/2048/8/512 | False | optimized | 8.816300 | 10.125348 | 0.87x | 8.503744 | 9.354615 |
| remat_packed / full | 1/8192/32/25 | True | optimized | 9.220772 | 8.795993 | 1.05x | 8.877366 | 8.466047 |
| remat_packed / vjp | 1/8192/32/25 | True | optimized | 5.627737 | 5.623081 | 1.00x | 5.323133 | 5.287578 |

## Same-input gate-parameter investigation and fix

The H32/B1/T8192/K=V128/N25 compact training profile exposed an A_log discrepancy relative to main. All three implementations were checked with the identical primal/cotangent digest `5b8f1fdf439b6d39eb90c819872a8f82b1dd9152ce12e5088f7a45cc8d5d3116` against an independent FP32, highest-dot-precision grouped recurrence.

| Implementation | A_log relative RMS | Bias relative RMS |
|---|---:|---:|
| Upstream main | 6.3884% | 5.1049% |
| Compact before fix | 3.1483% | 2.9126% |
| Compact after parameter-only fix | 1.0854% | 0.8919% |

Main is a performance comparator, not exact golden arithmetic. Before the fix, three A_log elements also exceeded the existing elementwise budget. The final fix returns the FP32 intra cotangent for parameter reductions, while independently reproducing the former BF16 store/reload for public token gradients. Output and q/k/v/gate/beta gradients are bitwise equal to the old implementation on the full profile. All eight real output/gradient leaves now pass the original compact bounds: 2% norm for token tensors, 5% norm for gate parameters, elementwise rtol0.05/atol1e-4, and atol1e-7 for zero-reference derivatives.

The new portable regression preserves the original benchmark RNG stream and selects eight heads, including the failing heads. It fails the old kernel and passes the corrected kernel. Its sorted-layout checks are functionalized with checkify; no checks or numerical bounds were disabled.

The formal ArgSpecs all enable Q/K normalization, so they do not dispatch this opt-in compact M5 path. Their executed code is unchanged by the isolated parameter reduction fix. Both compact feature modes and their complete device captures were rerun after the fix. Existing reference modules and non-CP token-gradient precision policy remain unchanged.

## Raw evidence and reproduction

[Data, full samples and per-leaf errors](benchmarks/review_20261116/) include the formal manifest, both-run samples, complete device-module durations, input hashes, environment and memory records. Original legacy feature-report headers inherited stale warmup/sample constants; derived records correct them to the executed loops and preserve the original header. No timing or output was changed.

The exact runtime harness is under `benchmarks/review_20261116/harness/`. Place pinned source trees under `/workspace/ready/main` and `/workspace/ready/backward`, generate or restore the serialized `/workspace/ready/inputs`, then run `python worker.py backward-default 0` (and the opt-in counterpart where available), followed by the reversed order in a fresh process. `feature_worker.py --suite backward --variant compact --round 0 --output-root /workspace/ready/features` runs the additional matrix. Set `JAX_PLATFORMS=tpu,cpu` and `LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=true`.

Protective raw archive: `gs://data_ant/diagnostics/kda-ready-review-20261116/95c92169-545d-4130-99bb-f6e0b88fad08/2c3c43fd3857/formal-stage1-raw.tgz#1794770854263763`. Final backward feature/accuracy archive: `gs://data_ant/diagnostics/kda-ready-review-20261116/95c92169-545d-4130-99bb-f6e0b88fad08/gatefix-5adb7a493e91/backward-final-raw.tgz#1794776280747726`. It includes current compact measurements, raw traces, the independent reference and failed old-kernel control. Immutable objects have verified size/CRC32C/SHA256 metadata; see archive-index.json. These forensic archives require internal GCS access; the manifest, methods, samples and kernels are public in this branch.


## Formal sweep provenance after the isolated M5 fix

The complete15-ArgSpec/18-point, two-order formal sweep finished before the
parameter-only M5 correction. Its source fingerprints are retained. The only
subsequent calculation-file change is compact_backward.py; every formal shape
enables Q/K normalization and the formal configurations do not enable compact
execution, so none dispatches that M5 file. All executed calculation files are
byte-identical after LF normalization. The machine-readable proof is
formal-dispatch-equivalence.json. The corrected compact forward/VJP profiles,
full-size independent reference and direct native regression were rerun on the
final calculation source.

An additional redundant final-source full sweep was interrupted twice by
production-quota preemption. It is not reported as completed acceptance, and
none of its partial observations replaces the finished formal dataset. This
preserves complete shape coverage without claiming uncompleted runs passed.
