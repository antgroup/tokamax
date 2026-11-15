# KDA standalone inference: final-source review validation

PR #1691. Tested calculation commit `a3096ff39ae1aeb78a4454203627e926527b9da1`; baseline upstream main `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18`. Measurements were collected on 2026-11-16 (Asia/Shanghai).

## Scope

Standalone native-segment BF16 prefill with fused safe gate activation, Q/K normalization and optional FP32 initial/final states. Padding, resumed-label grouping, unused state slots and grouped input layouts are preserved. This entry has no training VJP.

## Correctness and source acceptance

Final synchronized source passed the 50-case native TPU7x inference/packing/padding suite, including the 28 standalone inference/lowering cases. All five additional standalone prefill profiles pass the existing numerical bounds. Formal default-API checks are distinct from standalone inference profiles.

All prior reference modules and numerical thresholds are unchanged. Default formal endpoints are bitwise equal to main at 15/18 execution points; short multi-state convolution differences have maximum per-leaf relative RMS below 1.4e-5. Explicit fused/packed formal paths retain their original BF16 rounding. Both reversed sweeps reproduce endpoint hashes.

## Formal operator coverage and method

- All 15 official ArgSpecs, comprising 15 forward and three full forward/VJP execution points. Serialized primals are reused byte-for-byte; seed619, beta=0.5, with logical-input hashes in the manifest.
- `tokamax.standardize_function(..., seed=None)` supplies the modes. Its full forward/VJP uses the output itself as cotangent, behind an optimization barrier. Cotangents are consequently output-derived; this is not a fixed external-dY experiment.
- Dedicated TPU7x allocation: four physical chips / eight JAX devices; single-device timing on logical device0. JAX/jaxlib0.12.0, libtpu0.0.50.1. Complete observed versions and flags are in environments.json.
- Two fresh-process sweeps reverse implementation order. Inputs are prepared once using compiled input Formats. Ten warmups precede thirty blocking host-wall samples per cell per sweep; the table uses the median of all sixty samples.
- Each formal execution point has two independent five-call XPlane captures. Device time is the complete jit_invoke module, including metadata, copies and all forward/reverse stages. Every capture has zero input-reshard modules.

### Shapes

Layout is [H,B,T,K/V]; B=1 and K=V=128. Tokens/gates/beta use BF16; gate parameters and initial/final states use FP32. All formal shapes enable Q/K normalization. Stateful cases include width-4 convolution.

| Official case | H | T | N | Gate width | State | Modes |
|---|---:|---:|---:|---:|---|---|
| fixed_training_b1_t8192_bf16 | 32 | 8192 | fixed | 128 | none | F, F+VJP |
| packed_training_b1_t8192_n25_bf16 | 32 | 8192 | 25 | 128 | none | F, F+VJP |
| gdn_fixed_training_b1_t8192_bf16 | 32 | 8192 | fixed | 1 | none | F, F+VJP |
| tp32_prefill_b1_t128_bf16 | 3 | 128 | fixed | 128 | FP32 | F |
| tp32_prefill_b1_t512_bf16 | 3 | 512 | fixed | 128 | FP32 | F |
| tp32_prefill_b1_t1024_bf16 | 3 | 1024 | fixed | 128 | FP32 | F |
| tp32_verify_b1_t64_n8_bf16 | 3 | 64 | 8 | 128 | FP32 | F |
| tp16_prefill_b1_t1024_bf16 | 6 | 1024 | fixed | 128 | FP32 | F |
| tp8_prefill_b1_t2048_bf16 | 12 | 2048 | fixed | 128 | FP32 | F |
| tp32_decode_b1_t1_n1_bf16 | 3 | 1 | 1 | 128 | FP32 | F |
| tp32_dspark_decode_b1_t8_n1_bf16 | 3 | 8 | 1 | 128 | FP32 | F |
| tp32_dspark_decode_b1_t64_n16_bf16 | 3 | 64 | 16 | 128 | FP32 | F |
| tp32_decode_b1_t64_n64_bf16 | 3 | 64 | 64 | 128 | FP32 | F |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 | 6 | 64 | 16 | 128 | FP32 | F |
| tp16_dp2_decode_b1_t64_n64_bf16 | 6 | 64 | 64 | 128 | FP32 | F |

### Complete formal wall/device matrix

Times are milliseconds. Each row explicitly identifies the default or opt-in path; no universal speedup is claimed. Compiler temporary bytes are provided in the CSV/JSON and are not model/allocator peak memory.

| Case / mode | Path | Main wall | PR wall | Change | Main device | PR device |
|---|---|---:|---:|---:|---:|---:|
| fixed_training_b1_t8192_bf16 / forward | inference-default | 1.696256 | 1.684970 | -0.67% | 1.506584 | 1.507192 |
| fixed_training_b1_t8192_bf16 / forward_and_vjp | inference-default | 4.019903 | 3.993761 | -0.65% | 3.725200 | 3.725407 |
| packed_training_b1_t8192_n25_bf16 / forward | inference-default | 4.039549 | 3.795079 | -6.05% | 3.825856 | 3.601781 |
| packed_training_b1_t8192_n25_bf16 / forward_and_vjp | inference-default | 7.615922 | 7.912871 | +3.90% | 7.215888 | 7.535714 |
| gdn_fixed_training_b1_t8192_bf16 / forward | inference-default | 1.729614 | 1.692128 | -2.17% | 1.520061 | 1.520464 |
| gdn_fixed_training_b1_t8192_bf16 / forward_and_vjp | inference-default | 3.371878 | 3.354042 | -0.53% | 3.093274 | 3.093317 |
| tp32_prefill_b1_t128_bf16 / forward | inference-default | 0.231913 | 0.209168 | -9.81% | 0.017728 | 0.017654 |
| tp32_prefill_b1_t512_bf16 / forward | inference-default | 0.245647 | 0.237195 | -3.44% | 0.034594 | 0.034713 |
| tp32_prefill_b1_t1024_bf16 / forward | inference-default | 0.292392 | 0.257725 | -11.86% | 0.055379 | 0.055337 |
| tp32_verify_b1_t64_n8_bf16 / forward | inference-default | 0.477208 | 0.378155 | -20.76% | 0.219741 | 0.160229 |
| tp16_prefill_b1_t1024_bf16 / forward | inference-default | 0.291921 | 0.278338 | -4.65% | 0.069692 | 0.069896 |
| tp8_prefill_b1_t2048_bf16 / forward | inference-default | 0.428630 | 0.404998 | -5.51% | 0.185579 | 0.185558 |
| tp32_decode_b1_t1_n1_bf16 / forward | inference-default | 0.255163 | 0.215736 | -15.45% | 0.029384 | 0.026588 |
| tp32_dspark_decode_b1_t8_n1_bf16 / forward | inference-default | 0.259823 | 0.238611 | -8.16% | 0.043058 | 0.042948 |
| tp32_dspark_decode_b1_t64_n16_bf16 / forward | inference-default | 0.645495 | 0.499156 | -22.67% | 0.381563 | 0.277827 |
| tp32_decode_b1_t64_n64_bf16 / forward | inference-default | 1.563943 | 1.235778 | -20.98% | 1.304905 | 0.997161 |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 / forward | inference-default | 0.655948 | 0.518081 | -21.02% | 0.396683 | 0.294966 |
| tp16_dp2_decode_b1_t64_n64_bf16 / forward | inference-default | 1.669431 | 1.301476 | -22.04% | 1.412362 | 1.057271 |

Standalone profiles call `kimi_delta_attention_inference` with safe gate activation, `use_qk_l2norm_in_kernel=True`, bounded raw gate with lower_bound=-5, and optional FP32 state. Formal rows test the shared default KDA operator separately.

## Feature-specific comparison

These additional profiles use identical Gaussian inputs and a fixed external Gaussian cotangent (where applicable), seed619. BF16 Q/K/V have std0.05, raw gate std0.1, beta is sigmoid of Gaussian logits with std0.1; A_log is uniform[-1,0] and bias uniform[-4,-3]. Lower bound is -5, scale is 128**-0.5. Both implementations use matching Q/K normalization options. Two reversed fresh-process sweeps retain sixty host-wall samples per cell. Five-call complete-device captures in each sweep contain zero input-reshard modules.

Pure VJP prepares residuals before timing and supplies them as runtime inputs. The harness rejects a largest closed-over constant of 1MiB or more. Full forward+VJP includes preparation and its seven operand/parameter gradients.

| Case / mode | B/T/H/N | Norm | Path | Main wall | PR wall | Speedup | Main device | PR device |
|---|---|---|---|---:|---:|---:|---:|---:|
| tokens512 / forward | 1/512/8/2 | True | native | 0.291922 | 0.197008 | 1.48x | 0.118696 | 0.043809 |
| tokens8192 / forward | 1/8192/8/2 | True | native | 1.200785 | 0.557651 | 2.15x | 1.009069 | 0.364851 |
| arg_packed / forward | 1/8192/32/25 | True | native | 3.990626 | 4.656812 | 0.86x | 3.778398 | 4.445441 |
| fragmented / forward | 1/2048/8/512 | True | native | 14.685554 | 1.783583 | 8.23x | 14.019798 | 1.565655 |
| state512 / forward | 1/512/8/2 | True | native | 0.354039 | 0.291668 | 1.21x | 0.119034 | 0.060201 |

Both standalone inference wrappers accept BTHD inputs and return BTHD output plus optional BNHKV FP32 state. Main layout conversions are inside the timed wrapper. These are prefill profiles; standalone inference does not add training-VJP registration. The H32/T8192/N25 profile remains slower than main, as retained above; improvements depend on shape.

## Raw evidence and reproduction

[Data, full samples and per-leaf errors](benchmarks/review_20261116/) include the formal manifest, both-run samples, complete device-module durations, input hashes, environment and memory records. Original legacy feature-report headers inherited stale warmup/sample constants; derived records correct them to the executed loops and preserve the original header. No timing or output was changed.

The exact runtime harness is under `benchmarks/review_20261116/harness/`. Place pinned source trees under `/workspace/ready/main` and `/workspace/ready/inference`, generate or restore the serialized `/workspace/ready/inputs`, then run `python worker.py inference-default 0` (and the opt-in counterpart where available), followed by the reversed order in a fresh process. `feature_worker.py --suite inference --variant native --round 0 --output-root /workspace/ready/features` runs the additional matrix. Set `JAX_PLATFORMS=tpu,cpu` and `LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=true`.

Protective raw archive: `gs://data_ant/diagnostics/kda-ready-review-20261116/95c92169-545d-4130-99bb-f6e0b88fad08/2c3c43fd3857/formal-stage1-raw.tgz#1794770854263763`. Final forward/inference feature archive: `gs://data_ant/diagnostics/kda-ready-review-20261116/95c92169-545d-4130-99bb-f6e0b88fad08/2c3c43fd3857/forward-inference-final-raw.tgz#1794775402685712`. It includes the feature traces, arrays, acceptance and pinned source trees. Immutable objects have verified size/CRC32C/SHA256 metadata; see archive-index.json. These forensic archives require internal GCS access; the manifest, methods, samples and kernels are public in this branch.
