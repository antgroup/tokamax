# KDA forward: final-source review validation

PR #1688. Tested calculation commit `1160af2a995129a5e6fd6d0a231f632e3d0d086f`; baseline upstream main `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18`. Measurements were collected on 2026-11-16 (Asia/Shanghai).

## Scope

Fused TPU forward stages, packed I/O, original-token compact execution, bounded metadata prefetch and resource guards. Shared packing fixes preserve Tokamax label-indexed state and original-position convolution. The compact training VJP is a separate increment in #1689.

## Correctness and source acceptance

Final synchronized source passed the 76-case native TPU7x forward/packing/padding suite. All eight additional aligned/compact forward profiles pass the original 2% relative-norm and 5%/1e-4 elementwise checks. Existing logical-state-capacity and convolution-parameter gradient tests remain enabled.

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
| fixed_training_b1_t8192_bf16 / forward | forward-default | 1.696256 | 1.704601 | +0.49% | 1.506584 | 1.507167 |
| fixed_training_b1_t8192_bf16 / forward | forward-opt | 1.696256 | 1.817690 | +7.16% | 1.506584 | 1.626680 |
| fixed_training_b1_t8192_bf16 / forward_and_vjp | forward-default | 4.019903 | 4.034019 | +0.35% | 3.725200 | 3.726272 |
| fixed_training_b1_t8192_bf16 / forward_and_vjp | forward-opt | 4.019903 | 4.155538 | +3.37% | 3.725200 | 3.856124 |
| packed_training_b1_t8192_n25_bf16 / forward | forward-default | 4.039549 | 3.822160 | -5.38% | 3.825856 | 3.600816 |
| packed_training_b1_t8192_n25_bf16 / forward | forward-opt | 4.039549 | 2.812769 | -30.37% | 3.825856 | 2.612209 |
| packed_training_b1_t8192_n25_bf16 / forward_and_vjp | forward-default | 7.615922 | 7.931989 | +4.15% | 7.215888 | 7.531835 |
| packed_training_b1_t8192_n25_bf16 / forward_and_vjp | forward-opt | 7.615922 | 8.948424 | +17.50% | 7.215888 | 8.523910 |
| gdn_fixed_training_b1_t8192_bf16 / forward | forward-default | 1.729614 | 1.720707 | -0.51% | 1.520061 | 1.519297 |
| gdn_fixed_training_b1_t8192_bf16 / forward | forward-opt | 1.729614 | 1.709966 | -1.14% | 1.520061 | 1.521145 |
| gdn_fixed_training_b1_t8192_bf16 / forward_and_vjp | forward-default | 3.371878 | 3.369789 | -0.06% | 3.093274 | 3.092822 |
| gdn_fixed_training_b1_t8192_bf16 / forward_and_vjp | forward-opt | 3.371878 | 3.370486 | -0.04% | 3.093274 | 3.094723 |
| tp32_prefill_b1_t128_bf16 / forward | forward-default | 0.231913 | 0.238279 | +2.74% | 0.017728 | 0.017662 |
| tp32_prefill_b1_t128_bf16 / forward | forward-opt | 0.231913 | 0.235895 | +1.72% | 0.017728 | 0.017600 |
| tp32_prefill_b1_t512_bf16 / forward | forward-default | 0.245647 | 0.252088 | +2.62% | 0.034594 | 0.034643 |
| tp32_prefill_b1_t512_bf16 / forward | forward-opt | 0.245647 | 0.258624 | +5.28% | 0.034594 | 0.034713 |
| tp32_prefill_b1_t1024_bf16 / forward | forward-default | 0.292392 | 0.279465 | -4.42% | 0.055379 | 0.055507 |
| tp32_prefill_b1_t1024_bf16 / forward | forward-opt | 0.292392 | 0.275717 | -5.70% | 0.055379 | 0.055517 |
| tp32_verify_b1_t64_n8_bf16 / forward | forward-default | 0.477208 | 0.422347 | -11.50% | 0.219741 | 0.160143 |
| tp32_verify_b1_t64_n8_bf16 / forward | forward-opt | 0.477208 | 0.468406 | -1.84% | 0.219741 | 0.226774 |
| tp16_prefill_b1_t1024_bf16 / forward | forward-default | 0.291921 | 0.303922 | +4.11% | 0.069692 | 0.070051 |
| tp16_prefill_b1_t1024_bf16 / forward | forward-opt | 0.291921 | 0.295388 | +1.19% | 0.069692 | 0.070008 |
| tp8_prefill_b1_t2048_bf16 / forward | forward-default | 0.428630 | 0.432351 | +0.87% | 0.185579 | 0.185463 |
| tp8_prefill_b1_t2048_bf16 / forward | forward-opt | 0.428630 | 0.431569 | +0.69% | 0.185579 | 0.185493 |
| tp32_decode_b1_t1_n1_bf16 / forward | forward-default | 0.255163 | 0.257203 | +0.80% | 0.029384 | 0.026598 |
| tp32_decode_b1_t1_n1_bf16 / forward | forward-opt | 0.255163 | 0.244968 | -4.00% | 0.029384 | 0.026778 |
| tp32_dspark_decode_b1_t8_n1_bf16 / forward | forward-default | 0.259823 | 0.273691 | +5.34% | 0.043058 | 0.043011 |
| tp32_dspark_decode_b1_t8_n1_bf16 / forward | forward-opt | 0.259823 | 0.266533 | +2.58% | 0.043058 | 0.044858 |
| tp32_dspark_decode_b1_t64_n16_bf16 / forward | forward-default | 0.645495 | 0.523169 | -18.95% | 0.381563 | 0.277704 |
| tp32_dspark_decode_b1_t64_n16_bf16 / forward | forward-opt | 0.645495 | 0.633971 | -1.79% | 0.381563 | 0.377510 |
| tp32_decode_b1_t64_n64_bf16 / forward | forward-default | 1.563943 | 1.251168 | -20.00% | 1.304905 | 0.997688 |
| tp32_decode_b1_t64_n64_bf16 / forward | forward-opt | 1.563943 | 1.607787 | +2.80% | 1.304905 | 1.351001 |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 / forward | forward-default | 0.655948 | 0.534064 | -18.58% | 0.396683 | 0.294965 |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 / forward | forward-opt | 0.655948 | 0.654966 | -0.15% | 0.396683 | 0.394690 |
| tp16_dp2_decode_b1_t64_n64_bf16 / forward | forward-default | 1.669431 | 1.320695 | -20.89% | 1.412362 | 1.056382 |
| tp16_dp2_decode_b1_t64_n64_bf16 / forward | forward-opt | 1.669431 | 1.713036 | +2.61% | 1.412362 | 1.451233 |

Formal opt-in flags: `fuse_forward=True`, `packed_forward=True`, `packed_output=True`. Compact feature profiles additionally set `compact_forward=True` and `compact_sorted_segments=True`. All flags remain opt-in.

## Feature-specific comparison

These additional profiles use identical Gaussian inputs and a fixed external Gaussian cotangent (where applicable), seed619. BF16 Q/K/V have std0.05, raw gate std0.1, beta is sigmoid of Gaussian logits with std0.1; A_log is uniform[-1,0] and bias uniform[-4,-3]. Lower bound is -5, scale is 128**-0.5. Both implementations use matching Q/K normalization options. Two reversed fresh-process sweeps retain sixty host-wall samples per cell. Five-call complete-device captures in each sweep contain zero input-reshard modules.

Pure VJP prepares residuals before timing and supplies them as runtime inputs. The harness rejects a largest closed-over constant of 1MiB or more. Full forward+VJP includes preparation and its seven operand/parameter gradients.

| Case / mode | B/T/H/N | Norm | Path | Main wall | PR wall | Speedup | Main device | PR device |
|---|---|---|---|---:|---:|---:|---:|---:|
| compact_packed / forward | 1/8192/32/25 | False | compact | 3.747197 | 1.340058 | 2.80x | 3.559145 | 1.055396 |
| fragmented / forward | 1/2048/8/512 | False | compact | 14.341632 | 0.412449 | 34.77x | 13.661526 | 0.135409 |
| long64k / forward | 1/65536/8/64 | False | compact | 8.011243 | 3.317516 | 2.41x | 7.813831 | 3.008962 |
| batch2 / forward | 2/8192/64/25 | False | compact | 14.993448 | 4.413643 | 3.40x | 14.190504 | 4.102051 |
| compact_packed / forward | 1/8192/32/25 | False | aligned | 3.747197 | 2.653976 | 1.41x | 3.559145 | 2.460715 |
| fragmented / forward | 1/2048/8/512 | False | aligned | 14.341632 | 4.745816 | 3.02x | 13.661526 | 4.543172 |
| long64k / forward | 1/65536/8/64 | False | aligned | 8.011243 | 5.842321 | 1.37x | 7.813831 | 5.645061 |
| batch2 / forward | 2/8192/64/25 | False | aligned | 14.993448 | 10.850404 | 1.38x | 14.190504 | 10.156752 |

The compact rows disable in-kernel Q/K normalization in both implementations. Forward-only improvements are the feature target of this PR. Optimized training VJP is supplied separately by #1689; the full formal table retains the effects of pairing these forward flags with the original backward.

## Raw evidence and reproduction

[Data, full samples and per-leaf errors](benchmarks/review_20261116/) include the formal manifest, both-run samples, complete device-module durations, input hashes, environment and memory records. Original legacy feature-report headers inherited stale warmup/sample constants; derived records correct them to the executed loops and preserve the original header. No timing or output was changed.

The exact runtime harness is under `benchmarks/review_20261116/harness/`. Place pinned source trees under `/workspace/ready/main` and `/workspace/ready/forward`, generate or restore the serialized `/workspace/ready/inputs`, then run `python worker.py forward-default 0` (and the opt-in counterpart where available), followed by the reversed order in a fresh process. `feature_worker.py --suite forward --variant compact --round 0 --output-root /workspace/ready/features` runs the additional matrix. Set `JAX_PLATFORMS=tpu,cpu` and `LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=true`.

Protective raw archive: `gs://data_ant/diagnostics/kda-ready-review-20261116/95c92169-545d-4130-99bb-f6e0b88fad08/2c3c43fd3857/formal-stage1-raw.tgz#1794770854263763`. Final forward/inference feature archive: `gs://data_ant/diagnostics/kda-ready-review-20261116/95c92169-545d-4130-99bb-f6e0b88fad08/2c3c43fd3857/forward-inference-final-raw.tgz#1794775402685712`. It includes the feature traces, arrays, acceptance and pinned source trees. Immutable objects have verified size/CRC32C/SHA256 metadata; see archive-index.json. These forensic archives require internal GCS access; the manifest, methods, samples and kernels are public in this branch.
