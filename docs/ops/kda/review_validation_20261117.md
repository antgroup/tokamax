# #1688 performance regression investigation (2026-11-16/17)

Status: **Draft.** This report records the final paired measurements for #1688 against the pinned latest-main snapshot. It supersedes the current-head performance conclusions in the 2026-11-16 report; historical numbers retain their original baselines.

## Source pins and scope

- Latest upstream main at measurement start: `9d65d382dfa59324d2707c5bfc71b7d7719ac34f`.
- Measured and published candidate: `7b3102e462a9baca278ba5dbb2db2268a6560d60`.
- Initial regression reproduction: original forward PR `649506c61ca9b623da380170c4e739dfb95e142e`, upstream `48ed8b16145b43d56e608ade2f701ba003e7eb7c`. These are separate diagnostic measurements, not the final latest-main baseline.
- Internal pallas-kernel main: `b8bfae8ae5f3678f68b327e3da108302cf542135`; older internal control: `dca6a09feec2659f2777a6db02f2ee55560f991e`.
- All 38 KDA Python source files and `tokamax/conftest.py` in the final benchmark tree match the published candidate Git blobs after CRLF/LF normalization. Experimental full metadata unrolling was rejected and is absent from the candidate.

## What changed

Training shares one packing preparation/pullback, normalizes Q/K once, and reuses the aligned inputs required by backward instead of preparing a second direct-packed training representation. Forward-only direct-packed execution remains available. Gradient padding clearing is fused into output restoration, retaining convolution input clearing and gate-parameter gradient protection. Aligned/unpacked fused forward uses a maximum 32-head group within the existing VMEM budget; direct-packed forward retains its prior grouping and constraints.

Backward consumes a saved FP32 gate prefix rather than computing it again. If the prefix was discarded, the rematerialization path still computes it. Forward fused and staged prefix accumulation need not be bitwise identical; the existing numerical tests and tolerances are retained.

A separate native CP compiler failure was reproduced in the original PR and did not reproduce on upstream main. Chunk-local gated Q/K construction avoids the TPU compiler's flattened BF16 tuple fusion failure. The original W/u matrix arithmetic is unchanged. Both BF16 and FP32 CP rematerialization checks passed.

Output restoration uses a static internal guard guarantee for buffers allocated by `_align_seqs` with alignment greater than one. It zeros one guard row and redirects invalid indices there, avoiding a full floating-point output selection. The tight allocation-bound case is covered by explicit tests; the last allocated row is then padding. Arbitrary external buffers retain the masked fallback. This restoration change preserves the complete forward output and gradient checksums on the frozen target input.

The packed-output fallback forwards the same guard proof and its actual chunk-alignment metadata, reusing guarded restoration and avoiding redundant floating-point clearing. The dedicated packed-output kernel and its eligibility rules remain. Native checks cover both guard flags, poisoned tails and scalar-gate chunk sizes 16/32/64.

No optimization path was removed and no numerical tolerance was relaxed.

## Method

TPU7x, one logical TPU device per operator invocation. The diagnostic Pod has four physical chips/eight logical devices; these numbers do not describe whole-slice throughput. Python 3.12.14, JAX/jaxlib 0.12.0, libtpu 0.0.50.1, XProf 2.24.2. All official input bytes are frozen in the manifest and verified with SHA-256 in each worker. Both sources use `LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=true`.

The 15 official ArgSpecs produce 18 valid case/mode pairs: three training specs have both forward and full forward+VJP; the 12 `forward_only` specs have forward only. Full forward+VJP includes forward execution, not pure backward. The public wrapper has the same explicit `query,key,value,gate,beta,**kwargs` signature for both sources. The official `standardize_function` uses the forward output as cotangent. This is the same cotangent construction, not a promise of identical cotangent bytes when implementations return different outputs. Independent constant-cotangent tests cover correctness separately.

Inputs are transferred/resharded to each compiled executable's input formats once before warmup. Compilation and that initial input preparation are excluded. Wall time includes dispatch and `block_until_ready`. Each source/config has two fresh-process rounds, 10 warmups and 30 timed samples per round. Round 2 reverses source/config order. Tables use the median of all 60 wall samples, retaining per-round medians in CSV.

Device time comes from the `jit_invoke` events on `/device:TPU:0`, `XLA Modules`, not summed nested/async events. Each fresh-process round has two separate five-call traces of its executable. The final table uses twenty device-event durations from two compilations; per-round device medians are also retained. Traces within a round share that compilation. Profiling of the second round was extended before that round started to assess microsecond-scale compiler variability; earlier wall samples and trace files were left unchanged. XProf picoseconds are divided by 1e9 to obtain milliseconds. Wall and device time are distinct measurements; neither excludes or subtracts arbitrary outliers.

Configurations are default, `fuse_forward`, `fuse_forward+packed_forward`, the combination with `packed_output`, and `compact_forward`. All official inputs enable Q/K normalization, so the compact-only configuration exercises its documented fallback rather than active compact execution. Convolution and short-shape fallbacks must likewise be distinguished from active fused paths. Requested and applied configs and HLO are retained in raw evidence. The prior normalization-disabled compact speedup is outside this official matrix.

## Internal main comparison

On the same frozen packed-training input (B1/H32/T8192/N25, K=V128, BF16), the older internal implementation had round wall medians 5.5604/5.6403 ms; current internal main had 5.8696/5.7359 ms. Medians over 60 samples were 5.6147 and 5.8372 ms respectively (+3.96%). These diagnostic calls use a direct operator wrapper and are not interchangeable with the final public-wrapper matrix.

Device profiles showed approximately 5.18 -> 5.38 ms, with the major forward/backward kernels substantially unchanged. A diagnostic bypass of padding compaction approached the older wall time. That bypass was used only for attribution and was not adopted, because the padding correctness requirement remains. This smaller internal overhead must not be described as the same 17.5% Tokamax regression. Internal native-normalization probes use another contract and are not substituted into this comparison.

## Numerical validation

The final packed-output-fallback cleanup passed **43 native tests** (152.22 seconds), including seven added cases for both guard flags, BF16/FP32 poisoned tails and scalar-gate chunk sizes 16/32/64. The eight poisoned-tail CPU checks also passed. Native short-shape probes on the official T1/T8 inputs preserved the preceding checkpoint's full forward/state output checksums across default and opt-in configurations.

The guard-row checkpoint `f0fd1018` passed 190 targeted native tests and 3 subtests, both BF16/FP32 CP-rematerialization checks and 56 CPU guard cases. Its native selection included the original long CP compiler reproduction. The preceding saved-prefix checkpoint `79eb620d` passed all 448 native tests/3 subtests in the default non-long selection. Both preceding checkpoints completed their upstream TPU6e/TPU7x/H100 CI successfully. These overlapping test selections are not summed as independent coverage. The entire long stress matrix is not claimed complete.

The final source's [complete upstream CI](https://github.com/openxla/tokamax/actions/runs/43947692708) passed, including TPU6e, TPU7x and H100. Existing tolerances and all original tests remain unchanged.

## Final results

### Shapes

The `tp32/tp16/tp8` names denote tensor-parallel factors, not head counts. N is logical state/sequence capacity; unsegmented rows have implicit N=1. Q/K/V are BF16; exact dtypes for all parameters are retained in each raw manifest.

| ArgSpec | H / B / T / K / V | N | Gate width | Conv width |
|---|---:|---:|---:|---:|
| `fixed_training_b1_t8192_bf16` | 32 / 1 / 8192 / 128 / 128 | 1 | 128 | - |
| `packed_training_b1_t8192_n25_bf16` | 32 / 1 / 8192 / 128 / 128 | 25 | 128 | - |
| `gdn_fixed_training_b1_t8192_bf16` | 32 / 1 / 8192 / 128 / 128 | 1 | 1 | - |
| `tp32_prefill_b1_t128_bf16` | 3 / 1 / 128 / 128 / 128 | 1 | 128 | 4 |
| `tp32_prefill_b1_t512_bf16` | 3 / 1 / 512 / 128 / 128 | 1 | 128 | 4 |
| `tp32_prefill_b1_t1024_bf16` | 3 / 1 / 1024 / 128 / 128 | 1 | 128 | 4 |
| `tp32_verify_b1_t64_n8_bf16` | 3 / 1 / 64 / 128 / 128 | 8 | 128 | 4 |
| `tp16_prefill_b1_t1024_bf16` | 6 / 1 / 1024 / 128 / 128 | 1 | 128 | 4 |
| `tp8_prefill_b1_t2048_bf16` | 12 / 1 / 2048 / 128 / 128 | 1 | 128 | 4 |
| `tp32_decode_b1_t1_n1_bf16` | 3 / 1 / 1 / 128 / 128 | 1 | 128 | 4 |
| `tp32_dspark_decode_b1_t8_n1_bf16` | 3 / 1 / 8 / 128 / 128 | 1 | 128 | 4 |
| `tp32_dspark_decode_b1_t64_n16_bf16` | 3 / 1 / 64 / 128 / 128 | 16 | 128 | 4 |
| `tp32_decode_b1_t64_n64_bf16` | 3 / 1 / 64 / 128 / 128 | 64 | 128 | 4 |
| `tp16_dp2_dspark_decode_b1_t64_n16_bf16` | 6 / 1 / 64 / 128 / 128 | 16 | 128 | 4 |
| `tp16_dp2_decode_b1_t64_n64_bf16` | 6 / 1 / 64 / 128 / 128 | 64 | 128 | 4 |

### Timings

The same frozen input bytes are used on both sources. Positive percentages mean a slowdown. All times are milliseconds.

| ArgSpec / mode | Main wall / device | Default wall / device | Default change wall / device | Opt-in wall / device | Opt-in change wall / device |
|---|---:|---:|---:|---:|---:|
| `fixed_training_b1_t8192_bf16` / FWD | 1.686122 / 1.505859 | 1.708766 / 1.505826 | +1.34% / -0.00% | 1.685938 / 1.492857 | -0.01% / -0.86% |
| `fixed_training_b1_t8192_bf16` / FWD+VJP | 3.922046 / 3.635707 | 3.545141 / 3.234444 | -9.61% / -11.04% | 3.554455 / 3.253286 | -9.37% / -10.52% |
| `packed_training_b1_t8192_n25_bf16` / FWD | 3.957461 / 3.767022 | 3.746259 / 3.541185 | -5.34% / -6.00% | 2.784722 / 2.570594 | -29.63% / -31.76% |
| `packed_training_b1_t8192_n25_bf16` / FWD+VJP | 7.529640 / 7.115837 | 7.445359 / 7.018262 | -1.12% / -1.37% | 7.396705 / 6.963541 | -1.77% / -2.14% |
| `gdn_fixed_training_b1_t8192_bf16` / FWD | 1.697998 / 1.517094 | 1.713094 / 1.517322 | +0.89% / +0.02% | 1.701492 / 1.516978 | +0.21% / -0.01% |
| `gdn_fixed_training_b1_t8192_bf16` / FWD+VJP | 3.345201 / 3.081466 | 3.156485 / 2.883584 | -5.64% / -6.42% | 3.143149 / 2.883361 | -6.04% / -6.43% |
| `tp32_prefill_b1_t128_bf16` / FWD | 0.220178 / 0.017654 | 0.234079 / 0.017747 | +6.31% / +0.52% | 0.228135 / 0.017703 | +3.61% / +0.28% |
| `tp32_prefill_b1_t512_bf16` / FWD | 0.234951 / 0.034391 | 0.247381 / 0.034367 | +5.29% / -0.07% | 0.266694 / 0.034267 | +13.51% / -0.36% |
| `tp32_prefill_b1_t1024_bf16` / FWD | 0.275277 / 0.055267 | 0.278626 / 0.055216 | +1.22% / -0.09% | 0.283541 / 0.055159 | +3.00% / -0.20% |
| `tp32_verify_b1_t64_n8_bf16` / FWD | 0.449128 / 0.218655 | 0.375538 / 0.132650 | -16.38% / -39.33% | 0.325835 / 0.092797 | -27.45% / -57.56% |
| `tp16_prefill_b1_t1024_bf16` / FWD | 0.279697 / 0.069739 | 0.310968 / 0.069670 | +11.18% / -0.10% | 0.298845 / 0.069836 | +6.85% / +0.14% |
| `tp8_prefill_b1_t2048_bf16` / FWD | 0.413883 / 0.185199 | 0.436403 / 0.185274 | +5.44% / +0.04% | 0.436350 / 0.185200 | +5.43% / +0.00% |
| `tp32_decode_b1_t1_n1_bf16` / FWD | 0.227640 / 0.029267 | 0.253236 / 0.027169 | +11.24% / -7.17% | 0.254599 / 0.026887 | +11.84% / -8.13% |
| `tp32_dspark_decode_b1_t8_n1_bf16` / FWD | 0.245113 / 0.042866 | 0.269610 / 0.043215 | +9.99% / +0.81% | 0.271915 / 0.043138 | +10.93% / +0.63% |
| `tp32_dspark_decode_b1_t64_n16_bf16` / FWD | 0.628581 / 0.378813 | 0.480197 / 0.222930 | -23.61% / -41.15% | 0.383253 / 0.149353 | -39.03% / -60.57% |
| `tp32_decode_b1_t64_n64_bf16` / FWD | 1.546375 / 1.295106 | 1.016335 / 0.748687 | -34.28% / -42.19% | 0.704319 / 0.442952 | -54.45% / -65.80% |
| `tp16_dp2_dspark_decode_b1_t64_n16_bf16` / FWD | 0.637590 / 0.394034 | 0.502861 / 0.240984 | -21.13% / -38.84% | 0.396239 / 0.164228 | -37.85% / -58.32% |
| `tp16_dp2_decode_b1_t64_n64_bf16` / FWD | 1.625042 / 1.389666 | 1.124637 / 0.864833 | -30.79% / -37.77% | 0.793019 / 0.536770 | -51.20% / -61.37% |

The six-configuration scan contains 108 case/mode/config cells. [Complete CSV](benchmarks/regression_guard_20261117/restoration-combined.csv) and [full-precision JSON](benchmarks/regression_guard_20261117/restoration-combined.json) retain all samples/round summaries and the fused-only, packed-input and compact-fallback configurations. Each raw report retains its exact ArgSpec shapes, array dtypes, requested/applied config, input hash, executable formats, temporary-memory estimate and software/source hashes. Compilation and the one-time resident-input preparation are excluded from these steady-state times.

### Positive device deltas

These observed median increases remain visible. Per-compilation device medians in the CSV help assess repeatability and separate small absolute differences from larger changes.

| Configuration / case / mode | Main device ms | Candidate device ms | Change |
|---|---:|---:|---:|
| forward-default / `gdn_fixed_training_b1_t8192_bf16` / forward | 1.517094 | 1.517322 | +0.02% |
| forward-default / `tp32_prefill_b1_t128_bf16` / forward | 0.017654 | 0.017747 | +0.52% |
| forward-default / `tp8_prefill_b1_t2048_bf16` / forward | 0.185199 | 0.185274 | +0.04% |
| forward-default / `tp32_dspark_decode_b1_t8_n1_bf16` / forward | 0.042866 | 0.043215 | +0.81% |
| forward-opt / `tp32_prefill_b1_t128_bf16` / forward | 0.017654 | 0.017703 | +0.28% |
| forward-opt / `tp16_prefill_b1_t1024_bf16` / forward | 0.069739 | 0.069836 | +0.14% |
| forward-opt / `tp8_prefill_b1_t2048_bf16` / forward | 0.185199 | 0.185200 | +0.00% |
| forward-opt / `tp32_dspark_decode_b1_t8_n1_bf16` / forward | 0.042866 | 0.043138 | +0.63% |
| forward-fused / `gdn_fixed_training_b1_t8192_bf16` / forward | 1.517094 | 1.517160 | +0.00% |
| forward-fused / `tp32_prefill_b1_t128_bf16` / forward | 0.017654 | 0.017741 | +0.49% |
| forward-fused / `tp16_prefill_b1_t1024_bf16` / forward | 0.069739 | 0.069794 | +0.08% |
| forward-fused / `tp8_prefill_b1_t2048_bf16` / forward | 0.185199 | 0.185311 | +0.06% |
| forward-fused / `tp32_dspark_decode_b1_t8_n1_bf16` / forward | 0.042866 | 0.043179 | +0.73% |
| forward-packed / `gdn_fixed_training_b1_t8192_bf16` / forward | 1.517094 | 1.517335 | +0.02% |
| forward-packed / `tp32_prefill_b1_t512_bf16` / forward | 0.034391 | 0.034451 | +0.18% |
| forward-packed / `tp8_prefill_b1_t2048_bf16` / forward | 0.185199 | 0.185307 | +0.06% |
| forward-packed / `tp32_dspark_decode_b1_t8_n1_bf16` / forward | 0.042866 | 0.043156 | +0.68% |
| forward-compact / `fixed_training_b1_t8192_bf16` / forward | 1.505859 | 1.506130 | +0.02% |
| forward-compact / `gdn_fixed_training_b1_t8192_bf16` / forward | 1.517094 | 1.517335 | +0.02% |
| forward-compact / `tp32_prefill_b1_t128_bf16` / forward | 0.017654 | 0.017689 | +0.20% |
| forward-compact / `tp32_prefill_b1_t512_bf16` / forward | 0.034391 | 0.034442 | +0.15% |
| forward-compact / `tp8_prefill_b1_t2048_bf16` / forward | 0.185199 | 0.185367 | +0.09% |
| forward-compact / `tp32_dspark_decode_b1_t8_n1_bf16` / forward | 0.042866 | 0.043263 | +0.93% |

### Active compact supplement

Main wall/device 3.682240/3.490855 ms; compact 1.352151/1.051130 ms. Speedups: 2.72x wall, 3.32x device. Thirty wall samples and ten device-event samples per source.

The official normalized matrix exercises the compact fallback. A separate forward-only supplement uses the historical seed-619 Gaussian feature recipe: Q/K/V std 0.05, gate std 0.1, beta from sigmoid logits std 0.1, A_log uniform[-1,0], bias uniform[-4,-3]. B1/H32/T8192/N25, K=V128; normalization is disabled on both sources, with no external L2 normalization. This uses distinct feature inputs rather than relabeling the official matrix. The PR enables fused/packed-input/packed-output/compact/sorted-segment flags, while upstream main uses its default path. Both receive the same frozen feature-input bytes. These supplementary numbers are retained separately from the 60-sample official table.

Supplementary wall samples and raw profiles are in `restoration-compact-supplement/results` and the immutable phase archives. Device durations are reported in the supplement summary alongside those raw reports.


### Host-wall paired diagnostic

The official independent-process table is retained in full. In particular, the T8 combined opt-in row still records +10.93% wall and +0.63% device (about +0.27 microseconds of device time). All three originally targeted regressions are eliminated, while small short-shape overhead remains; this report does not claim a universal per-case speedup.

To characterize host/launch variability, the pinned main and candidate were compiled in one Python/JAX client, with each executable using its own resident input formats and the same frozen input bytes. After ten warmups per side, each comparison used forty alternating ABBA/BAAB blocks: eighty synchronized wall samples per side. The block delta is mean(candidate calls) minus mean(main calls) within each block. This is a separate diagnostic protocol and does not replace the official table.

| Case / candidate | Main wall ms | Candidate wall ms | Median change | Median paired-block delta us |
|---|---:|---:|---:|---:|
| `tp32_decode_b1_t1_n1_bf16` / forward-default | 0.247813 | 0.246750 | -0.43% | -0.229 |
| `tp32_decode_b1_t1_n1_bf16` / forward-opt | 0.248935 | 0.249177 | +0.10% | -0.751 |
| `tp32_decode_b1_t1_n1_bf16` / forward-compact | 0.244382 | 0.244602 | +0.09% | +0.567 |
| `tp32_dspark_decode_b1_t8_n1_bf16` / forward-default | 0.265697 | 0.270560 | +1.83% | +4.283 |
| `tp32_dspark_decode_b1_t8_n1_bf16` / forward-opt | 0.274749 | 0.280293 | +2.02% | +2.321 |
| `tp32_dspark_decode_b1_t8_n1_bf16` / forward-compact | 0.273081 | 0.276446 | +1.23% | +3.449 |
| `tp32_prefill_b1_t1024_bf16` / forward-default | 0.278701 | 0.279369 | +0.24% | +0.404 |
| `tp32_prefill_b1_t1024_bf16` / forward-opt | 0.277513 | 0.276035 | -0.53% | -0.537 |
| `tp32_prefill_b1_t1024_bf16` / forward-compact | 0.283250 | 0.283077 | -0.06% | +0.388 |
| `fixed_training_b1_t8192_bf16` / forward-default | 1.692798 | 1.692550 | -0.01% | +0.358 |
| `fixed_training_b1_t8192_bf16` / forward-opt | 1.702500 | 1.688308 | -0.83% | -16.009 |
| `fixed_training_b1_t8192_bf16` / forward-compact | 1.696482 | 1.697689 | +0.07% | -0.307 |

The T8 opt-in paired comparison retains a +2.02% median wall difference and a +2.321 us paired-block median delta. Separate five-call T8 traces contain exactly five device-module events per executable and no external input-reshard modules. Their device medians are 0.042340 ms for main and 0.043153 ms for the candidate; these are new diagnostic compilations, not substitutions into the twenty-sample official device medians.

[Paired raw samples and input-format records](benchmarks/regression_guard_20261117/paired-wall-diagnostic/report.json), [all-device-module audit](benchmarks/regression_guard_20261117/paired-wall-diagnostic/device-module-audit.json), and [immutable raw trace/script archive](benchmarks/regression_guard_20261117/paired-wall-archive.json) retain the complete diagnostic.

## Raw evidence

Dedicated JobSet `kda1688-regression-1116`, UID `9677faf9-3040-4150-b734-2c7302995e6d`, us-east1. Completed phase archives contain raw samples, result arrays, HLO, profiles, input hashes and software/source pins. Their immutable GCS generations and checksums will be listed alongside the final tables. Bulky artifacts are excluded from the code PR.


## Artifact index

- [All phase archives with immutable GCS generations, size, CRC32C and SHA-256](benchmarks/regression_guard_20261117/restoration-archive-manifest.jsonl).
- [Final source/input/log archive proof](benchmarks/regression_guard_20261117/restoration-final-evidence-archive.json).
- [Published-source verification](benchmarks/regression_guard_20261117/restoration-source-published.json).
- [Device capture protocol](benchmarks/regression_guard_20261117/restoration-profile-protocol.json), including per-compilation sampling and unchanged earlier traces.
- [Official input manifest](benchmarks/regression_guard_20261117/inputs/manifest.json) and [separate compact feature input manifest](benchmarks/regression_guard_20261117/compact-feature-inputs/manifest.json).
- [Native selection/result](benchmarks/regression_guard_20261117/fallback-native-result.json) and [JUnit report](benchmarks/regression_guard_20261117/fallback-native.xml).

The code PR contains the source fixes, tests and design explanation. Performance reports and raw measurement records are published only on this evidence branch.

- [Independent NumPy recalculation of all 108 cells](benchmarks/regression_guard_20261117/final-independent-audit.json).
- [Post-measurement frozen-source audit](benchmarks/regression_guard_20261117/restoration-post-measurement-source-audit.json).
- [Cloud generation/size/CRC audit of all twenty immutable archives](benchmarks/regression_guard_20261117/final-cloud-archive-audit.json).
