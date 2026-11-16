# Default packing single-core refactor (2026-11-16)

## Provenance and audit correction

This is a pinned historical measurement record, not live PR status or a new
measurement of an arbitrary future main/head. The audit checked table values
against the retained JSON/CSV, units, shape/static options and baseline scope.
This correction changes explanations and labels only; measured values, raw
samples and numerical thresholds are unchanged. All evidence remains on this
separate branch rather than in the implementation PR.

Calculation commit `16c5afa1` identifies the packing changes. The executed
candidate was integrated with upstream main `0d3e82b4`; the retained source-hash
checks cover the measured calculation files. It is not a claim that the literal
16c5afa1 checkout alone contains all fifteen current ArgSpecs. The serialized
15-spec manifest comes from the frozen upstream main. The later restored branch
contains main's ArgSpec additions, short-convolution reference correction and
base tests; its calculation-kernel files are unchanged. Reference-file identity
must therefore be interpreted relative to that integrated main, not the older
standalone branch snapshot.

Calculation commit: `16c5afa11b6ace187fafd746b912d6e1dae6eabc`. Frozen upstream-main baseline: `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18` (verified before and after the sweep). Serialized input bytes are reused from the preceding complete sweep.

## Changes

- Local default training uses packing metadata/operand transforms, one KDA custom VJP, and output/state restoration. The large KDA tape is outside packing conditionals.
- Packing order is composed with existing alignment and unalignment gathers. The private map pair reuses the already computed inverse rather than reconstructing it at the kernel boundary.
- Canonical-layout validation checks adjacent IDs instead of performing a full prefix sum. Exhaustive exact checks cover 1296 short sequences including invalid IDs, gaps, repeated labels and padding.
- Short multi-state default decode evaluates convolution in the original token domain before packing. Convolution keeps original physical-lag semantics and parameter gradients. Long prefill, single-state decode, explicit opt-in fused/packed selectors, and cross-rank scheduling retain their existing implementations.
- No migrated optimization or correctness fix was reverted. Reference implementations and existing numerical tolerances are unchanged. The additional packing_order ABI is private to the Mosaic adapter.

## Validation

- Native broad regression: 243 passed (1436.99 s). After static dispatch refinements that restore existing optimized paths: 54 focused cases passed, then 8 final-candidate contract/parameter cases passed. Counts overlap; they are not summed. The final calculation files are SHA256-checked against the measured source.
- CPU Pallas compatibility and new gradient/state tests passed. New exact alignment/permutation checks and 1296-layout canonical checks pass; convolution parameter gradients additionally retain the existing 2% relative-RMS bound.
- Complete official shapes: all 15 ArgSpecs; 15 forward plus three full forward/VJP execution points. Three variants (main-default, candidate-default, existing opt-in), two reversed fresh-process sweeps, 10 warmups and 30 synchronized samples per cell per sweep: 3240 retained wall samples.
- All 54 profile points have two independent 5-call captures: 108 raw XPlanes/540 complete jit_invoke device-module events. Zero input-reshard modules. Input Formats are prepared once before warmup.
- Additional all-18-point same-process control interleaves main/candidate in 30 randomized palindromic blocks, 60 calls per implementation. The actual prepared input tuple is shared when compiled Formats match; details and paired intervals are retained.
- Endpoints are finite and repeat across reversed sweeps. Full numerical deltas and profile-output consistency are retained. No tolerance widening.

## Two regression cases

| Shape and mode | main wall ms | candidate wall ms | wall change | main device ms | candidate device ms | device change | paired wall change |
|---|---:|---:|---:|---:|---:|---:|---:|
| H32/B1/T8192/K=V128/N25 full forward+VJP | 7.636893 | 7.665196 | +0.37% | 7.220372 | 7.295590 | +1.04% | +0.58% |
| H6/B1/T64/K=V128/N64/W4 decode | 1.658673 | 1.295153 | -21.92% | 1.404823 | 1.057693 | -24.71% | -21.06% |

The original material packed-training regression (about +7.94% device) is reduced to about +1.04% device, +0.37% wall / +0.58% paired wall. This residual metadata cost remains measurable; zero regression is not claimed. Its paired mean interval is approximately +0.035 to +0.055 ms. H6 multi-state decode improves about 24.71% device and 21.92% wall, with the paired control also confirming improvement.

Compiler temporary buffers for packed full VJP are 2603707456 bytes on main and 2444280256 bytes on the candidate, compared with 3347874528 bytes before the refactor. Decode uses 183268320 bytes versus main 110852736 bytes: latency improves, but temporary memory is still above main. These are compiler-planned temporary bytes, not model/allocator peak memory.

## Complete wall/device matrix

All Q/K/V dimensions and static options are in the unchanged official input manifest linked from the preceding report. Layout is [H,B,T,K/V], B=1 and K=V=128. GW=1 for GDN, GW=128 otherwise. Inference cases have FP32 initial/final state and W4 convolution; training cases have no initial/final state.

| Official case / mode | main wall | default wall | delta | main device | default device | delta | opt-in wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_training_b1_t8192_bf16` / forward | 1.698996 | 1.678636 | -1.20% | 1.506759 | 1.507241 | +0.03% | 1.649024 |
| `fixed_training_b1_t8192_bf16` / forward_and_vjp | 4.018245 | 3.979024 | -0.98% | 3.726489 | 3.725099 | -0.04% | 3.728096 |
| `packed_training_b1_t8192_n25_bf16` / forward | 4.020265 | 3.778632 | -6.01% | 3.830969 | 3.604670 | -5.91% | 2.774972 |
| `packed_training_b1_t8192_n25_bf16` / forward_and_vjp | 7.636893 | 7.665196 | +0.37% | 7.220372 | 7.295590 | +1.04% | 7.699319 |
| `gdn_fixed_training_b1_t8192_bf16` / forward | 1.705698 | 1.680080 | -1.50% | 1.520535 | 1.519617 | -0.06% | 1.707504 |
| `gdn_fixed_training_b1_t8192_bf16` / forward_and_vjp | 3.354668 | 3.331128 | -0.70% | 3.092405 | 3.091416 | -0.03% | 3.356339 |
| `tp32_prefill_b1_t128_bf16` / forward | 0.213454 | 0.213525 | +0.03% | 0.017643 | 0.017875 | +1.31% | 0.220024 |
| `tp32_prefill_b1_t512_bf16` / forward | 0.247352 | 0.229246 | -7.32% | 0.034544 | 0.034299 | -0.71% | 0.253760 |
| `tp32_prefill_b1_t1024_bf16` / forward | 0.262630 | 0.262890 | +0.10% | 0.055438 | 0.055526 | +0.16% | 0.273362 |
| `tp32_verify_b1_t64_n8_bf16` / forward | 0.453395 | 0.368131 | -18.81% | 0.220044 | 0.160194 | -27.20% | 0.463757 |
| `tp16_prefill_b1_t1024_bf16` / forward | 0.292925 | 0.268971 | -8.18% | 0.069794 | 0.069786 | -0.01% | 0.289856 |
| `tp8_prefill_b1_t2048_bf16` / forward | 0.418262 | 0.403840 | -3.45% | 0.185550 | 0.185459 | -0.05% | 0.413082 |
| `tp32_decode_b1_t1_n1_bf16` / forward | 0.240965 | 0.213740 | -11.30% | 0.029220 | 0.026415 | -9.60% | 0.243433 |
| `tp32_dspark_decode_b1_t8_n1_bf16` / forward | 0.250766 | 0.238001 | -5.09% | 0.042850 | 0.043022 | +0.40% | 0.257855 |
| `tp32_dspark_decode_b1_t64_n16_bf16` / forward | 0.616894 | 0.506335 | -17.92% | 0.381759 | 0.277824 | -27.23% | 0.620756 |
| `tp32_decode_b1_t64_n64_bf16` / forward | 1.556733 | 1.229831 | -21.00% | 1.306559 | 0.997424 | -23.66% | 1.608604 |
| `tp16_dp2_dspark_decode_b1_t64_n16_bf16` / forward | 0.628618 | 0.521224 | -17.08% | 0.396735 | 0.294808 | -25.69% | 0.666984 |
| `tp16_dp2_decode_b1_t64_n64_bf16` / forward | 1.658673 | 1.295153 | -21.92% | 1.404823 | 1.057693 | -24.71% | 1.713587 |

## Numerical review and limitations

Default is bitwise equal to main at 15/18 execution points. Remaining differences are in short multi-state convolution profiles; maximum per-leaf relative RMS is below 1.4e-5 and maximum absolute difference is below 0.000275. Opt-in numerical behavior is unchanged (packed fused profiles retain their documented BF16 rounding differences). Full errors are in output-comparison.json. Model LM/MTP loss and end-to-end MaxText throughput are separate validations.

Opt-in paths retain the prior implementations. Their complete rows remain visible; no universal speedup across configurations is claimed. The recorded consumer audit at MaxText commit fd1446d5 routed KDA through TOPS; this is not a live claim about all later MaxText revisions. Review status is managed separately; this report asserts no live Ready status. The measured training metadata cost and decode memory tradeoff remain in the complete table.

## Reproduction and raw evidence

Data and full samples: `benchmarks/packing_single_call_20261116/`. The original complete-sweep input manifest and exact input hashes remain under `benchmarks/official_argspec_sync_20261115/manifest.json`. Scripts execute the standardizer forward/forward_and_vjp modes; full VJP uses the output as cotangent, whereas the separate paired control retains its own timing method. All matrix times are milliseconds; percentage deltas=(candidate/main-1)*100. Positive is longer latency.  beta=0.5 and seed 619 are unchanged. JAX/jaxlib 0.12.0, libtpu 0.0.50.1, TPU7x allocation of four chips/eight logical devices, timing on logical device0. Complete observed flags and versions are retained in environment.json.

Immutable raw evidence: `gs://data_ant/diagnostics/kda-packing-singlecall-20261116/7e64b520-0f78-4d90-b6a6-ce39fb8b40a4/evidence-16c5afa1-main0d3e82b4.tgz#1794765685858753`.
Bytes `4050423500`, SHA256 `cbfef06988b9124d7ccc23ce5fb79470f22fb88092c02128a9dbbf30ff4dcbd0`, CRC32C `nxcq4g==`. Streamed size, server CRC and SHA metadata verified. Raw accepted XPlanes/HLO, serialized inputs, final source and failed-experiment records are retained. Rejected-prototype tensor NPZ/profile payloads are excluded where their sample/HLO records suffice; accepted raw captures are complete.
