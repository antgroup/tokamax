# Compact all-gather CP: pinned-source review validation

## Provenance and audit correction

This is a pinned historical measurement record, not live PR status or a new
measurement of an arbitrary future main/head. The audit checked table values
against the retained JSON/CSV, units, shape/static options and baseline scope.
This correction changes explanations and labels only; measured values, raw
samples and numerical thresholds are unchanged. All evidence remains on this
separate branch rather than in the implementation PR.

The restored KDA implementation/tests at `3f18fc01bd1fea7e51caef6fb312fc59dc0fb861` match the original report snapshot `9b67d57c482c0cdb621b54d9162c82de5daf62f8`. Changes to those files in future require renewed validation.

Tested calculation commit `b2a3df47ff42f4f787a0e6e691efce2ad28db4ab`. Formal operator baseline: upstream main `0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18`.

## Feature and supported contract

The explicit compact entry bounds aligned workspace by the staged run cap and evaluates interior labels on the original token domain. It preserves global labels for boundary state exchange and does not change default operator dispatch.

BF16 Q/K/V, B=1, local T divisible by64, globally grouped positive labels with zero padding. Sequence continuation uses the existing aligned-CP contiguous owning-rank chain contract. Arbitrary cross-rank interleaving, or resuming through an entirely empty rank, is outside this contract. Initial/final states, convolution and in-kernel Q/K normalization are not supported.

Depends on the shared compact core in #1688/#1689 and aligned CP support in #1690. While parents are unmerged, this branch carries the dependency code. The feature-specific review focuses on cp_compact.py, its tests and design document.

## Native acceptance

- CP4/CP8 fixed, split, padding, empty-rank and all-padding fixtures; H64 large/local1024/local4096, manual recompute and nested scan/remat; global256K/H8/CP8.
- Sixteen-device, two-process TPU7x validation: H4 small, H64 nested scan/remat, H64/global256K, randomized cross-rank packing, and the independent-reference anchor all pass. The shared gate-parameter regression also passes independently on both processes.
- New random packing uses an independent grouped FP32 recurrence with highest-precision dots. A separate small test anchors all its gradients to the existing plain-JAX operator under highest matmul precision. The original aligned-comparison tests and their2% bounds remain unchanged.
- The inherited M5 parameter fix keeps FP32 intra gate cotangents for A_log/bias and preserves the former BF16 public token-gradient rounding. Old code fails the new packed regression; corrected code passes the existing norm and elementwise budgets. See #1689 for the complete same-input investigation.

## H64/256K performance and compiler memory

Dedicated TPU7x allocation: eight physical chips / sixteen JAX devices across two processes. B1/H64/globalT262144/localT16384/K=V128. BF16 tokens/raw gate, FP32 A_log=-0.5 and bias=-3.5, beta=0.5, raw-gate lower_bound=-5, Q/K normalization disabled. NumPy seed 619 and fixed Gaussian dO. Labels exceed262100; the middle rank is padding. Identical logical primals and dO are reused across routes; each route is placed into its own compiled input Formats before timing.

Ten warmups and thirty blocking calls per implementation. Host time is the median of thirty per-call maxima across the two hosts; it is not the maximum of two host medians. Two independent five-call captures give complete-module device time, taking the slowest of all sixteen ranks per call; zero input reshards. These whole-call operator measurements include forward and all seven gradients.

| Profile | Route | Host wall ms | Full-device max-rank ms | Compiler temporary GiB |
|---|---|---:|---:|---:|
| forced split, cap2 | reference_capacity | 54.957605 | 53.030262 | 10.8654 |
| forced split, cap2 | compact | 71.345911 | 69.123272 | 11.0986 |
| default entry, cap64 | reference_capacity | 54.788780 | 53.047394 | 10.8654 |
| default entry, cap64 | compact | 77.737208 | 75.718477 | 13.1809 |

The reference route is a manually bounded aligned parent using local capacity32, not a full-global-capacity upstream-main measurement. Neither CP route in this table establishes a speedup against upstream main. The 0d3e82b4 baseline applies only to the separate formal default-operator sweep. This comparison shows the extra work of the compact composition against an already bounded fast control. The feature provides capacity-independent coverage and memory fit; it is not claimed to be universally faster than a hand-sized aligned route. Compiler temporary bytes are not allocator/model training peak memory. Earlier lower_bound=-1 probes are retained as separate validation and are not substituted for the -5 table above.

## Complete formal default-operator sweep

All 15 official ArgSpecs /18 execution points use identical serialized primals, seed 619 and beta=0.5. Full-VJP cotangents follow the standardizer output-derived convention. Two reversed fresh-process sweeps retain sixty wall samples per point; each has two five-call complete jit_invoke captures. This sweep checks the shared default API separately from the explicit compact CP entry. Formal wall/device values are milliseconds; Change=(PR wall/main wall-1)*100%, positive means longer latency. CP compiler memory uses GiB (2**30 bytes) and is not allocator/model peak memory.

| Case / mode | Main wall | PR wall | Change | Main device | PR device |
|---|---:|---:|---:|---:|---:|
| fixed_training_b1_t8192_bf16 / forward | 1.696256 | 1.683414 | -0.76% | 1.506584 | 1.506465 |
| fixed_training_b1_t8192_bf16 / forward_and_vjp | 4.019903 | 3.988564 | -0.78% | 3.725200 | 3.726515 |
| packed_training_b1_t8192_n25_bf16 / forward | 4.039549 | 3.802618 | -5.87% | 3.825856 | 3.603373 |
| packed_training_b1_t8192_n25_bf16 / forward_and_vjp | 7.615922 | 7.681928 | +0.87% | 7.215888 | 7.291732 |
| gdn_fixed_training_b1_t8192_bf16 / forward | 1.729614 | 1.703526 | -1.51% | 1.520061 | 1.521067 |
| gdn_fixed_training_b1_t8192_bf16 / forward_and_vjp | 3.371878 | 3.358696 | -0.39% | 3.093274 | 3.094252 |
| tp32_prefill_b1_t128_bf16 / forward | 0.231913 | 0.221799 | -4.36% | 0.017728 | 0.017598 |
| tp32_prefill_b1_t512_bf16 / forward | 0.245647 | 0.251487 | +2.38% | 0.034594 | 0.034529 |
| tp32_prefill_b1_t1024_bf16 / forward | 0.292392 | 0.273520 | -6.45% | 0.055379 | 0.055471 |
| tp32_verify_b1_t64_n8_bf16 / forward | 0.477208 | 0.397185 | -16.77% | 0.219741 | 0.160444 |
| tp16_prefill_b1_t1024_bf16 / forward | 0.291921 | 0.283991 | -2.72% | 0.069692 | 0.069766 |
| tp8_prefill_b1_t2048_bf16 / forward | 0.428630 | 0.423542 | -1.19% | 0.185579 | 0.185451 |
| tp32_decode_b1_t1_n1_bf16 / forward | 0.255163 | 0.222403 | -12.84% | 0.029384 | 0.026643 |
| tp32_dspark_decode_b1_t8_n1_bf16 / forward | 0.259823 | 0.263877 | +1.56% | 0.043058 | 0.042950 |
| tp32_dspark_decode_b1_t64_n16_bf16 / forward | 0.645495 | 0.511226 | -20.80% | 0.381563 | 0.277516 |
| tp32_decode_b1_t64_n64_bf16 / forward | 1.563943 | 1.243945 | -20.46% | 1.304905 | 0.997440 |
| tp16_dp2_dspark_decode_b1_t64_n16_bf16 / forward | 0.655948 | 0.530636 | -19.10% | 0.396683 | 0.294797 |
| tp16_dp2_decode_b1_t64_n64_bf16 / forward | 1.669431 | 1.305495 | -21.80% | 1.412362 | 1.057867 |

Default formal outputs match main bitwise at 15/18 points; the short multi-state convolution differences have maximum per-leaf relative RMS below 1.4e-5. Both sweeps reproduce endpoint hashes. No reference module or prior numerical tolerance was weakened.

## Evidence and reproduction

JAX/jaxlib 0.12.0, libtpu 0.0.50.1; large second-minor layout enabled. Full shape/options, host samples, module events, source fingerprints and immutable archive generations are retained in `benchmarks/review_20261116/`. Native tests run with `python -m pytest -q tokamax/_src/ops/kda/cp_compact_test.py` and the direct shared gate-parameter regression. CP16 requires two initialized processes with eight local devices each.

The CP profile archive index includes per-route checkpoints, final profile parsing and numerical controls, with verified size/CRC32C/SHA256. A first CP16 allocation was preempted; final acceptance and tables use the subsequently reallocated, immutably archived run. Failed fixture/diagnostic observations are retained and are not acceptance results.


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
