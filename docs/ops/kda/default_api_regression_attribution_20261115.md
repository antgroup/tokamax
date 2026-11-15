# Attribution of the Tokamax default API regressions

## Result and consumer scope

The two regressions in the complete 15-ArgSpec report are relative to upstream Tokamax main `0d3e82b4`. They are localized to the new Tokamax packing adapter; they do not establish a slowdown of internal pallas-kernel main.

MaxText main `fd1446d5` calls TOPS KDA (`chunk_kda`, `chunk_kda_cp`, or `chunk_kda_cp_compact`). It does not call this Tokamax default adapter. The packed benchmark is nevertheless a genuine training-type operator workload, so direct Tokamax default training callers can encounter the overhead.

## Controlled ablation

Dedicated TPU7x, same pinned Python image/JAX0.12.0/libtpu0.0.50.1 environment and the exact serialized inputs from the complete sweep. Two fresh-process sweeps in opposite variant order; 10 warmups and 30 synchronized resident-input samples per round, 60 per cell. These are diagnostic wall-time measurements, separate from the prior full-device capture matrix.

The raw control changes only `canonicalize_packed_inputs=False`. Both selected inputs are checked to have dense, sorted, canonical segment IDs and no interior padding. The same migrated Pallas computational kernels and their correctness fixes remain enabled. The bypass is a diagnostic control only; it is not a proposed release fix for arbitrary packing.

| Case | Upstream main ms | Current adapter ms | Raw diagnostic ms |
|---|---:|---:|---:|
| `packed_training_b1_t8192_n25_bf16` / forward_and_vjp | 7.624926 | 8.239584 | 7.599663 |
| `tp16_dp2_decode_b1_t64_n64_bf16` / forward | 1.671189 | 1.762987 | 1.669556 |

Removing only the adapter restores baseline-level time. The packed full-VJP compiler temporary size drops from 3,347,874,528 to 2,603,762,688 bytes (upstream main: 2,603,707,456). Decode drops from 193,452,160 to 110,852,736 bytes, equal to main.

The adapter wraps whole operator calls in a packing conditional and differentiates that composite. Branch-residual/workspace planning and state-restoration/layout handling add cost even for ordinary canonical data. The ablation locates the causal layer; it does not claim a complete per-instruction breakdown or that both branches execute on every invocation.

## Internal pallas-kernel before/main control

Frozen internal versions: before `dca6a09f` and main `18ea6891`. Identical seed-619 H32/B1/T8192/K=V128/N25 input arrays, full forward plus seven-gradient pullback. Same allocation/environment, reversed fresh-process sweeps, 60 synchronized samples per cell.

| Internal route | Before ms | Main ms | Change |
|---|---:|---:|---:|
| `norm_inside` | 5.424323 | 5.400189 | -0.44% |
| `maxtext_like_norm_outside` | 4.396677 | 4.241968 | -3.52% |

`norm_inside` uses internal Q/K normalization with preactivated beta. `maxtext_like_norm_outside` normalizes Q/K outside, enables fused beta sigmoid, and keeps saved-state training, matching the component switches of current MaxText. Each route compares its own identical inputs/options across the two internal versions; absolute times across different routes are not a model throughput comparison.

These internal controls show no matching regression in the tested training cases. They do not certify every internal shape or an end-to-end MaxText training run. The H6/T64/N64/W4 case is a stateful inference/decode profile, rather than current MaxText training.

## Follow-up and publication decision

The generic Tokamax adapter regression remains disclosed for upstream review and later optimization. At the author's request it is not treated as a current MaxText delivery blocker. No computational-kernel/correctness rollback or tolerance widening was applied.

A selected-tape prototype passed the 16 padding checks but did not improve these performance measurements; it is retained as a rejected local experiment and was not committed to the PR.

Raw diagnostic scripts, HLO, input hashes, sample arrays and the rejected patch are archived (NPZ inputs are reused from the earlier immutable complete-sweep archive): `gs://data_ant/diagnostics/kda-default-perf-attribution-20261115/5966f48d-213d-470d-91ef-75aaa9f384f6/diagnosis.tgz#1794755936425675`.
Archive bytes `5382662`, SHA256 `57137b8b6535434452bdcc8adb54ed189281afe58585bc3582a7673b6c73a2ff`, CRC32C `bL2Guw==`; streamed size/server checksum/metadata SHA verified. Endpoint tensor NPZ files are excluded from this small diagnostic archive; its scripts and the original input archive reproduce them.
