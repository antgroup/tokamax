# KDA internal-main migration ledger (2026-11-15)

Source main: `18ea6891f0e8d9affe9e81815267f5251bdb0b3b`.
Original native-kernel port: `dca6a09feec2659f2777a6db02f2ee55560f991e` (2026-10-26).
This is a pinned audit, not a claim about unmerged internal PRs or later commits.
Status at the November 15 audit: all four existing PRs were Draft. Current validation is recorded in [the final-source report](review_validation_20261116.md). Compact all-gather CP is a separate feature.

## Changes since the original port

`git diff dca6a09f 18ea6891 -- tops/ops/kda tops/ops/common tops/utils.py`
contains five changed KDA files and no changes to those shared dependencies.
`git log` identifies exactly three KDA commits in this interval:

| Internal commit | Change | Destination | Validation |
|---|---|---|---|
| `6bd65934` (#677) | Compact interior padding before aligned/CP calls, restore output/cotangent/gradient rows | #1688/#1689/#1690 common `packing.py`; #1691 standalone inference | Original fused/packed/remat suites plus new holes, leading padding, repeated labels, empty-batch state/VJP cases |
| `6bd65934` (#677) | Native chunk first/last valid token; general fragment routing; zero padding gate decay | #1688/#1689 shared `compact_forward.py`; #1691 ports metadata into its state-safe inference core | Existing compact/reference suite and inference padding/many-fragment tests |
| `6bd65934` (#677) | Backward metadata first/last valid token and general path for non-tail padding | #1689 `compact_backward.py`; inherited by separate compact-CP Draft | Existing compact backward full gradient suite |
| `3c1f14a5` (#699) | Expose batch-first `[H/MB, MB]` input tile to avoid Mosaic E2002 | #1688/#1689 `compact_forward.py`; ported into #1691; inherited by compact-CP Draft | Native multi-head forward/inference and shared-consumer/CP lowering |
| `993692c3` (#687) | Capacity-free compact all-gather CP, aligned boundary + native interior VJPs | Separate Draft `cp_compact.py`; excluded from #1690 | CP4/CP8 within-cap/split/padding output and seven-gradient tests; full performance acceptance pending |

## Required semantic adaptation

Tokamax's unchanged reference indexes recurrent state by label, including repeated
labels. Internal `chunk_kda` instead assigns contiguous-run identities. Copying
`relabel_padding_split_runs` into Tokamax would change its existing API. The port
stably groups each label, gathers the corresponding initial-state slots and restores
both token order and logical final-state slots. Padding and missing states retain
zero/identity derivatives respectively. Convolution is evaluated at original token
positions before gathering so its physical lag semantics remain unchanged.
The native compact path preserves the original token positions for grouped
labels, including leading/interior padding; only interleaved labels require
permutation. Its compact residual contract is preserved.

Compact CP requires globally grouped labels (the existing boundary-chain packing
contract), BF16, B=1, local T divisible by 64, and no explicit state, convolution or
Q/K normalization. It does not claim arbitrary repeated labels across remote ranks.
These limits must remain visible in the new Draft until independently extended.

A discovered custom-VJP dtype issue is corrected at the boundary: cotangents match
their primal dtype, including BF16 initial state. The empty-state analytic assertion
uses exact equality to a dtype-matched 0.1, instead of comparing BF16 against an
unrepresentable Python float. No numerical tolerances or reference arithmetic change.

## Baseline optimization coverage and ownership

| Internal modules / optimization | Existing owner / corresponding Tokamax implementation |
|---|---|
| `chunk_fwd_mega`, `chunk_intra_fwd_fused`: fused forward and triangular solve | #1688 fused forward kernel; packed input/output switches |
| `forward_inference_kernel`, `packed_segmented`: original-token compact forward, per-token fragments, saved compact tape, metadata prefetch | #1688 shared compact core; #1689 consumes its tape; #1691 retains its state-safe forward-only core |
| `chunk_bwd_mega`, `chunk_bwd_fusion`, `chunk_intra`: saved-state backward, packed/compact gradients, fused recomputation | #1689 compact and optimized reverse kernels; CP-specific orchestration #1690 |
| `chunk_bwd`, `chunk_fwd_h_o`: cross-rank state exchange and local reverse merge | Existing CP primitives plus #1690 fused CP reverse selectors |
| `workspace`, `head_groups`: capacity buckets and sequential head groups | #1690 opt-in workspace adapter; no default feature dispatch added |
| `gate`, `wy_fast`: gate math, bounded activation, FP32 accumulation and WY transforms | #1688 forward/common; #1689 reverse/compact gate math; #1690 CP precision path |
| `varlen`, `packed_chunk_plan`: segment metadata, align/unalign, original-token chunk plans | Existing common helpers and compact metadata; #677 adaptation above |
| `forward_inference`: standalone safe fused gate / norm prefill | #1691; state-safe native core with full per-label states, no training VJP import |
| `chunk.py`: input checks, layout, source dispatch and residual contract | Explicit Tokamax Config switches and existing public Op/VJP boundary; internal environment flags are not a public API requirement |
| `naive`, `fused_recurrent`: CPU/scan references and segmented fallback | Tokamax's independent existing reference is retained; no reference replacement to make tests pass |
| `short_conv`: standalone short-conv/gated-RMSNorm APIs surrounding KDA | Not silently counted as part of the four core KDA PRs. Upstream already owns convolution fusion; standalone APIs/production tile policies require a separate scope audit and are recorded as outside this recent three-commit delta |
| `__init__`: internal exports | Tokamax preserves its own API registration and naming |

The machine-readable companion inventories all 24 internal KDA source files, their
Git blob IDs, SHA256 and top-level symbols. It is an omission checklist, not evidence
that differently structured implementations are byte-identical.

## Acceptance and performance evidence

Only final-source passing tests may authorize commit/push. Preserve failed runs as
well. No tolerance widening. The earlier 15-official-ArgSpec sweep on #1690 head
`35c7c850` versus main `0d3e82b4` is historical evidence: it predates these correctness
changes and cannot establish final migrated-version default-path performance.
A final official ArgSpec comparison remains required before Ready for Review.

## Additional safety finding during final migration validation

Preserving original token positions exposed a leading-full-padding-chunk carry
initialization gap in the native source. The pinned internal main and the former
upstream core leave recurrent scratch/previous-segment state undefined when the
first chunk is entirely padding. The upstream fix initializes the chain before any
chunk, zeros saved h for all-padding chunks, and never saves a nonexistent segment
ID 0 as a final state. A native scratch-poison regression retains the existing
reference bounds. With identical H2/B1/T128/K=V128 leading-padding input and the
same NaN/ID-1 scratch poison, internal main produced 16,384 nonfinite outputs;
the corrected Tokamax core produced zero. This is a deterministic scratch-safety
control, not a claim to reproduce a production training failure. Internal main was
kept immutable as the comparison baseline; this additional fix is in the upstream
branches and recorded separately from its three recent commits.

Grouped native inputs retain physical positions across record padding. A label
resuming after padding, or interleaved labels, still uses the stable grouping
fallback to preserve Tokamax label recurrence. Inference retains its state-safe
standalone implementation. Short decode convolution ignores nonexistent history,
matching latest main for T smaller than the convolution width.
