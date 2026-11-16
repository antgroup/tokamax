# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Ported from primatrix/pallas-kernel dca6a09feec2659f2777a6db02f2ee55560f991e.

"""Device-side primitives for compact packed KDA chunks.

The production packed mega kernel currently keeps token inputs compact but
executes a sequence-aligned chunk grid.  A compact grid may contain multiple
sequence fragments in one chunk, so its scan and pairwise operations must be
segmented.  The helpers in this module use fixed-shape JAX operations and are
safe to trace from a Pallas kernel; they do not introduce a host-side loop over
the dynamic number of fragments.
"""

import jax
import jax.numpy as jnp

COMPACT_CHUNK_FULL = 0
COMPACT_CHUNK_TWO_FRAGMENT = 1
COMPACT_CHUNK_GENERAL = 2


def fixed_width_prefix_sum(values: jax.Array) -> jax.Array:
  """Inclusive prefix sum using the fixed-width tree used by KDA kernels.

  Keeping a continuous coordinate across packed segment boundaries is safe
  because all semantic contractions are masked to equal segment IDs and gate
  differences within one segment are translation invariant.  It also avoids
  a large positive jump when a segmented prefix is reset inside a chunk,
  which can overflow the blockwise Q/K exponential rescaling before the
  cross-segment entries are masked.
  """
  width = values.shape[-2]
  result = values
  shift = 1
  while shift < width:
    shifted = jnp.concatenate(
        [jnp.zeros_like(result[..., :shift, :]), result[..., :-shift, :]],
        axis=-2,
    )
    result = result + shifted
    shift *= 2
  return result


def same_segment_causal_mask(
    segment_ids: jax.Array, *, include_diagonal: bool = True
) -> jax.Array:
  """Return a per-chunk causal mask with cross-segment entries removed."""
  if segment_ids.ndim < 1:
    raise ValueError("segment_ids must have a token axis")
  width = segment_ids.shape[-1]
  row = jnp.arange(width, dtype=jnp.int32)[:, None]
  col = jnp.arange(width, dtype=jnp.int32)[None, :]
  causal = row >= col if include_diagonal else row > col
  # Expand int32 IDs before comparison. Mosaic cannot lower the equivalent
  # bool-vector shape cast from ``valid[..., :, None]`` on TPU.
  row_ids = segment_ids[..., :, None]
  col_ids = segment_ids[..., None, :]
  return (row_ids == col_ids) & (row_ids > 0) & (col_ids > 0) & causal
