# Copyright 2026 Ant Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CP saved-state and rematerialized local fusion across real shard_map collectives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadata
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu_fwd_fused_test as f
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu_bwd_kernel as kernels

interpret_on_cpu = f.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("state_policy", ["saved", "remat"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_megakernel(dtype, split, state_policy, cp_size):
  if jax.device_count() < cp_size:
    pytest.skip(
        "Requires two devices; CPU interpret uses XLA host device count=2"
    )
  mesh = Mesh(np.array(jax.devices()[:cp_size]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  tokens = 128 * cp_size
  keys = jax.random.split(jax.random.key(97), 3)
  q = (jax.random.normal(keys[0], (2, 2, tokens, 128)) * 0.05).astype(dtype)
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(dtype)
  v = jax.random.normal(keys[2], q.shape).astype(dtype)
  g = jnp.full_like(q, -0.01)
  beta = jnp.full(q.shape[:-1], 0.5, dtype)
  segments = (
      jnp.array(
          [[1] * 64 + [2] * (tokens - 64), [1] * (tokens - 64) + [2] * 64],
          jnp.int32,
      )
      if split
      else jnp.ones((2, tokens), jnp.int32)
  )
  kernels.chunk_kda_bwd_custom.clear_cache()
  results = []
  for fused in (False, True, "reference"):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            cp_megakernel=fused is True,
            rematerialize_for_backward=state_policy != "saved",
            fuse_rematerialization=state_policy == "remat",
        )
    )

    if fused == "reference":
      op = lambda *args, **kwargs: api.kimi_delta_attention(
          *args, **kwargs, implementation="xla"
      )

    def local(q, k, v, g, beta, segments):
      def forward(q, k, v, g, beta):
        return op(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=segments,
            max_num_segments=2,
            context_parallel_metadata=meta,
        )[0]

      out, back = jax.vjp(forward, q, k, v, g, beta)
      return back(jnp.ones_like(out))

    spec = P(None, None, "context", None)
    # Match the existing CP suite's batched-metadata annotation workaround.
    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      grads = jax.jit(
          jax.shard_map(
              local,
              mesh=mesh,
              in_specs=(spec,) * 4
              + (P(None, None, "context"), P(None, "context")),
              out_specs=(spec,) * 4 + (P(None, None, "context"),),
              check_vma=False,
          )
      )(q, k, v, g, beta, segments)
    results.append(grads)
  f._assert_close(
      results[1], results[0], tolerance=0.01 if dtype == jnp.bfloat16 else 1e-5
  )

  # The source fusion retains more FP32 intermediates than staged BF16.
  # Check both against the independent recurrent reference, not just each other.
  for gradients in results[:2]:
    f._assert_close(
        gradients, results[2], tolerance=0.02 if dtype == jnp.bfloat16 else 1e-5
    )
