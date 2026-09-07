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

"""Device-time comparison of staged and fused CP saved-state and rematerialized backward.

Run on at least two TPU devices: python -m tokamax.benchmarks.kda_cp
"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import cp_utils
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as mosaic


class KdaCpBenchmark(parameterized.TestCase):

  @parameterized.product(
      cp_size=(2, 4), fused=(False, True), remat=(False, True)
  )
  def test_forward_and_vjp(self, cp_size, fused, remat):
    if jax.default_backend() != "tpu" or jax.device_count() < cp_size:
      self.skipTest("Requires sufficient TPU devices")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:cp_size]), ("context",))
    meta = cp_utils.ContextParallelMetadata(mesh=mesh, axis_name="context")
    spec = jax.sharding.PartitionSpec(None, None, "context", None)
    beta_spec = jax.sharding.PartitionSpec(None, None, "context")
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_cp_backward=fused,
            rematerialize_for_backward=remat,
            fuse_rematerialization=fused and remat,
        )
    )

    def local(q, k, v, g, beta):
      def forward(q, k, v, g, beta):
        return op(q, k, v, g, beta, context_parallel_metadata=meta)[0]

      output, backward = jax.vjp(forward, q, k, v, g, beta)
      return output, backward(jnp.ones_like(output))

    fn = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(spec,) * 4 + (beta_spec,),
            out_specs=(spec, (spec,) * 4 + (beta_spec,)),
            check_vma=False,
        )
    )
    shape = (32, 1, 8192, 128)
    keys = jax.random.split(jax.random.key(97), 3)
    q, k, v = [jax.random.normal(key, shape, jnp.bfloat16) for key in keys]
    args = (
        q * 0.05,
        k * 0.05,
        v,
        jnp.full_like(q, -0.01),
        jnp.full(shape[:-1], 0.5, jnp.bfloat16),
    )
    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      result = tokamax.benchmark(fn, args)
    logging.info(
        "device_kind=%s cp_size=%s fuse_cp_backward=%s remat=%s median_time_ms=%s",
        jax.devices()[0].device_kind,
        cp_size,
        fused,
        remat,
        result.median_evaluation_time_ms,
    )


if __name__ == "__main__":
  absltest.main()
