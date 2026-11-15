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
"""Packing correction around the existing KDA custom VJP.

Canonicalization preserves Tokamax's per-label recurrence. Convolution is
performed at original token positions before gathering, so padding does not
change its lag semantics. Common canonical layouts retain the fused path.
"""

import dataclasses
from typing import Annotated, Any
import jax
import jax.numpy as jnp
import pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PackingResiduals:
  pullback: Annotated[Any, pydantic.PlainSerializer(
      lambda _: None, return_type=type(None), when_used="json"
  )]


ARRAY_NAMES = (
    "query", "key", "value", "gate", "beta", "a_log", "delta_time_bias",
    "initial_state", "conv_weight_q", "conv_weight_k", "conv_weight_v",
    "conv_bias_q", "conv_bias_k", "conv_bias_v",
)


def _take(x, order):
  index = order[None, ...]
  if x.ndim == 4:
    index = index[..., None]
  return jnp.take_along_axis(x, index, axis=2)


def _conv(x, weight, bias, ids):
  width = weight.shape[1]
  x_f, w_f = x.astype(jnp.float32), weight.astype(jnp.float32)
  acc = x_f * w_f[:, None, width - 1:width, :]
  for lag in range(1, width):
    shifted = jnp.pad(x_f[:, :, :-lag], ((0, 0), (0, 0), (lag, 0), (0, 0)))
    same = (ids[:, lag:] == ids[:, :-lag]) & (ids[:, lag:] > 0)
    same = jnp.pad(same, ((0, 0), (lag, 0)))
    shifted = jnp.where(same[None, ..., None], shifted, 0)
    acc = acc + shifted * w_f[:, None, width - 1 - lag:width - lag, :]
  if bias is not None:
    acc = acc + bias.astype(jnp.float32)[:, None, None, :]
  return jax.nn.silu(acc).astype(x.dtype)


def invoke(op, config, arguments):
  """Return corrected output and a compositional, shape-safe pullback tape."""
  original = dict(arguments)
  return_residuals = original.pop("return_residuals")
  ids = original["segment_ids"]
  if ids.ndim == 1:
    ids = jnp.broadcast_to(ids[None], original["query"].shape[1:3])
  if ids.ndim != 2:
    raise ValueError("segment_ids must have shape [B, T] or [T]")
  batch, tokens = ids.shape
  context = original["context_parallel_metadata"]
  cp = context is not None and context.is_cp_enabled
  h0 = original["initial_state"]
  capacity = original["max_num_segments"]
  if h0 is not None:
    if h0.ndim != 5:
      raise ValueError("Packed initial_state must have shape [B,N,H,K,V]")
    if capacity is not None and capacity != h0.shape[1]:
      raise ValueError("initial_state state count must match max_num_segments")
    capacity = h0.shape[1]
  if capacity is None:
    raise ValueError("max_num_segments is required for packed inputs without state")
  if capacity <= 0:
    raise ValueError("max_num_segments must be positive")
  if cp and (h0 is not None or original["output_final_state"]):
    raise ValueError("CP does not support initial/final states")
  valid = ids > 0
  if not cp:
    valid = valid & (ids <= capacity)
  clean_ids = jnp.where(valid, ids, 0)
  iota = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32), ids.shape)
  if cp:
    needs = jnp.any(valid[:, 1:] & ~valid[:, :-1]) | jnp.any(ids < 0)
    needs = jax.lax.pmax(needs.astype(jnp.int32), context.axis_name) != 0
  else:
    keys = jnp.where(valid, ids.astype(jnp.uint32), jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32))
    starts = valid & (clean_ids != jnp.pad(clean_ids[:, :-1], ((0,0),(1,0))))
    dense = jnp.where(valid, jnp.cumsum(starts, axis=1), 0)
    needs = jnp.any(keys[:, 1:] < keys[:, :-1]) | jnp.any(clean_ids != dense)
    needs = needs | jnp.any(ids != clean_ids)
  inner_cfg = dataclasses.replace(config, canonicalize_packed_inputs=False)
  inner = type(op)(config=inner_cfg, bypass_device_check=op.bypass_device_check)
  if not isinstance(needs, jax.core.Tracer) and not bool(needs):
    return inner(**original, return_residuals=return_residuals) if return_residuals else (inner(**original), None)
  static = {k:v for k,v in original.items() if k not in ARRAY_NAMES}
  if cp:
    static["max_num_segments"] = min(capacity, tokens)

  def composed(*values):
    def unchanged(xs):
      options = {**static, **dict(zip(ARRAY_NAMES, xs, strict=True))}
      y, state = inner(**options)
      return jnp.where(valid[None, ..., None], y, 0), state

    def corrected(xs):
      params = dict(zip(ARRAY_NAMES, xs, strict=True))
      opts = {**static, **params}
      q, k, v = (opts[n] for n in ("query", "key", "value"))
      if opts["use_conv1d_in_kernel"]:
        q, k, v = (
            _conv(x, opts["conv_weight_"+n], opts["conv_bias_"+n], ids)
            for n,x in zip(("q","k","v"),(q,k,v),strict=True)
        )
        opts["use_conv1d_in_kernel"] = False
        for n in ("q","k","v"):
          opts["conv_weight_"+n] = None
          opts["conv_bias_"+n] = None
      if cp:
        count = jnp.sum(valid, axis=1, keepdims=True)
        inverse = jnp.where(valid, jnp.cumsum(valid,axis=1)-1,
                            count+jnp.cumsum(~valid,axis=1)-1).astype(jnp.int32)
        order = jnp.zeros_like(inverse).at[jnp.arange(batch)[:,None],inverse].set(iota)
      else:
        order = jnp.argsort(keys, axis=1, stable=True).astype(jnp.int32)
        inverse = jnp.zeros_like(order).at[jnp.arange(batch)[:,None],order].set(iota)
      packed_ids = jnp.take_along_axis(clean_ids, order, axis=1)
      for n,x in zip(("query","key","value","gate","beta"),
                     (q,k,v,opts["gate"],opts["beta"]),strict=True):
        opts[n] = _take(x,order)
      if not cp:
        positive = packed_ids > 0
        starts = positive & (packed_ids != jnp.pad(packed_ids[:,:-1],((0,0),(1,0))))
        opts["segment_ids"] = jnp.where(positive,jnp.cumsum(starts,axis=1),0)
        physical = min(capacity,tokens)
        opts["max_num_segments"] = physical
        positions = jax.vmap(lambda row:jnp.nonzero(row,size=physical,fill_value=tokens)[0])(starts)
        labels = jnp.where(positions < tokens, jnp.take_along_axis(packed_ids,jnp.minimum(positions,tokens-1),axis=1), 0)
        if params["initial_state"] is not None:
          state_index=jnp.maximum(labels-1,0)[...,None,None,None]
          opts["initial_state"]=jnp.take_along_axis(params["initial_state"],state_index,axis=1)
          opts["initial_state"]=jnp.where((labels>0)[...,None,None,None],opts["initial_state"],0)
      else:
        opts["segment_ids"] = packed_ids
      y,state = inner(**opts)
      y = _take(y,inverse)
      y = jnp.where(valid[None,...,None],y,0)
      if state is not None:
        logical = (jnp.zeros((batch,capacity)+state.shape[2:],state.dtype)
                   if params["initial_state"] is None else params["initial_state"].astype(state.dtype))
        dest = jnp.where(labels>0,labels-1,capacity)
        state = logical.at[jnp.arange(batch)[:,None],dest].set(state,mode="drop")
      return y,state

    return jax.lax.cond(needs, corrected, unchanged, values)

  values = tuple(original[n] for n in ARRAY_NAMES)
  if return_residuals:
    output,pullback = jax.vjp(composed,*values)
    return output,PackingResiduals(pullback)
  return composed(*values),None


def backward(residuals, cotangent, segment_ids):
  grads = residuals.pullback(cotangent)
  result = {name:value for name,value in zip(ARRAY_NAMES,grads,strict=True)
            if value is not None}
  result["segment_ids"] = jnp.zeros_like(segment_ids)
  return result,None
