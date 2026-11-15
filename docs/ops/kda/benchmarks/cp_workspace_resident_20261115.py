"""Frozen same-input CP experiment; prepare /workspace sources as documented.

Steady-state timing prepares compiled input sharding/layout before warmup.
Use --include-input-reshard only to reproduce the historical transfer-inclusive
scenario. Numeric input identities and all outputs/gradients are preserved.
"""

import argparse, dataclasses, gc, hashlib, importlib.metadata, json, os, pathlib, sys, time, traceback
from cp_workspace_matrix_20261115 import CASES, selected, profile_case

parser = argparse.ArgumentParser()
parser.add_argument(
    "--include-input-reshard",
    action="store_true",
    help="Reproduce historical transport-inclusive host timing",
)
parser.add_argument("--suite", required=True, choices=list(CASES))
parser.add_argument("--variant", required=True)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--cases")
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--output-root", default="/workspace/results")
parser.add_argument("--batch-first", action="store_true")
parser.add_argument("--profile-only", action="store_true")
args = parser.parse_args()
if args.profile_only:
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "")
      + " --xla_xprof_register_llo_debug_info=true --xla_enable_custom_call_region_trace=true"
  )
revision = (
    "internal"
    if args.variant.startswith("internal_")
    else args.variant.split("__")[0]
    if "__" in args.variant
    else "before"
    if args.variant.startswith("before_")
    else "candidate"
)
role = args.variant.split("__")[-1] if "__" in args.variant else args.variant

variant = (
    role.removeprefix("before_")
    .removeprefix("candidate_")
    .removeprefix("internal_")
)
sys.path.insert(0, "/workspace/candidate")
sys.path.insert(0, "/workspace/" + revision)
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src import jaxtyping
from jax.sharding import Mesh, PartitionSpec as P
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata

outdir = (
    pathlib.Path(args.output_root) / f"{args.suite}-{args.variant}-{args.round}"
)
outdir.parent.mkdir(parents=True, exist_ok=True)
outdir.mkdir(exist_ok=True)
report = {
    "suite": args.suite,
    "variant": args.variant,
    "round": args.round,
    "revision": revision,
    "sha": json.loads(pathlib.Path("/workspace/revisions.json").read_text())[
        revision
    ],
    "device_kind": jax.devices()[0].device_kind,
    "python": sys.version,
    "versions": {
        p: importlib.metadata.version(p)
        for p in [
            "jax",
            "jaxlib",
            "libtpu",
            "numpy",
            "flax",
            "qwix",
            "xprof",
            "pydantic",
            "typeguard",
        ]
    },
    "flags": {
        p: os.environ.get(p)
        for p in [
            "LIBTPU_INIT_ARGS",
            "XLA_FLAGS",
            "JAX_PLATFORMS",
            "PALLAS_INTERPRET",
        ]
    },
    "protocol": {
        "warmup": 5,
        "calls_per_round": 20,
        "rounds": 2,
        "timing": "synchronized host wall",
        "profile_calls": 20,
    },
    "cases": {},
}


def dump():
  (outdir / "report.json").write_text(json.dumps(report, indent=2))


def host_inputs(c):
  b, t, h, n = (c[k] for k in ["b", "t", "h", "n"])
  rng = np.random.default_rng(619)

  def normal(shape, std):
    return (rng.standard_normal(shape, dtype=np.float32) * std).astype(
        ml_dtypes.bfloat16
    )

  shape = (b, t, h, 128)
  q, k, v, g = (
      normal(shape, s).transpose(2, 0, 1, 3) for s in [0.05, 0.05, 0.05, 0.1]
  )
  if c["preactivated"]:
    g = np.full(q.shape, -0.01, ml_dtypes.bfloat16)
  logits = rng.standard_normal(shape[:-1], dtype=np.float32) * 0.1
  beta = (
      (1 / (1 + np.exp(-logits))).astype(ml_dtypes.bfloat16).transpose(2, 0, 1)
  )
  a = rng.uniform(-1, 0, h).astype(np.float32)
  bias = rng.uniform(-4, -3, h * 128).astype(np.float32)
  cot = normal(shape, 0.05).transpose(2, 0, 1, 3)
  if c["cp"] > 1:
    ids = np.zeros((b, t), np.int32)
    local = t // c["cp"]
    for rank in range(c["cp"]):
      start = rank * local
      ids[:, start : start + local // 2 + 1] = 2 * rank + 1
      ids[:, start + local // 2 + 1 : start + local] = 2 * rank + 3
    if c["preactivated"]:
      ids[:] = 1
      ids[:, 64:] = 2
    if c["over"]:
      ids[-1, -local + local // 4 : -local + local // 2 + 1] = 2 * c["cp"]
  else:
    lengths = np.full(n, t // n, np.int32)
    lengths[: t % n] += 1
    ids = np.stack(
        [
            np.repeat(
                np.arange(1, n + 1, dtype=np.int32), np.roll(lengths, row)
            )
            for row in range(b)
        ]
    )
  state = (
      (rng.standard_normal((b, n, h, 128, 128), dtype=np.float32) * 0.01)
      if c["state"]
      else np.zeros((0,), np.float32)
  )
  if c.get("batch_first"):
    q, k, v, g = (x.transpose(1, 2, 0, 3) for x in (q, k, v, g))
    beta = beta.transpose(1, 2, 0)
    cot = cot.transpose(1, 2, 0, 3)
  values = (q, k, v, g, beta, a, bias, ids, cot, state)
  sha = hashlib.sha256()
  for x in values:
    sha.update(str((x.shape, str(x.dtype))).encode())
    sha.update(np.ascontiguousarray(x).view(np.uint8))
  return values, sha.hexdigest()


def configuration(c):
  flags = dict(rematerialize_for_backward=c["remat"])
  if variant in ("main", "default") or args.suite == "inference":
    return mosaic.Config(**flags)
  flags.update(fuse_forward=True, packed_forward=True, packed_output=True)
  if args.suite == "backward" and variant != "forward":
    flags.update(
        fuse_backward=True,
        packed_backward=True,
        packed_gradients=True,
        fuse_rematerialization=True,
    )
  if variant == "compact":
    flags.update(compact_forward=True, compact_sorted_segments=True)
  if args.suite == "cp" and variant == "parent":
    flags.update(
        fuse_backward=True,
        fuse_rematerialization=True,
        packed_backward=True,
        packed_gradients=True,
    )
  elif args.suite == "cp":
    flags.update(
        fuse_backward=True,
        fuse_cp_backward=True,
        fuse_rematerialization=True,
        cp_megakernel=variant == "mega",
    )
  return mosaic.Config(**flags)


def make_call(c, config, meta=None):
  if revision == "internal":
    from tops.ops.kda.workspace import chunk_kda_cp
    from tops.ops.common.cp_utils import CPContext

    context = CPContext(mesh=meta.mesh, axis_name="context")

    def internal_call(xs, ids, state):
      q, k, v, g, b, a, d = xs
      q, k, v, g = (x.transpose(1, 2, 0, 3) for x in (q, k, v, g))
      b = b.transpose(1, 2, 0)
      y, _ = chunk_kda_cp(
          q,
          k,
          v,
          g,
          b,
          A_log=a,
          dt_bias=d,
          segment_ids=ids,
          cp_context=context,
          N_max=c["n"],
          fast_segment_capacity=2,
          head_chunk_size=2,
          use_qk_l2norm_in_kernel=c["norm"],
          use_gate_in_kernel=not c["preactivated"],
          lower_bound=None if c["preactivated"] else -5.0,
          disable_recompute=not c["remat"],
          apply_beta_sigmoid=False,
      )
      return y.transpose(2, 0, 1, 3), None

    return internal_call
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(config=config)

  def call(xs, ids, state):
    q, k, v, g, beta, a, bias = xs
    options = dict(
        a_log=a,
        delta_time_bias=bias,
        segment_ids=None if c["fixed"] else ids,
        max_num_segments=None if c["fixed"] else c["n"],
        use_gate_in_kernel=not c["preactivated"],
        lower_bound=None if c["preactivated"] else -5.0,
        use_qk_l2norm=c["norm"],
        initial_state=state if c["state"] else None,
        output_final_state=c["state"],
        context_parallel_metadata=meta,
    )
    if args.suite == "cp" and variant == "workspace":
      from tokamax._src.ops.kda.pallas_mosaic_tpu_workspace import kimi_delta_attention_cp

      return kimi_delta_attention_cp(
          q,
          k,
          v,
          g,
          beta,
          config=config,
          fast_segment_capacity=2 if c["n"] > 2 else 0,
          head_group_size=2,
          **options,
      )
    if args.suite == "inference" and variant == "native":
      from tokamax._src.ops.kda.inference import kimi_delta_attention_inference

      options.pop("context_parallel_metadata")
      options["use_qk_l2norm_in_kernel"] = options.pop("use_qk_l2norm")
      native_xs = (
          (q, k, v, g)
          if c.get("batch_first")
          else tuple(x.transpose(1, 2, 0, 3) for x in (q, k, v, g))
      )
      native_beta = beta if c.get("batch_first") else beta.transpose(1, 2, 0)
      out, final = kimi_delta_attention_inference(
          *native_xs, native_beta, **options
      )
      return out, final
    if args.suite == "inference" and c.get("batch_first"):
      q, k, v, g = (x.transpose(2, 0, 1, 3) for x in (q, k, v, g))
      beta = beta.transpose(2, 0, 1)
    out, final = op(q, k, v, g, beta, **options)
    return (
        out.transpose(1, 2, 0, 3) if args.suite == "inference" else out
    ), final

  return call


def save_arrays(path, value):
  arrays = {}
  dtypes = []
  for i, x in enumerate(jax.tree.leaves(value)):
    x = np.asarray(jax.device_get(x))
    assert np.isfinite(x.astype(np.float32)).all()
    dtypes.append(str(x.dtype))
    arrays["a" + str(i)] = (
        x.view(np.uint16) if x.dtype == ml_dtypes.bfloat16 else x
    )
  np.savez_compressed(path, **arrays)
  return dtypes, [list(x.shape) for x in arrays.values()]


def output_hash(value):
  digest = hashlib.sha256()
  for x in jax.tree.leaves(value):
    host = np.asarray(jax.device_get(x))
    digest.update(str((host.shape, str(host.dtype))).encode())
    digest.update(host.tobytes(order="C"))
  return digest.hexdigest()


for name, c in CASES[args.suite].items():
  if args.cases and name not in args.cases.split(","):
    continue
  if variant == "mega" and name != "cp4_mega":
    continue
  if variant != "mega" and name == "cp4_mega":
    continue
  if args.batch_first:
    assert args.suite == "inference"
    c = dict(c, batch_first=True)
  if args.pilot:
    c = dict(
        c,
        t=128 if c["cp"] == 1 else c["cp"] * 128,
        h=2,
        n=min(c["n"], 2 if c["cp"] == 1 else 32),
    )
    if c["over"]:
      c["n"] = 32
  entry = {"spec": c, "modes": {}}
  report["cases"][name] = entry
  print("CASE", args.suite, args.variant, args.round, name, c, flush=True)
  try:
    host, digest = host_inputs(c)
    entry["input_sha256"] = digest
    with jax.default_device(jax.devices("tpu")[0]):
      device = tuple(jax.device_put(x) for x in host)
    del host
    cfg = configuration(c)
    entry["config"] = dataclasses.asdict(cfg)
    mesh = (
        Mesh(np.array(jax.devices()[: c["cp"]]), ("context",))
        if c["cp"] > 1
        else None
    )
    meta = (
        ContextParallelMetadata(mesh=mesh, axis_name="context")
        if mesh
        else None
    )
    call = make_call(c, cfg, meta)
    modes = (
        ["full", "vjp"]
        if args.suite == "backward"
        else ["full"]
        if args.suite == "cp"
        else ["forward"]
    )
    for mode in modes:

      def run(*values):
        xs = values[:7]
        ids, cot, state = values[7:]
        if mode == "forward":
          return call(xs, ids, state)
        y, pb = jax.vjp(lambda *xs: call(xs, ids, state)[0], *xs)
        grads = pb(cot)
        if meta is not None:
          grads = (
              *grads[:5],
              jax.lax.psum(grads[5], "context"),
              jax.lax.psum(grads[6], "context"),
          )
        return y, grads

      run.__name__ = f"{args.suite}_{args.variant}_{name}_{mode}"
      with jaxtyping.disable_jaxtyping():
        if mesh:
          spec = P(None, None, "context", None)
          outputspec = (
              spec,
              (spec, spec, spec, spec, P(None, None, "context"), P(), P()),
          )
          with jax.set_mesh(mesh):
            fn = jax.jit(
                jax.shard_map(
                    run,
                    mesh=mesh,
                    in_specs=(spec,) * 4
                    + (
                        P(None, None, "context"),
                        P(),
                        P(),
                        P(None, "context"),
                        spec,
                        P(),
                    ),
                    out_specs=outputspec,
                    check_vma=False,
                )
            )
            executable = fn.lower(*device).compile()
          inputs = device
        elif mode == "vjp":

          def prepare(*values):
            xs = values[:7]
            ids = values[7]
            state = values[9]
            return jax.vjp(lambda *xs: call(xs, ids, state)[0], *xs)

          _, pb = jax.jit(prepare)(*device)
          jax.block_until_ready(pb)

          def backward(pb, dy):
            return pb(dy)

          backward.__name__ = run.__name__
          inputs = (pb, device[8])
          closed = jax.make_jaxpr(backward)(*inputs)
          sizes = [
              int(getattr(x, "size", 0))
              * int(getattr(getattr(x, "dtype", None), "itemsize", 0))
              for x in closed.consts
          ]
          entry["vjp_largest_closed_constant_bytes"] = max(sizes, default=0)
          assert (
              max(sizes, default=0) < 1024 * 1024
          ), "Residuals must be runtime inputs, not baked-in constants"
          executable = jax.jit(backward).lower(*inputs).compile()
          del closed
        else:
          inputs = device
          executable = jax.jit(run).lower(*inputs).compile()
      # Inputs must already have the executable's CP sharding and local
      # layout; repeated resharding is not part of operator timing.
      if mesh and not args.include_input_reshard:
        formats = executable.input_formats[0]
        inputs = tuple(
            jax.device_put(x, f) for x, f in zip(inputs, formats, strict=True)
        )
        jax.block_until_ready(inputs)
        entry["prepared_input_formats"] = [str(f) for f in formats]
        entry["input_placement"] = "compiled input Format, once before warmup"
      mem = executable.memory_analysis()
      details = {
          "memory": {
              k: int(getattr(mem, k))
              for k in [
                  "temp_size_in_bytes",
                  "argument_size_in_bytes",
                  "output_size_in_bytes",
                  "alias_size_in_bytes",
              ]
          }
      }
      (outdir / f"{name}-{mode}.hlo.txt").write_text(executable.as_text())
      for _ in range(5):
        jax.block_until_ready(executable(*inputs))
      samples = []
      for _ in range(0 if args.profile_only else 2 if args.pilot else 20):
        start = time.perf_counter_ns()
        jax.block_until_ready(executable(*inputs))
        samples.append((time.perf_counter_ns() - start) / 1e6)
      details.update(
          samples_ms=samples,
          median_ms=float(np.median(samples)) if samples else None,
      )
      actual = jax.block_until_ready(executable(*inputs))
      details["output_sha256"] = output_hash(actual)
      if args.round == 0 and not args.profile_only:
        details["array_dtypes"], details["array_shapes"] = save_arrays(
            outdir / f"{name}-{mode}.npz", actual
        )
      else:
        details["finite"] = all(
            bool(np.isfinite(np.asarray(x, dtype=np.float32)).all())
            for x in jax.tree.leaves(actual)
        )
        assert details["finite"]
      if args.profile_only:
        with jax.profiler.trace(str(outdir / f"profile-{name}-{mode}")):
          for step in range(20):
            with jax.profiler.StepTraceAnnotation(run.__name__, step_num=step):
              jax.block_until_ready(executable(*inputs))
      entry["modes"][mode] = details
      dump()
      print(
          "RESULT",
          name,
          mode,
          details["median_ms"],
          details["memory"],
          flush=True,
      )
      del executable, inputs, actual
      if mode == "vjp":
        del pb
      jax.clear_caches()
      gc.collect()
    del device
  except Exception:
    entry["error"] = traceback.format_exc()
    print(entry["error"], flush=True)
    dump()
  jax.clear_caches()
  gc.collect()
dump()
