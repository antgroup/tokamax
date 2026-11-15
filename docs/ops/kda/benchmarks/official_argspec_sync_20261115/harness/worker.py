import sys, os, json, pathlib, time, hashlib, traceback, gc, importlib.metadata

variant = sys.argv[1]
round = int(sys.argv[2])
source = {
    "main-default": "main",
    "head-default": "pr1690",
    "merge-default": "merge",
    "merge-opt": "merge",
}[variant]
sys.path.insert(0, "/workspace/official-sync/" + source)
import jax, jax.numpy as jnp, numpy as np, ml_dtypes
import tokamax
from tokamax._src.ops.kda import pallas_mosaic_tpu as m

root = pathlib.Path("/workspace/official-sync/results") / (
    variant + "-" + str(round)
)
root.mkdir(parents=True, exist_ok=True)
manifest = json.loads(
    pathlib.Path("/workspace/official-sync/inputs/manifest.json").read_text()
)
report = {
    "variant": variant,
    "round": round,
    "source": source,
    "cases": {},
    "device": str(jax.devices("tpu")[0]),
    "device_kind": jax.devices("tpu")[0].device_kind,
    "versions": {
        k: importlib.metadata.version(k)
        for k in ["jax", "jaxlib", "libtpu", "numpy", "flax", "qwix", "xprof"]
    },
    "python": sys.version,
    "flags": {
        k: os.environ.get(k)
        for k in ["LIBTPU_INIT_ARGS", "XLA_FLAGS", "PALLAS_INTERPRET"]
    },
}


def dump():
  (root / "report.json").write_text(json.dumps(report, indent=2))


for meta in manifest:
  name = meta["name"]
  report["cases"][name] = {"spec": meta, "modes": {}}
  entry = report["cases"][name]
  for mode in ["forward"] + (
      [] if "forward_only" in meta["tags"] else ["forward_and_vjp"]
  ):
    print("START", variant, round, name, mode, flush=True)
    try:
      with jax.default_device(jax.devices("tpu")[0]):
        raw = np.load("/workspace/official-sync/inputs/" + name + ".npz")
        kw = dict(meta["static"])
        digest = hashlib.sha256()
        for key, info in sorted(meta["arrays"].items()):
          x = (
              raw[key].view(ml_dtypes.bfloat16)
              if info["dtype"] == "bfloat16"
              else raw[key]
          )
          digest.update(key.encode())
          digest.update(str((x.shape, str(x.dtype))).encode())
          digest.update(x.tobytes())
          kw[key] = jax.device_put(x)
        assert digest.hexdigest() == meta["input_sha256"]
        cfg = (
            m.Config(
                fuse_forward=True,
                packed_forward=True,
                packed_output=True,
                fuse_backward=True,
                packed_backward=True,
                packed_gradients=True,
                fuse_cp_backward=True,
            )
            if variant == "merge-opt"
            else None
        )
        op = m.PallasMosaicTpuKimiDeltaAttention(config=cfg)
        fn, arrays = tokamax.standardize_function(
            op, kwargs=kw, mode=mode, seed=None
        )

        def invoke(*xs):
          return fn(xs)

        begin = time.perf_counter()
        exe = jax.jit(invoke).lower(*arrays).compile()
        compile_s = time.perf_counter() - begin
        formats = exe.input_formats[0]
        resident = tuple(
            jax.device_put(x, f) for x, f in zip(arrays, formats, strict=True)
        )
        jax.block_until_ready(resident)
        for _ in range(10):
          jax.block_until_ready(exe(*resident))
        samples = []
        for _ in range(30):
          begin = time.perf_counter_ns()
          value = exe(*resident)
          jax.block_until_ready(value)
          samples.append((time.perf_counter_ns() - begin) / 1e6)
        mem = exe.memory_analysis()
        record = {
            "input_sha256": digest.hexdigest(),
            "samples_ms": samples,
            "median_ms": float(np.median(samples)),
            "compile_s": compile_s,
            "input_formats": list(map(str, formats)),
            "temp_bytes": int(mem.temp_size_in_bytes),
            "argument_bytes": int(mem.argument_size_in_bytes),
            "output_bytes": int(mem.output_size_in_bytes),
        }
        result = jax.block_until_ready(exe(*resident))
        host = {}
        types = []
        hashout = hashlib.sha256()
        finite = True
        for i, x in enumerate(jax.tree.leaves(result)):
          x = np.asarray(x)
          types.append(str(x.dtype))
          hashout.update(str((x.shape, str(x.dtype))).encode())
          hashout.update(x.tobytes())
          if x.dtype.kind != "V" or str(x.dtype) == "bfloat16":
            finite = finite and bool(np.isfinite(x.astype(np.float32)).all())
          host[str(i)] = x.view(np.uint16) if str(x.dtype) == "bfloat16" else x
        record.update(
            output_sha256=hashout.hexdigest(),
            finite=finite,
            output_dtypes=types,
        )
        if round == 0:
          np.savez_compressed(root / (name + "-" + mode + ".npz"), **host)
        (root / (name + "-" + mode + ".hlo.txt")).write_text(exe.as_text())
        entry["modes"][mode] = record
        dump()
        print("DONE", name, mode, record["median_ms"], finite, flush=True)
        del exe, resident, result, value, host, kw, arrays, fn, invoke, op
    except Exception:
      entry["modes"][mode] = {"error": traceback.format_exc()}
      dump()
      print("ERROR", name, mode, entry["modes"][mode]["error"], flush=True)
    jax.clear_caches()
    gc.collect()
dump()
