import subprocess, os, pathlib, json, sys

os.environ.update(
    JAX_PLATFORMS="tpu,cpu",
    LIBTPU_INIT_ARGS="--xla_tpu_impure_enable_large_2nd_minor_layout=true",
)
root = pathlib.Path("/workspace/official-sync")
out = root / "results"
out.mkdir(exist_ok=True)
subprocess.run([sys.executable, str(root / "generate.py")], check=True)
variants = ["main-default", "merge-default", "merge-opt"]
for round, order in [(0, variants), (1, list(reversed(variants)))]:
  for v in order:
    print("SWEEP", round, v, flush=True)
    with (out / (v + "-" + str(round) + ".log")).open("w") as stream:
      subprocess.run(
          [sys.executable, str(root / "worker.py"), v, str(round)],
          stdout=stream,
          stderr=subprocess.STDOUT,
          check=True,
      )
print("SWEEP_COMPLETE", flush=True)
