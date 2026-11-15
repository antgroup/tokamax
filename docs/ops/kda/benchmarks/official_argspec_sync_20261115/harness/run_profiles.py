import os, pathlib, subprocess, sys

os.environ.update(
    JAX_PLATFORMS="tpu,cpu",
    LIBTPU_INIT_ARGS="--xla_tpu_impure_enable_large_2nd_minor_layout=true",
)
p = pathlib.Path("/workspace/official-sync/profiles")
p.mkdir(exist_ok=True)
for variant in ["main-default", "merge-default", "merge-opt"]:
  print("PROFILE", variant, flush=True)
  with (p / (variant + ".log")).open("w") as f:
    subprocess.run(
        [
            sys.executable,
            "/workspace/official-sync/profile_worker.py",
            variant,
            "10",
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
print("PROFILES_COMPLETE", flush=True)
