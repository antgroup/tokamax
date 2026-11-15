import pathlib, subprocess, json, statistics, concurrent.futures, os, time, threading

os.environ.update(
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1",
)
failures = []
failure_lock = threading.Lock()


def run_query(cmd):
  for attempt in range(3):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
      return result
    with failure_lock:
      failures.append(
          dict(
              command=cmd,
              attempt=attempt,
              code=result.returncode,
              stderr=result.stderr,
              stdout=result.stdout,
          )
      )
      (root / "parse-errors.json").write_text(json.dumps(failures, indent=2))
    time.sleep(0.5)
  raise RuntimeError(failures[-1])


root = pathlib.Path("/workspace/official-sync/profiles")
files = list(root.glob("*-10/*/[01]/plugins/profile/*/*.xplane.pb"))
assert len(files) == 108, len(files)
previous = (
    {
        r["xplane"]: r
        for r in json.loads((root / "device-captures.json").read_text())
    }
    if (root / "device-captures.json").exists()
    else {}
)


def extract(pb):
  if str(pb) in previous:
    return previous[str(pb)]
  parts = pb.relative_to(root).parts
  variant = parts[0].removesuffix("-10")
  name, mode = parts[1].rsplit("-", 1)
  capture = int(parts[2])
  cmd = [
      "xprof",
      "list_xplane_events",
      str(pb),
      "--plane_regex=^/device:TPU:0$",
      "--max_events=100",
      "--bypass_cache=True",
  ]
  result = run_query(cmd + ["--event_regex=jit_invoke"])
  data = json.loads(result.stdout)
  assert not data.get("truncated")
  events = [e for e in data["events"] if e["line_id"] == "XLA Modules"]
  assert len(events) == 5, (str(pb), len(events))
  (pb.parent / "modules.json").write_text(json.dumps(data, indent=2))
  check = json.loads(run_query(cmd + ["--event_regex=jit__multi_slice"]).stdout)
  assert check.get("total_matched", len(check["events"])) == 0
  return {
      "variant": variant,
      "case": name,
      "mode": mode,
      "capture": capture,
      "module_us": [
          e["duration_ps"] / 1e6
          for e in sorted(events, key=lambda e: e["offset_ps"])
      ],
      "module_events": [e["event"] for e in events],
      "input_reshard_modules": 0,
      "xplane": str(pb),
  }


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  rows = list(pool.map(extract, files))
(root / "device-captures.json").write_text(json.dumps(rows, indent=2))
groups = {}
for row in rows:
  groups.setdefault((row["variant"], row["case"], row["mode"]), []).append(row)
summary = []
for (v, n, m), captures in sorted(groups.items()):
  assert len(captures) == 2
  summary.append(
      {
          "variant": v,
          "case": n,
          "mode": m,
          "median_device_ms": statistics.median(
              sum([c["module_us"] for c in captures], [])
          )
          / 1000,
          "capture_medians_ms": [
              statistics.median(c["module_us"]) / 1000
              for c in sorted(captures, key=lambda c: c["capture"])
          ],
      }
  )
(root / "device-summary.json").write_text(json.dumps(summary, indent=2))
print("VERIFIED", len(summary), "points and", len(rows), "captures")
