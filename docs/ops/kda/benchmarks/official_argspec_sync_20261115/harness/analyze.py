import pathlib, json, csv, statistics, numpy as np

root = pathlib.Path("/workspace/official-sync/results")
variants = ["main-default", "merge-default", "merge-opt"]
reports = {
    (v, r): json.loads((root / (v + "-" + str(r)) / "report.json").read_text())
    for v in variants
    for r in [0, 1]
}
manifest = json.loads(
    pathlib.Path("/workspace/official-sync/inputs/manifest.json").read_text()
)
rows = []
all_data = {}
for meta in manifest:
  name = meta["name"]
  all_data[name] = {}
  for mode in ["forward"] + (
      [] if "forward_only" in meta["tags"] else ["forward_and_vjp"]
  ):
    entry = {}
    all_data[name][mode] = entry
    for v in variants:
      cells = [reports[v, r]["cases"][name]["modes"][mode] for r in [0, 1]]
      assert all("error" not in c and c["finite"] for c in cells), (
          v,
          name,
          mode,
          cells,
      )
      samples = sum([c["samples_ms"] for c in cells], [])
      assert len(samples) == 60
      entry[v] = {
          "median_ms": statistics.median(samples),
          "round_medians_ms": [c["median_ms"] for c in cells],
          "p10_ms": float(np.percentile(samples, 10)),
          "p90_ms": float(np.percentile(samples, 90)),
          "samples_ms": samples,
          "temp_bytes": cells[0]["temp_bytes"],
          "argument_bytes": cells[0]["argument_bytes"],
          "output_bytes": cells[0]["output_bytes"],
          "input_formats": cells[0]["input_formats"],
          "repeat_bitwise_equal": cells[0]["output_sha256"]
          == cells[1]["output_sha256"],
      }
    main = entry["main-default"]["median_ms"]
    row = {"case": name, "mode": mode, "main_ms": main}
    for v in variants[1:]:
      row[v + "_ms"] = entry[v]["median_ms"]
      row[v + "_delta_percent"] = (entry[v]["median_ms"] / main - 1) * 100
    rows.append(row)
    print(
        name,
        mode,
        "default %.2f%% opt %.2f%%"
        % (row["merge-default_delta_percent"], row["merge-opt_delta_percent"]),
        flush=True,
    )
(root / "comparison.json").write_text(
    json.dumps(
        {"manifest": manifest, "timings": all_data, "rows": rows}, indent=2
    )
)
with (root / "summary.csv").open("w") as f:
  w = csv.DictWriter(f, fieldnames=list(rows[0]))
  w.writeheader()
  w.writerows(rows)
