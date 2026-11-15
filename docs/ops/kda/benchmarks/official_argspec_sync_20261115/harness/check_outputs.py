import pathlib, json, numpy as np, ml_dtypes

p = pathlib.Path("/workspace/official-sync/results")
variants = ["merge-default", "merge-opt"]
main = json.loads((p / "main-default-0/report.json").read_text())
rows = []
for name, c in main["cases"].items():
  for mode, base in c["modes"].items():
    for v in variants:
      r = json.loads((p / (v + "-0") / "report.json").read_text())["cases"][
          name
      ]["modes"][mode]
      row = {
          "case": name,
          "mode": mode,
          "variant": v,
          "bitwise_equal_to_main": base["output_sha256"] == r["output_sha256"],
          "finite": r["finite"],
      }
      if not row["bitwise_equal_to_main"]:
        a = np.load(p / "main-default-0" / (name + "-" + mode + ".npz"))
        b = np.load(p / (v + "-0") / (name + "-" + mode + ".npz"))
        deltas = []
        assert a.files == b.files
        for i, k in enumerate(a.files):
          x, y = a[k], b[k]
          assert x.shape == y.shape and x.dtype == y.dtype
          if x.dtype.kind == "V":
            continue
          if base["output_dtypes"][i] == "bfloat16":
            x, y = x.view(ml_dtypes.bfloat16), y.view(ml_dtypes.bfloat16)
          x, y = x.astype(np.float64), y.astype(np.float64)
          assert np.isfinite(x).all() and np.isfinite(y).all()
          d = y - x
          deltas.append(
              {
                  "leaf": i,
                  "shape": list(x.shape),
                  "relative_rms": float(
                      np.linalg.norm(d.ravel())
                      / max(np.linalg.norm(x.ravel()), 1e-30)
                  ),
                  "max_abs": float(np.max(np.abs(d))),
              }
          )
        row["deltas"] = deltas
        print(name, mode, v, deltas, flush=True)
      rows.append(row)
(p / "output-comparison.json").write_text(json.dumps(rows, indent=2))
print("OUTPUT_COMPARISONS_COMPLETE", len(rows))
