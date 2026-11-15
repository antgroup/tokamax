"""Certify internal BF16 pointwise acceptance from complete saved max errors.

This re-scores existing full-array measurements; it does not rerun TPU kernels
or claim an elementwise replay. max_abs < atol is sufficient for every element
to satisfy allclose (and any additional ULP allowance), independent of relative
magnitude. A larger maximum is inconclusive and requires the original arrays.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

NAMES = ("output", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
INTERNAL_SHA = "18ea6891f0e8d9affe9e81815267f5251bdb0b3b"


def rescore(path):
  data = json.loads(path.read_text())
  rows = []
  for case, leaves in data.items():
    assert sorted(x["index"] for x in leaves) == list(range(8)), case
    for leaf in leaves:
      name = NAMES[leaf["index"]]
      maximum = leaf["max_abs"]
      assert math.isfinite(maximum) and maximum >= 0, (case, name)
      # Native packed contract checks all eight leaves with these tolerances.
      rules = [("native_packed_contract", 0.005, 0.05, None)]
      if name == "ddt_bias":
        rules.append(("full_kda_bias", 0.008, 0.008, 1))
      elif name == "dA_log":
        rules.append(("full_kda_a_log_bf16", 0.05, 0.05, 1))
      assessments = []
      for policy, atol, rtol, ulp in rules:
        passed = maximum < atol
        assessments.append(
            dict(
                policy=policy,
                atol=atol,
                rtol=rtol,
                max_bf16_ulp=ulp,
                certified_all_elements_pass=passed,
                status="PASS_BY_COMPLETE_MAX_ABS"
                if passed
                else "REQUIRES_ARRAY_REPLAY",
                absolute_bound_utilization=maximum / atol,
            )
        )
      rows.append(
          dict(
              case=case,
              name=name,
              max_abs=maximum,
              relative_rms_diagnostic=leaf["relative_rms"],
              assessments=assessments,
          )
      )
  return dict(
      internal_main=INTERNAL_SHA,
      internal_sources={
          "full_kda_bias": "tests/ops/kda/test_chunk_kda_tpu.py:417",
          "full_kda_a_log_bf16": "tests/ops/kda/test_chunk_kda_tpu.py:412",
          "native_packed_contract": "tests/ops/kda/test_native_segids_residual_contract.py:685",
          "ulp_rule": "tests/utils/__init__.py:106",
      },
      source_measurement_sha256=hashlib.sha256(
          path.read_bytes().replace(b"\r\n", b"\n")
      ).hexdigest(),
      method="Sufficient bound from previously measured complete max_abs; no new device run or element replay",
      rms_diagnostic_is_review_gate=False,
      original_finiteness_checks="Reference and produced arrays were asserted finite before metrics were saved",
      all_certified=all(
          a["certified_all_elements_pass"]
          for r in rows
          for a in r["assessments"]
      ),
      rows=rows,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input",
      type=Path,
      default=Path(__file__).with_name(
          "backward_reference_followup_20261115.json"
      ),
  )
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = rescore(args.input)
  serialized = json.dumps(result, indent=2) + "\n"
  if args.output:
    args.output.write_text(serialized)
  else:
    print(serialized)
  if not result["all_certified"]:
    raise SystemExit(
        "Saved bounds are inconclusive; replay full arrays before deciding readiness"
    )


if __name__ == "__main__":
  main()
