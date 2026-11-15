import pathlib, json

r = pathlib.Path("/workspace/official-sync")
env = []
checks = []
for variant in ("main-default", "merge-default", "merge-opt"):
  wall = json.loads(
      (r / "results" / (variant + "-0") / "report.json").read_text()
  )
  profile = json.loads(
      (r / "profiles" / (variant + "-10") / "report.json").read_text()
  )
  for round in (0, 1):
    d = json.loads(
        (
            r / "results" / (variant + "-" + str(round)) / "report.json"
        ).read_text()
    )
    env.append(
        {
            k: d[k]
            for k in (
                "variant",
                "round",
                "device",
                "device_kind",
                "versions",
                "python",
                "flags",
            )
        }
    )
  for name, c in wall["cases"].items():
    for mode, entry in c["modes"].items():
      p = profile["cases"][name]["modes"][mode]
      checks.append(
          dict(
              variant=variant,
              case=name,
              mode=mode,
              input_sha_equal=entry["input_sha256"] == p["input_sha256"],
              profile_output_equals_wall=entry["output_sha256"]
              == p["output_sha256"],
              input_formats_equal=entry["input_formats"] == p["input_formats"],
          )
      )
assert (
    len(
        {
            json.dumps(
                {
                    k: e[k]
                    for k in ("device_kind", "versions", "python", "flags")
                },
                sort_keys=True,
            )
            for e in env
        }
    )
    == 1
)
assert all(c["input_sha_equal"] and c["input_formats_equal"] for c in checks)
(r / "environment.json").write_text(json.dumps(env, indent=2))
(r / "profile-output-consistency.json").write_text(json.dumps(checks, indent=2))
print(
    "ENVIRONMENTS_IDENTICAL",
    len(env),
    "PROFILE_MATCHES",
    sum(c["profile_output_equals_wall"] for c in checks),
    "of",
    len(checks),
)
