#!/usr/bin/env python3
"""Parse the WER Sum/Avg row, the SGECF result (with per-edit-type detail) and the step timings of
one stage-B run directory into <run_dir>/metrics.json."""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--mode", required=True)
a = ap.parse_args()
R = Path(a.run_dir)

m = {"run": R.name, "mode": a.mode}
row = [l for l in (R / "gec_wer_sum_avg.txt").read_text().splitlines() if "Sum/Avg" in l][0]
nums = re.findall(r"-?\d+\.?\d*", row)
m["gec_wer"] = dict(zip(["n_snt", "n_wrd", "corr", "sub", "del", "ins", "err", "s_err", "nce"], map(float, nums)))

t = (R / "sgecf_result_edit_detail.txt").read_text()
m["sgecf_per_type"] = {k: dict(zip(["P", "R", "F05"], map(float, v.split())))
                       for k, v in re.findall(r"Edit Type:\t(\S+)\nP:\tR:\tF0\.5\n([\d.]+\t[\d.]+\t[\d.]+)", t)}
o = re.search(r"Overall results:\nP:\tR:\tF0\.5\n([\d.]+)\t([\d.]+)\t([\d.]+)", t)
m["sgecf_overall"] = dict(zip(["P", "R", "F05"], map(float, o.groups())))

steps = {}
for l in (R / "step_timings.txt").read_text().splitlines():
    p = l.split()
    if p[0] in ("STEP", "SUBSTEP") and len(p) >= 4:
        steps[p[1]] = int(p[3]) - int(p[2])
m["step_seconds"] = steps
json.dump(m, open(R / "metrics.json", "w"), indent=1)
print(json.dumps({k: m[k] for k in ("gec_wer", "sgecf_overall", "step_seconds")}))
