#!/usr/bin/env python3
"""Stage D modelling (eswa-c env). Reuses the frozen stage-C scripts UNCHANGED (c_model.py, c6_final.py, c6b_addendum.py,
c7_eval.py) by running them in-process with the c_model module's data root redirected (c_model.EXP / c_model.F), so the
procedure is byte-for-byte the stage-C procedure (checked: --selftest reproduces the C6 artifacts from the B1 features).
  (a) frozen : python -I d_model.py frozen  --system W --eval_features <dir> --eval_labels <dir> --out 06_results/D_W_eval_frozen.json
      = c7_eval.py with the frozen C6 artifacts (03_configs/final.yaml) on the new system's eval edits (no refitting).
  (b) retrain: python -I d_model.py retrain --system W --features_dirs <train feat dir> <dev feat dir> --eval_features <dir>
               --eval_labels <dir> --root 05_runs/2026-11-10_D_W_model --out 06_results/D_W_eval_retrained.json
      = C2-C4 (nested grouped CV, OOF, 1000-resample evaluation), C6 final fit, C6 addendum (risk_lr, OOF thresholds,
        calibration) on the new system's train+dev, then c7_eval.py ONCE on its eval edits.
  --selftest: (b) on the B1 train+dev features without the eval step; asserts the artifacts equal 05_runs/2026-11-10_C6_final.
"""
import argparse, json, os, runpy, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c_model as C   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["frozen", "retrain", "selftest"]); ap.add_argument("--system", default="")
ap.add_argument("--features_dirs", nargs=2, default=None, help="train dir, dev dir (features_<ds>.csv, utt_counts_<ds>.csv)")
ap.add_argument("--eval_features"); ap.add_argument("--eval_labels"); ap.add_argument("--root"); ap.add_argument("--out")
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
REAL_EXP = C.EXP


def run(script, *args):
    sys.argv = [script, *map(str, args)]
    print(f"== {script} {' '.join(map(str, args))}", flush=True)
    if script == "c_model.py":
        C.main()          # the imported (redirected) module; run_path would re-create its globals with the real paths
    else:
        runpy.run_path(f"{HERE}/{script}", run_name="__main__")   # these scripts import c_model -> the redirected module


if a.mode == "frozen":
    assert not os.path.exists(a.out), f"{a.out} exists (eval scored once only)"
    run("c7_eval.py", "--features", a.eval_features, "--labels", a.eval_labels, "--out", a.out)
    sys.exit(0)

if a.mode == "selftest":
    a.root = a.root or f"{REAL_EXP}/05_runs/d_model_selftest"   # inside EXP_DIR
    a.features_dirs = [f"{REAL_EXP}/05_runs/2026-11-10_C1_features"] * 2
root = os.path.abspath(a.root)
assert not os.path.exists(f"{root}/05_runs/2026-11-10_C6_final"), f"{root} already has a final fit"
os.makedirs(f"{root}/features", exist_ok=True)
for d, ds in zip(a.features_dirs, ["train", "dev"]):
    for f in [f"features_{ds}.csv", f"utt_counts_{ds}.csv"]:
        shutil.copy2(f"{d}/{f}", f"{root}/features/{f}")
C.EXP = root; C.F = f"{root}/features"
cv = f"{root}/05_runs/2026-11-10_C2_cv"; fin = f"{root}/05_runs/2026-11-10_C6_final"
os.makedirs(cv, exist_ok=True); os.makedirs(fin, exist_ok=True)
run("c_model.py", "cv", "--out", cv)
run("c_model.py", "evaluate", "--out", cv, "--n_boot", a.n_boot)
run("c6_final.py", "--out", fin)
run("c6b_addendum.py", "--out", fin)
if a.mode == "selftest":
    ref = json.load(open(f"{REAL_EXP}/05_runs/2026-11-10_C6_final/final_fit.json"))["artifact_sha256"]
    new = json.load(open(f"{fin}/final_fit.json"))["artifact_sha256"]
    radd = json.load(open(f"{REAL_EXP}/05_runs/2026-11-10_C6_final/addendum_fit.json"))
    nadd = json.load(open(f"{fin}/addendum_fit.json"))
    res = dict(artifacts_equal=ref == new, risk_lr_equal=radd["risk_lr"]["artifact_sha256"] == nadd["risk_lr"]["artifact_sha256"],
               thresholds_equal=radd["operating_points"] == nadd["operating_points"])
    print("SELFTEST", json.dumps(res)); sys.exit(0 if all(res.values()) else 3)
assert not os.path.exists(a.out), f"{a.out} exists (eval scored once only)"
run("c7_eval.py", "--features", a.eval_features, "--labels", a.eval_labels, "--out", a.out)
