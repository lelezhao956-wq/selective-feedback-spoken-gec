#!/usr/bin/env python3
"""Stage D-ASR: Whisper large-v3 (mlx-whisper) decode of a file list -> per-file Whisper JSON -> pre-norm CTM.
Run in the eswa-d env from the corpus root:
  python -I d_asr.py --flist reference-materials/flists.flac/dev-gec.tsv --out_dir stageD/W_dev/decode
Decoding settings follow the official baseline-pipeline/run-scripts/run-asr-decode.sh + asr-tools/whisper_transcribe.py
as far as mlx-whisper allows: language en, word_timestamps, condition_on_previous_text False, no_speech_threshold 0.6,
suppress_blank True, length_penalty None, initial_prompt None, fp16, default temperature fallback
(0.0, 0.2, ..., 1.0; compression_ratio_threshold 2.4, logprob_threshold -1.0).
DEVIATION: beam_size 5 is not implemented in mlx-whisper 0.4.2 -> greedy decoding at T=0 (see 03_configs/2026-11-10_D_plan.yaml).
Sampling at T>0 (fallback only) uses mx.random.seed(SEED + file index) -> deterministic per file, also after a resume.
CTM: the official asr-tools/whisper_json.json_to_ctm_lines (unchanged; word probability -> confidence column), sorted as in
whisper_transcribe.py. Resume: a JSON is written to <id>.json.tmp and renamed, so every <id>.json is complete; existing JSONs
are kept, and complete JSONs in <out_dir>.partial-* (renamed by the queue) are COPIED (never moved) if they parse and were
decoded with the same settings. Prints one JSON summary line. Progress: number of files in <out_dir>/json.
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time
import mlx.core as mx
import mlx_whisper

SEED = 20261110
MODEL = "mlx-community/whisper-large-v3-mlx"
MODEL_REV = "49e6aa286ad60c14352c404340ded53710378a11"
KW = dict(language="en", word_timestamps=True, condition_on_previous_text=False, no_speech_threshold=0.6, suppress_blank=True,
          length_penalty=None, initial_prompt=None, fp16=True, verbose=None)
ap = argparse.ArgumentParser()
ap.add_argument("--flist", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--limit", type=int, default=0, help="first N files only (rehearsal)")
a = ap.parse_args()
sys.path.insert(0, os.path.abspath("baseline-pipeline/asr-tools"))
from whisper_json import json_to_ctm_lines  # noqa: E402  (official, unchanged)

L = [l.split() for l in open(a.flist) if l.strip()]
if a.limit:
    L = L[:a.limit]
J = f"{a.out_dir}/json"; os.makedirs(J, exist_ok=True)
SETTINGS = dict(model=MODEL, model_rev=MODEL_REV, decoding="greedy (beam search not available in mlx-whisper)", seed=SEED,
                mlx_whisper="0.4.2", **{k: v for k, v in KW.items() if k != "verbose"})
SET_ID = hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:16]
json.dump(dict(settings=SETTINGS, settings_id=SET_ID), open(f"{a.out_dir}/decode_settings.json", "w"), indent=1)
repo = os.path.join((os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")), f"models--mlx-community--whisper-large-v3-mlx/snapshots/{MODEL_REV}")
if not os.path.isdir(repo): raise SystemExit(f"model not found: {repo} (download it first: README, Models; cache = HF_HUB_CACHE or HF_HOME/hub)")


def ok_json(p):
    try:
        r = json.load(open(p)); return isinstance(r.get("segments"), list) and r.get("_settings_id") == SET_ID
    except Exception:
        return False


partials = sorted(glob.glob(f"{a.out_dir}.partial-*/json"))
dur = lambda p: float(subprocess.check_output([os.path.join(os.path.dirname(sys.executable), "ffprobe"), "-v", "error",
                                               "-show_entries", "format=duration", "-of", "csv=p=0", p]))
n_kept = n_copied = n_dec = n_missing = 0; T = A = 0.0
for i, (fid, wav) in enumerate(L):
    out = f"{J}/{fid}.json"
    if ok_json(out):
        n_kept += 1; continue
    src = next((f"{d}/{fid}.json" for d in reversed(partials) if ok_json(f"{d}/{fid}.json")), None)
    if src:
        shutil.copy2(src, out); n_copied += 1; continue
    if not os.path.exists(wav):
        print("ERROR: audio not found:", fid, file=sys.stderr); n_missing += 1; continue
    mx.random.seed(SEED + i)
    t = time.time(); r = mlx_whisper.transcribe(wav, path_or_hf_repo=repo, **KW); T += time.time() - t; A += dur(wav)
    r["_settings_id"] = SET_ID
    json.dump(r, open(out + ".tmp", "w"), ensure_ascii=False); os.replace(out + ".tmp", out); n_dec += 1

lines = json_to_ctm_lines(J)          # official conversion (reads segments/words only)
lines.sort(key=lambda x: x.split()[0])
with open(f"{a.out_dir}/asr.ctm", "w") as f:
    for l in lines:
        print(l, file=f)
summ = dict(n_files=len(L), kept=n_kept, copied_from_partial=n_copied, decoded=n_dec, missing_audio=n_missing,
            ctm_words=len(lines), files_with_words=len({l.split()[0] for l in lines}), decode_s=round(T, 1), audio_s=round(A, 1),
            rtf=round(T / A, 4) if A else None, settings_id=SET_ID)
json.dump(summ, open(f"{a.out_dir}/asr_summary.json", "w"), indent=1)
print(json.dumps(summ))
sys.exit(0 if n_missing == 0 and len(glob.glob(f"{J}/*.json")) >= len(L) else 3)
