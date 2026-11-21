#!/usr/bin/env python3
"""Stage D: check that the large-v3 pre-norm CTM has the same format as the official Whisper-small pre-norm CTM
(reference-materials/pre-norm/<ds>-gec.ctm) and write the ASR-STM subset for ASR WER on the decoded files.
Run from the corpus root (any python3):  python3 -I d_ctm_check.py --ctm <asr.ctm> --flist <flist> --ref_ctm <official pre-norm ctm>
   --asr_stm reference-materials/stms/<ds>-asr.stm --out_stm <dir>/ref-asr.stm
Checks (exit 3 on failure): 6 fields per line; channel 1; start, dur numeric >= 0; confidence in [0, 1] with 2 decimals;
file IDs subset of the flist; lines sorted by file ID; non-empty word field. Reports coverage vs the official CTM (numbers only).
The STM subset keeps the header lines and every segment of the decoded files (scoring reference for sclite; stays in the corpus tree).
"""
import argparse, json, re, sys
ap = argparse.ArgumentParser()
for k in ["ctm", "flist", "ref_ctm", "asr_stm", "out_stm"]:
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
ids = [l.split()[0] for l in open(a.flist) if l.strip()]; S = set(ids)
bad = dict(fields=0, channel=0, time=0, conf=0, conf_format=0, not_in_flist=0, empty_word=0, unsorted=0)
files = {}; prev = ""
for l in open(a.ctm):
    f = l.rstrip("\n").split(" ")
    if len(f) != 6:
        bad["fields"] += 1; continue
    fid, ch, st, du, w, c = f
    bad["channel"] += ch != "1"; bad["not_in_flist"] += fid not in S; bad["empty_word"] += not w
    try:
        bad["time"] += float(st) < 0 or float(du) < 0; bad["conf"] += not (0 <= float(c) <= 1)
    except ValueError:
        bad["time"] += 1
    bad["conf_format"] += not re.fullmatch(r"\d\.\d\d", c); bad["unsorted"] += fid < prev; prev = fid
    files[fid] = files.get(fid, 0) + 1
ref = {}
for l in open(a.ref_ctm):
    fid = l.split()[0]
    if fid in S:
        ref[fid] = ref.get(fid, 0) + 1
n_ref_fields = {len(l.rstrip("\n").split(" ")) for _, l in zip(range(1000), open(a.ref_ctm))}
with open(a.out_stm, "w") as o:
    n_seg = 0
    for l in open(a.asr_stm):
        if l.startswith(";;") or l.split()[0] in S:
            o.write(l); n_seg += not l.startswith(";;")
out = dict(lines=sum(files.values()), files_with_words=len(files), files_in_flist=len(S),
           files_without_words=len(S - set(files)), official_small_words=sum(ref.values()), official_small_files=len(ref),
           official_fields=sorted(n_ref_fields), problems=bad, stm_segments_kept=n_seg)
print(json.dumps(out))
sys.exit(3 if any(bad.values()) or n_ref_fields != {6} else 0)
