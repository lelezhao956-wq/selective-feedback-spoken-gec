#!/usr/bin/env python3
"""C5 side analysis: disfluent material that DD leaves in the B2b input (manual verbatim -> DD), counted by type.
Run with the sandi-all env from the corpus root:  python c5_b2b_residual_disfl.py --ds dev --out <json>
Token sequences as in b3_edit_labels.py (scored region, spaCy): A = V = manual verbatim tokens after the official
normalisation (B2b input), S = B2b system fluent tokens (after DD), M = manual fluent tokens.
Residual disfluency token = V token that the annotators removed (not aligned as equal in V<->M) but DD kept (aligned as
equal in S<->A). Also counted: fluent tokens wrongly deleted by DD (kept in M, deleted by DD).
Type of each residual token from the Phase-2 annotation marks ({ds}-trans-ref.json), aligned to V with Levenshtein.opcodes
over lower-cased tokens: hesitation tag (filled pause), partial mark, backchannel mark, disfluency mark (false start or
repetition; split here: 'repetition' if the marked run of tokens is immediately followed by the same tokens, otherwise
'false_start'), 'unmarked' (removed by annotators without a word mark), 'unaligned' (no equal JSON token).
Only counts are written.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b3_edit_labels as L   # noqa: E402  (official spoken_errant / create_files_for_errant via b3)
import Levenshtein           # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--run_suffix", default="_v2", help="_v2 = markup-free B2b (default); _regec = v1 (invalid, markup)")
a = ap.parse_args()
os.chdir(L.CORPUS)
ds = a.ds
rd = f"stageB/B2b_{ds}{a.run_suffix}"
stm = lambda k: f"reference-materials/stms/{ds}-{k}.stm"
final = L.rebuild_final(stm("fluent"), f"{rd}/gec/fluent.ctm", stm("gec"), f"{rd}/gec/gec.ctm")
ids = list(final["ID"])
stm_flt = L.CF.load_stm(stm("fluent"))
ign = stm_flt[stm_flt["Transcription"] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
A_txt = L.tokens_from_ctm(f"{rd}/gec/norm/gec.ctm", ign, ids)

J = {f["File-id"]: f["Transcript"] for f in json.load(open(f"reference-materials/annotations/{ds}-trans-ref.json"))["files"]}


def json_tokens(tr):
    """spaCy tokens of the annotated words with a class per token."""
    words, cls = [], []
    for w in tr:
        if "word" not in w:      # tag-only items: phrase boundaries, %hesitation% (removed by the normalisation)
            continue
        t = w.get("tag"); m = w.get("marks") or []
        c = "hesitation" if t == "hesitation" else "partial" if "partial" in m else "backchannel" if "backchannel" in m \
            else "disfluency" if "disfluency" in m else "fluent"
        words.append(str(w["word"])); cls.append(c)
    text = " ".join(words); starts, p = [], 0
    for w in words:
        starts.append(p); p += len(w) + 1
    toks, tcls = [], []
    for t in L.CF.nlp(text):
        k = max(i for i, s in enumerate(starts) if s <= t.idx)
        toks.append(t.text.lower().strip(".,?!")); tcls.append(cls[k])
    return toks, tcls


C = Counter(); n = Counter()
for i, u in enumerate(ids):
    S = final["flt_hyp"][i].split(" "); M = final["flt_ref"][i].split(" ")
    A = A_txt[u].split(" ") if A_txt[u] else []
    sa, vm = L.Align(S, A), L.Align(A, M)
    removed_by_ann = [v for v in range(len(A)) if v not in vm.eq_xy]
    resid = [v for v in removed_by_ann if v in sa.eq_yx]
    wrong_del = [v for v in range(len(A)) if v in vm.eq_xy and v not in sa.eq_yx]
    n["V_tokens"] += len(A); n["removed_by_annotators"] += len(removed_by_ann); n["residual_kept_by_DD"] += len(resid)
    n["fluent_deleted_by_DD"] += len(wrong_del); n["utts"] += 1
    if u not in J:
        C["unaligned"] += len(resid); n["utts_without_json"] += 1; continue
    jt, jc = json_tokens(J[u])
    Al = [x.lower().strip(".,?!") for x in A]
    v2j = {}
    for op, i1, i2, j1, j2 in Levenshtein.opcodes(Al, jt):
        if op == "equal":
            for k in range(i2 - i1):
                v2j[i1 + k] = j1 + k
    for v in resid:
        if v not in v2j:
            C["unaligned"] += 1; continue
        j = v2j[v]; c = jc[j]
        if c == "disfluency":
            r0 = j
            while r0 > 0 and jc[r0 - 1] == "disfluency":
                r0 -= 1
            r1 = j
            while r1 + 1 < len(jc) and jc[r1 + 1] == "disfluency":
                r1 += 1
            run = jt[r0:r1 + 1]; nxt = [t for t, cc in zip(jt[r1 + 1:], jc[r1 + 1:]) if cc not in ("hesitation",)][:len(run)]
            c = "repetition" if run == nxt else "false_start"
        elif c == "fluent":
            c = "unmarked"
        C[c] += 1
out = dict(ds=ds, counts=dict(n), residual_by_type=dict(C),
           residual_share_aligned=round(1 - C["unaligned"] / max(n["residual_kept_by_DD"], 1), 4))
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out))
