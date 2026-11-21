#!/usr/bin/env python3
"""C1b: inference-time features for every B1 hypothesis edit (one row per row of hyp_edit_labels_<ds>.csv).
Run with the sandi-all env from the corpus root:
  python c1_features.py --ds dev --run_dir stageB/B1_dev --labels <05_runs/...B1_dev/labels> --bart <bart_scores_dev.jsonl> --out_dir <dir>

Only information available at inference time is used for features: Whisper word confidences and times (fluent.ctm /
norm/gec.ctm), DD decisions (which pre-DD words were deleted), the GEC output and BART teacher-forced scores, edit
type/extent from ERRANT on (system fluent text, system GEC text), and the test part from the file ID. Reference-based
columns (false_feedback, asr_induced, ...) are copied as labels only. The scorer's IGNORE-time filtering uses the
official STM segment times; these are part of the official evaluation protocol (the same region is scored for every
system), so it defines which edits exist, not a feature.

Verification (exit 3 on failure): the utterance token sequences rebuilt here equal the official errant flt_ctm/gec_ctm
files line by line; every edit's token span maps to CTM words whose concatenated text contains the token text;
per-phrase word counts equal the BART score arrays; system fluent words are an ordered subsequence of the pre-DD words.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import pandas as pd

CORPUS = (os.environ.get("RUN_DIR") or __import__("sys").exit("RUN_DIR is not set: build a run root with make_run_root.sh and export RUN_DIR (README, Setup)"))
sys.path.insert(0, os.path.join(CORPUS, "scoring-tools/sgecf-tools"))
import spoken_errant as SE            # noqa: E402
import create_files_for_errant as CF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--bart", required=True)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
os.chdir(CORPUS)
rd = a.run_dir.rstrip("/")
nan = float("nan")


def filtered_ctm(ctm_path, stm_path):
    df = pd.read_csv(ctm_path, sep="\t", header=None, keep_default_na=False)
    df["Base_ID"] = df[0].str.rsplit("_", n=1).str[0]
    df["widx"] = df.groupby(0).cumcount()
    stm = CF.load_stm(stm_path)
    ign = stm[stm["Transcription"] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
    ignw = CF.get_non_scorable_words(ign, df)
    return df, df.loc[~df.index.isin(ignw.index)]


def utt_tokens(dfu):
    """spaCy tokens of ' '.join(words) as in create_files_for_errant; token -> row position in dfu."""
    words = [str(w) for w in dfu[4]]
    text = " ".join(words)
    starts, p = [], 0
    for w in words:
        starts.append(p); p += len(w) + 1
    doc = CF.nlp(text)
    t2w = []
    for t in doc:
        k = max(i for i, s in enumerate(starts) if s <= t.idx)
        t2w.append(k)
    return " ".join(t.text for t in doc), [t.text for t in doc], t2w, words


stm = lambda k: f"reference-materials/stms/{a.ds}-{k}.stm"
flt_all, flt = filtered_ctm(f"{rd}/gec/fluent.ctm", stm("fluent"))
gec_all, gec = filtered_ctm(f"{rd}/gec/gec.ctm", stm("gec"))
pre = pd.read_csv(f"{rd}/gec/norm/gec.ctm", sep=r"\s+", header=None, keep_default_na=False, engine="python")
B = {}
for line in open(a.bart):
    r = json.loads(line); B[r["id"]] = r

ver = defaultdict(int)
# phrase-level: BART arrays vs CTM word counts; S subsequence of A (pre-DD) per phrase
n_flt = flt_all.groupby(0).size().to_dict(); n_gec = gec_all.groupby(0).size().to_dict()
for pid, r in B.items():
    ver["phrases"] += 1
    ver["phrase_src_count_mismatch"] += int(r["n_src"] != n_flt.get(pid, 0))
    ver["phrase_tgt_count_mismatch"] += int(r["n_tgt"] != n_gec.get(pid, 0))
pre_by = defaultdict(list)
for pid, w, s, d, c in zip(pre[0], pre[4], pre[2], pre[3], pre[5]):
    pre_by[pid].append((str(w), float(s), float(d), float(c)))
a_idx = {}           # (pid, flt widx) -> index in pre-DD phrase
del_before = {}      # (pid, flt widx) -> list of conf of pre-DD words deleted between previous kept word and this one
del_after_last = {}  # pid -> conf of deleted words after the last kept word
for pid, g in flt_all.groupby(0, sort=False):
    P = pre_by[pid]; j = 0; prev = -1
    for widx, w, s in zip(g["widx"], g[4], g[2]):
        while j < len(P) and not (P[j][0] == str(w) and abs(P[j][1] - float(s)) < 1e-6):
            j += 1
        if j == len(P):
            ver["flt_word_not_in_predd"] += 1; break
        a_idx[(pid, widx)] = j
        del_before[(pid, widx)] = [x[3] for x in P[prev + 1:j]]
        prev = j; j += 1
    del_after_last[pid] = [x[3] for x in P[prev + 1:]]
ver["phrases_all_deleted"] = sum(1 for pid in pre_by if pid not in n_flt)

ed = f"{rd}/gec/sgecf-scoring/errant"
off_flt, off_gec = SE.get_file_data(f"{ed}/flt_ctm.txt"), SE.get_file_data(f"{ed}/gec_ctm.txt")
ids = list(pd.read_csv(f"{a.labels}/utt_index_{a.ds}.csv")["utt_id"])
fu = {u: g for u, g in flt.groupby("Base_ID", sort=False)}
gu = {u: g for u, g in gec.groupby("Base_ID", sort=False)}
U = {}
for i, u in enumerate(ids):
    if u in fu:
        st, toks, t2w, words = utt_tokens(fu[u])
    else:
        st, toks, t2w, words = "nan", ["nan"], [0], []
        ver["utts_without_flt_words"] += 1
    ver["flt_line_mismatch"] += int(st != off_flt[i])
    if u in gu:
        gst, gtoks, gt2w, gwords = utt_tokens(gu[u])
    else:
        gst, gtoks, gt2w, gwords = "nan", ["nan"], [0], []
        ver["utts_without_gec_words"] += 1
    ver["gec_line_mismatch"] += int(gst != off_gec[i])
    U[u] = dict(S=toks, t2w=t2w, words=words, G=gtoks, gt2w=gt2w, gwords=gwords,
                frows=fu[u] if u in fu else None, grows=gu[u] if u in gu else None)

lab = pd.read_csv(f"{a.labels}/hyp_edit_labels_{a.ds}.csv")
ref = pd.read_csv(f"{a.labels}/ref_edit_labels_{a.ds}.csv")
n_edits_utt = lab.groupby("utt_id").size().to_dict()


def fmin(x):
    x = [v for v in x if v == v]
    return min(x) if x else nan


def fmean(x):
    x = [v for v in x if v == v]
    return sum(x) / len(x) if x else nan


rows = []
utt_cache = {}
for e in lab.itertuples(index=False):
    u = U[e.utt_id]
    fr, gr = u["frows"], u["grows"]
    if fr is None or u["words"] == []:
        ver["edits_in_empty_utts"] += 1
    nS = len(u["S"])
    # utterance-level ASR / DD features
    if e.utt_id not in utt_cache:
        if fr is not None:
            confs = [float(c) for c in fr[5]]
            t0, t1 = float(fr[2].min()), float((fr[2].astype(float) + fr[3].astype(float)).max())
            pids = list(dict.fromkeys(fr[0]))
            n_pre = sum(len(pre_by[p]) for p in pids); n_kept = sum(n_flt.get(p, 0) for p in pids)
            utt_cache[e.utt_id] = dict(utt_asr_conf_mean=fmean(confs), utt_asr_conf_min=fmin(confs),
                                       utt_asr_lowconf_frac=sum(c < 0.5 for c in confs) / len(confs),
                                       utt_words=len(confs), utt_rate_wps=len(confs) / max(t1 - t0, 1e-3),
                                       utt_dd_del_rate=(n_pre - n_kept) / max(n_pre, 1))
        else:
            utt_cache[e.utt_id] = dict(utt_asr_conf_mean=nan, utt_asr_conf_min=nan, utt_asr_lowconf_frac=nan,
                                       utt_words=0, utt_rate_wps=nan, utt_dd_del_rate=nan)
    f = dict(utt_cache[e.utt_id])
    # source side (system fluent words)
    o_tok = list(range(e.o_start, e.o_end))
    w_tok = list(range(max(0, e.o_start - 1), min(nS, e.o_end + 1)))
    o_w = sorted(set(u["t2w"][t] for t in o_tok)) if fr is not None else []
    win_w = sorted(set(u["t2w"][t] for t in w_tok)) if fr is not None else []
    if fr is not None and o_tok:
        txt = "".join(u["words"][k] for k in o_w)
        ver["o_span_text_not_in_words"] += int("".join(u["S"][t] for t in o_tok) not in txt)
    W = lambda ks: [fr.iloc[k] for k in ks] if fr is not None else []
    conf = lambda ks: [float(r[5]) for r in W(ks)]
    dur = lambda ks: [float(r[3]) for r in W(ks)]
    ctx2 = sorted(set(range(max(0, (min(o_w) if o_w else (min(win_w) if win_w else 0)) - 2),
                            min(len(u["words"]), (max(o_w) if o_w else (max(win_w) if win_w else -1)) + 3))))
    f.update(asr_conf_o_min=fmin(conf(o_w)), asr_conf_o_mean=fmean(conf(o_w)), asr_conf_win_min=fmin(conf(win_w)),
             asr_conf_win_mean=fmean(conf(win_w)), asr_conf_ctx2_min=fmin(conf(ctx2)),
             asr_dur_win_min=fmin(dur(win_w)), asr_dur_o_mean=fmean(dur(o_w)))
    # DD deletions inside / adjacent to the window
    inside, adjacent = [], []
    rws = W(win_w)
    for n, r in enumerate(rws):
        d = del_before.get((r[0], r["widx"]), [])
        (inside if n > 0 else adjacent).extend(d)
    if rws:
        last = rws[-1]; nxt = fr[(fr[0] == last[0]) & (fr["widx"] == last["widx"] + 1)]
        adjacent.extend(del_before.get((last[0], last["widx"] + 1), []) if len(nxt) else del_after_last.get(last[0], []))
    f.update(dd_del_inside=len(inside), dd_del_adjacent=len(adjacent), dd_del_conf_min=fmin(inside + adjacent))
    # BART copy scores of source words
    s_lp = lambda ks: [B[r[0]]["s_lp"][r["widx"]] for r in W(ks)]
    s_min = lambda ks: [B[r[0]]["s_min"][r["widx"]] for r in W(ks)]
    f.update(gec_copy_lp_o_mean=fmean(s_lp(o_w)), gec_copy_lp_o_min=fmin(s_min(o_w)), gec_copy_lp_win_min=fmin(s_min(win_w)))
    # target side (system GEC words)
    nG = len(u["G"])
    c_tok = list(range(e.c_start, e.c_end))
    c_w = sorted(set(u["gt2w"][t] for t in c_tok)) if gr is not None else []
    nxt_w = [u["gt2w"][e.c_end]] if gr is not None and e.c_end < nG else []
    GW = lambda ks: [gr.iloc[k] for k in ks] if gr is not None else []
    if gr is not None and c_tok:
        txt = "".join(u["gwords"][k] for k in c_w)
        ver["c_span_text_not_in_words"] += int("".join(u["G"][t] for t in c_tok) not in txt)
    t_lp = lambda ks: [B[r[0]]["t_lp"][r["widx"]] for r in GW(ks)]
    t_min = lambda ks: [B[r[0]]["t_min"][r["widx"]] for r in GW(ks)]
    t_ent = lambda ks: [B[r[0]]["t_ent"][r["widx"]] for r in GW(ks)]
    t_ag = lambda ks: [B[r[0]]["t_agree"][r["widx"]] for r in GW(ks)]
    cs, os_ = sum(t_lp(c_w)), sum(s_lp(o_w))
    phr = (W(o_w) or W(win_w) or [None])[0]
    pr = B[phr[0]] if phr is not None else None
    f.update(gec_lp_c_mean=fmean(t_lp(c_w)), gec_lp_c_min=fmin(t_min(c_w)), gec_ent_c_mean=fmean(t_ent(c_w)),
             gec_agree_c_all=(min(t_ag(c_w)) if c_w else nan), gec_lp_next=fmean(t_lp(nxt_w)),
             gec_ent_next=fmean(t_ent(nxt_w)), gec_lp_ratio=cs - os_,
             phrase_lp_per_word=(pr["seq_lp"] / max(pr["n_tgt"], 1)) if pr else nan,
             phrase_copy_gap_per_word=((pr["seq_lp"] - pr["seq_copy_lp"]) / max(pr["n_src"], 1)) if pr else nan,
             phrase_len_words=pr["n_src"] if pr else nan,
             edit_at_phrase_edge=int(any(r["widx"] in (0, pr["n_src"] - 1) for r in W(win_w))) if pr else 0)
    # edit descriptors (ERRANT on system texts)
    f.update(op=e.op, type_pos=e.type.split(":", 1)[1] if ":" in e.type else e.type, part=e.part,
             o_len=e.o_len, c_len=e.c_len, n_src_tokens=e.n_src_tokens, rel_pos=e.o_start / max(nS, 1),
             n_edits_utt=n_edits_utt[e.utt_id])
    rows.append(dict(ds=a.ds, utt_id=e.utt_id, edit_idx=e.edit_idx, group="-".join(e.utt_id.split("-")[:2]),
                     type=e.type, false_feedback=e.false_feedback, asr_induced=e.asr_induced, dd_induced=e.dd_induced,
                     gec_induced=e.gec_induced, asr_induced_lit=e.asr_induced_lit, **f))

df = pd.DataFrame(rows)
# n_edits per phrase (inference time)
os.makedirs(a.out_dir, exist_ok=True)
df.to_csv(f"{a.out_dir}/features_{a.ds}.csv", index=False)
uc = pd.DataFrame(dict(utt_id=ids))
uc["group"] = uc["utt_id"].str.split("-").str[:2].str.join("-")
uc["n_ref_edits"] = uc["utt_id"].map(ref.groupby("utt_id").size()).fillna(0).astype(int)
uc["n_hyp_edits"] = uc["utt_id"].map(lab.groupby("utt_id").size()).fillna(0).astype(int)
uc.to_csv(f"{a.out_dir}/utt_counts_{a.ds}.csv", index=False)
ver = dict(ver)
ver.update(edits=len(df), groups=int(df["group"].nunique()), nan_share={c: round(float(df[c].isna().mean()), 4)
           for c in df.columns if df[c].dtype != object and df[c].isna().any()})
bad = ["flt_line_mismatch", "gec_line_mismatch", "o_span_text_not_in_words", "c_span_text_not_in_words",
       "phrase_src_count_mismatch", "phrase_tgt_count_mismatch", "flt_word_not_in_predd"]
ver["ok"] = all(ver.get(k, 0) == 0 for k in bad)
json.dump(ver, open(f"{a.out_dir}/features_{a.ds}.verification.json", "w"), indent=1)
print(json.dumps(ver))
sys.exit(0 if ver["ok"] else 3)
