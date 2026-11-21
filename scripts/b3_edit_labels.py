#!/usr/bin/env python3
"""B3: per-edit TP/FP labels (and per-reference-edit missed/FN labels) for one stage-B run, reusing the
OFFICIAL Spoken ERRANT code (scoring-tools/sgecf-tools/spoken_errant.py, create_files_for_errant.py).

Run with the sandi-all env from the corpus root:
  python b3_edit_labels.py --run_dir stageB/B1_dev --ds dev --out_dir <$EXP_DIR/05_runs/...>/labels

Correctness proofs written to <out_dir>/b3_verification.json (the script exits non-zero if any fails):
  (1) utterance-ID alignment: the four aligned texts re-created with create_files_for_errant.py logic are identical,
      line by line, to the official errant/*.txt files written by run-sgecf-scoring.sh; the ID order comes from the same logic.
  (2) the per-edit labels aggregated per ERRANT type equal the tp/fp/fn dicts returned by the official
      spoken_errant.errant_spoken() on the same files (exact integer equality).
  (3) overall and per-type P/R/F0.5 from (2) rounded to 3 decimals equal the official result.txt.

Labels (v2, three-way; definitions also in 08_paper/methods_notes.md). Four token sequences per utterance, all
spaCy-tokenised exactly like create_files_for_errant.py and restricted to the scored region (fluent/GEC IGNORE
segments removed with the official get_non_scorable_words time rule):
  S = system fluent tokens (official errant flt_ctm line)        M = manual fluent tokens (official errant flt_stm line)
  A = system tokens BEFORE DD (run_dir/gec/norm/gec.ctm)         V = manual verbatim tokens (--verbatim_norm_ctm, from
                                                                     {ds}-asr.stm via the official task-input normalisation)
Token alignments (Levenshtein.opcodes, as used by Spoken ERRANT; manual transcripts have no word times):
  S<->A (DD deletions), A<->V (ASR errors), V<->M (manual disfluency removal).
Window = edit source span +-1 token (hypothesis edits: in S; reference edits: in M), mapped to a contiguous span in A
(resp. V), which therefore also contains the tokens deleted between the window tokens.
  asr_err : in the A-span (resp. V-span) a token differs between A and V, or a V token is missing from A
  dd_err  : for some correctly recognised token in the span (equal in A<->V), the DD decision (kept in S?) differs from
            the manual decision (kept in M?), i.e. a wrongly deleted fluent word or a missed disfluency
Hypothesis edits: label TP|FP (official matching). For FP: asr_induced = asr_err (mixed = asr_err and dd_err);
  dd_induced = dd_err and not asr_err; gec_induced = neither.
Reference edits: missed (= not matched by any hypothesis edit) split the same way: missed_asr / missed_dd / missed_gec, mixed.
Modes: asr (B1: real ASR + DD), verbatim (B2b: A comes from the manual verbatim transcript, so asr_err should be 0),
  fluent (B2: no ASR, no DD; asr_err = dd_err = 0 by construction; checked that S == M).
v3 (user decision 2026-11-10): PRIMARY asr_err does NOT count ASR omissions of verbatim tokens that the manual fluent
transcript also removes (disfluencies): asr_err = literal asr_err and not asr_err_disfl_only. The literal definition (any
A<->V mismatch) is kept as a sensitivity analysis in the *_lit columns (asr_err_window_lit, asr_induced_lit, dd_induced_lit,
gec_induced_lit, mixed_lit, missed_*_lit). asr_err_disfl_only = literal asr_err but not primary asr_err.
Official FN per type = ref_total - tp (tp counts hypothesis edits); our per-reference "missed" count is reported next to it.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

CORPUS = (os.environ.get("RUN_DIR") or __import__("sys").exit("RUN_DIR is not set: build a run root with make_run_root.sh and export RUN_DIR (README, Setup)"))
SG = os.path.join(CORPUS, "scoring-tools/sgecf-tools")
sys.path.insert(0, SG)
import spoken_errant as SE            # noqa: E402  official module
import create_files_for_errant as CF  # noqa: E402  official module (loads spaCy en_core_web_sm)
import Levenshtein                    # noqa: E402
import errant                         # noqa: E402


def rebuild_final(flt_stm, flt_ctm, gec_stm, gec_ctm):
    """Same steps as create_files_for_errant.main(), but returns the DataFrame incl. ID instead of writing files."""
    stm_flt = CF.load_stm(flt_stm)
    ign = stm_flt[stm_flt["Transcription"] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
    ctm_flt = pd.read_csv(flt_ctm, sep="\t", header=None, keep_default_na=False)
    ctm_flt["Base_ID"] = ctm_flt[0].str.rsplit("_", n=1).str[0]
    ignw = CF.get_non_scorable_words(ign, ctm_flt)
    f_ctm = ctm_flt.loc[~ctm_flt.index.isin(ignw.index)]
    f_stm = stm_flt.loc[~stm_flt.index.isin(ign.index)]
    flt_hyp = f_ctm.groupby("Base_ID")[4].apply(" ".join).reset_index(); flt_hyp.columns = ["ID", "flt_hyp"]
    flt_ref = f_stm[["ID", "Transcription"]].groupby("ID")["Transcription"].apply(" ".join).reset_index(); flt_ref.columns = ["ID", "flt_ref"]
    flt = pd.merge(left=flt_ref, right=flt_hyp, on="ID", how="left")
    stm_gec = CF.load_stm(gec_stm)
    ign = stm_gec[stm_gec["Transcription"] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
    ctm_gec = pd.read_csv(gec_ctm, sep="\t", header=None, keep_default_na=False)
    ctm_gec["Base_ID"] = ctm_gec[0].str.rsplit("_", n=1).str[0]
    ignw = CF.get_non_scorable_words(ign, ctm_gec)
    g_ctm = ctm_gec.loc[~ctm_gec.index.isin(ignw.index)]
    g_stm = stm_gec.loc[~stm_gec.index.isin(ign.index)]
    gec_hyp = g_ctm.groupby("Base_ID")[4].apply(" ".join).reset_index(); gec_hyp.columns = ["ID", "gec_hyp"]
    gec_ref = g_stm[["ID", "Transcription"]].groupby("ID")["Transcription"].apply(" ".join).reset_index(); gec_ref.columns = ["ID", "gec_ref"]
    gec = pd.merge(left=gec_ref, right=gec_hyp, on="ID", how="left")
    final = pd.merge(left=flt, right=gec, on="ID")
    for c in ["flt_ref", "flt_hyp", "gec_ref", "gec_hyp"]:
        final[c] = final[c].apply(lambda x: " ".join([y.text for y in CF.nlp(str(x))]))
    return final


def asr_shifts_for(stc_asr, stc_trans):
    """Copied verbatim from spoken_errant.errant_spoken (position-shift computation)."""
    import numpy as np
    asr_editops = Levenshtein.opcodes(stc_asr.split(" "), stc_trans.split(" "))
    asr_shifts = np.zeros((2, len(stc_asr.split(" ")) + 1))
    for opcode, asr_start, asr_stop, trans_start, trans_stop in asr_editops:
        if opcode == "insert":
            try:
                asr_shifts[0, asr_start] += trans_stop - trans_start
                asr_shifts[1, asr_start + 1] += trans_stop - trans_start
            except Exception:
                continue
        elif opcode == "delete":
            asr_shifts[:, asr_stop] += -(asr_stop - asr_start)
    return np.cumsum(np.array(asr_shifts), axis=1), asr_editops


class Align:
    """Token alignment x -> y with Levenshtein.opcodes (same aligner as Spoken ERRANT)."""

    def __init__(self, x, y):
        self.nx, self.ny = len(x), len(y)
        self.eq_xy, self.eq_yx = {}, {}          # exact (equal-block) correspondences
        self.ap_xy, self.ap_yx = {}, {}          # approximate position maps (for span mapping)
        self.bad_x, self.bad_y, self.gap_x, self.gap_y = set(), set(), set(), set()
        self.gap_x_tok = defaultdict(list)       # x-side gap position -> y tokens present only in y
        self.yonly = set()                       # y tokens with no counterpart in x (pure insertions w.r.t. x)
        for op, i1, i2, j1, j2 in Levenshtein.opcodes(x, y):
            if op == "equal":
                for k in range(i2 - i1):
                    self.eq_xy[i1 + k] = j1 + k; self.eq_yx[j1 + k] = i1 + k
                    self.ap_xy[i1 + k] = j1 + k; self.ap_yx[j1 + k] = i1 + k
                continue
            self.bad_x.update(range(i1, i2)); self.bad_y.update(range(j1, j2))
            if i2 == i1:
                self.gap_x.add(i1)
                self.gap_x_tok[i1].extend(range(j1, j2))
                self.yonly.update(range(j1, j2))
            if j2 == j1:
                self.gap_y.add(j1)
            for i in range(i1, i2):
                self.ap_xy[i] = min(j1 + (i - i1), max(j1, j2 - 1)) if j2 > j1 else min(j1, max(self.ny - 1, 0))
            for j in range(j1, j2):
                self.ap_yx[j] = min(i1 + (j - j1), max(i1, i2 - 1)) if i2 > i1 else min(i1, max(self.nx - 1, 0))


def span_in_target(o_start, o_end, n_src, ap_map):
    """Source window = [o_start-1, o_end+1) clipped; returns inclusive (lo, hi) in the target or None."""
    lo, hi = max(0, o_start - 1), min(n_src, o_end + 1)
    tgt = [ap_map[i] for i in range(lo, hi) if i in ap_map]
    return (min(tgt), max(tgt)) if tgt else None


def asr_err_in(span, bad, gap):
    if span is None:
        return False
    lo, hi = span
    return any(t in bad for t in range(lo, hi + 1)) or any(g in gap for g in range(lo, hi + 2))


def asr_err_real_A(span, av, keep_v):
    """Like asr_err_in on the A side, but ASR omissions of verbatim tokens that the manual fluent transcript also
    removes (disfluencies) are not counted (diagnostic only)."""
    if span is None:
        return False
    lo, hi = span
    if any(t in av.bad_x for t in range(lo, hi + 1)):
        return True
    return any(any(keep_v(v) for v in av.gap_x_tok[g]) for g in range(lo, hi + 2) if g in av.gap_x)


def asr_err_real_V(span, av, keep_v):
    if span is None:
        return False
    lo, hi = span
    if any(v in av.bad_y and not (v in av.yonly and not keep_v(v)) for v in range(lo, hi + 1)):
        return True
    return any(g in av.gap_y for g in range(lo, hi + 2))


def tokens_from_ctm(ctm_path, ign_segments, ids):
    """Words of a 6-column phrase CTM grouped per utterance, scored region only (official time rule), spaCy-tokenised."""
    df = pd.read_csv(ctm_path, sep=r"\s+", header=None, keep_default_na=False, engine="python")
    df["Base_ID"] = df[0].str.rsplit("_", n=1).str[0]
    ignw = CF.get_non_scorable_words(ign_segments, df) if len(ign_segments) else df.iloc[0:0]
    df = df.loc[~df.index.isin(ignw.index)]
    joined = df.groupby("Base_ID")[4].apply(lambda w: " ".join(map(str, w))).to_dict()
    return {u: " ".join(t.text for t in CF.nlp(joined.get(u, ""))) for u in ids}


def part_of(uid):
    m = re.search(r"-P(\d)", uid)
    return f"P{m.group(1)}" if m else "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--mode", required=True, choices=["asr", "verbatim", "fluent"])
    ap.add_argument("--verbatim_norm_ctm", required=True, help="stageB/norm_verbatim_<ds>/gec/norm/gec.ctm")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.chdir(CORPUS)
    rd = a.run_dir.rstrip("/")
    stm = lambda k: f"reference-materials/stms/{a.ds}-{k}.stm"
    final = rebuild_final(stm("fluent"), f"{rd}/gec/fluent.ctm", stm("gec"), f"{rd}/gec/gec.ctm")
    ed = f"{rd}/gec/sgecf-scoring/errant"
    official = {k: SE.get_file_data(f"{ed}/{k}.txt") for k in ["flt_stm", "flt_ctm", "gec_stm", "gec_ctm"]}
    ver = {"run_dir": rd, "ds": a.ds, "mode": a.mode, "label_version": "v3 three-way (asr/dd/gec); primary asr_err excludes disfluency-only omissions; *_lit = literal"}
    ver["align_identical"] = {k: list(final[c]) == official[k] for k, c in
                              [("flt_stm", "flt_ref"), ("flt_ctm", "flt_hyp"), ("gec_stm", "gec_ref"), ("gec_ctm", "gec_hyp")]}
    ids = list(final["ID"])

    stm_flt = CF.load_stm(stm("fluent"))
    ign = stm_flt[stm_flt["Transcription"] == "IGNORE_TIME_SEGMENT_IN_SCORING"]
    A_txt = tokens_from_ctm(f"{rd}/gec/norm/gec.ctm", ign, ids)
    V_txt = tokens_from_ctm(a.verbatim_norm_ctm, ign, ids)

    o_tp, o_fp, o_fn = SE.errant_spoken(official["flt_ctm"], official["gec_ctm"], official["gec_stm"], official["flt_stm"])

    annotator = errant.load("en")
    hyp_rows, ref_rows = [], []
    tp, fp, ref_tot = defaultdict(int), defaultdict(int), defaultdict(int)
    diag = defaultdict(int)
    for i, (stc_asr, stc_pred, stc_tgt, stc_trans) in enumerate(zip(official["flt_ctm"], official["gec_ctm"],
                                                                     official["gec_stm"], official["flt_stm"])):
        uid = ids[i]
        hyp_edits = SE.get_stc_edits(annotator, stc_asr, stc_pred)
        ref_edits = SE.get_stc_edits(annotator, stc_trans, stc_tgt)
        for r in ref_edits:
            ref_tot[r.type] += 1
        shifts, _ = asr_shifts_for(stc_asr, stc_trans)
        S, M = stc_asr.split(" "), stc_trans.split(" ")
        A, V = A_txt[uid].split(" ") if A_txt[uid] else [], V_txt[uid].split(" ") if V_txt[uid] else []
        sa, av, vm = Align(S, A), Align(A, V), Align(V, M)
        diag["S_tokens"] += len(S); diag["S_tokens_equal_in_A"] += len(sa.eq_xy)
        diag["M_tokens"] += len(M); diag["M_tokens_equal_in_V"] += len(vm.eq_yx)
        diag["A_tokens"] += len(A); diag["A_tokens_differing_from_V"] += len(av.bad_x)
        diag["V_tokens"] += len(V); diag["V_tokens_missing_or_differing_in_A"] += len(av.bad_y)
        diag["utts_S_ne_M"] += int(S != M)
        manual_keep = lambda v: v in vm.eq_xy                  # verbatim token v kept in manual fluent
        dd_keep = lambda x: x in sa.eq_yx                      # pre-DD token x kept in system fluent

        def dd_err_A(span):
            if span is None:
                return False
            return any(manual_keep(av.eq_xy[x]) != dd_keep(x) for x in range(span[0], span[1] + 1) if x in av.eq_xy)

        def dd_err_V(span):
            if span is None:
                return False
            return any(manual_keep(v) != dd_keep(av.eq_yx[v]) for v in range(span[0], span[1] + 1) if v in av.eq_yx)

        ref_matched = [0] * len(ref_edits)
        for j, h in enumerate(hyp_edits):
            match = None
            for k, r in enumerate(ref_edits):
                if SE.equal_edits(h, r, shifts):
                    match = k
                    break
            if match is not None:
                tp[ref_edits[match].type] += 1
                ref_matched[match] += 1
            else:
                fp[h.type] += 1
            if a.mode == "fluent":
                ae = de = ae_real = False
            else:
                spanA = span_in_target(h.o_start, h.o_end, len(S), sa.ap_xy)
                ae, de = asr_err_in(spanA, av.bad_x, av.gap_x), dd_err_A(spanA)
                ae_real = asr_err_real_A(spanA, av, manual_keep)
            is_fp = match is None
            hyp_rows.append(dict(utt_id=uid, part=part_of(uid), utt_idx=i, edit_idx=j, o_start=h.o_start, o_end=h.o_end,
                                 c_start=h.c_start, c_end=h.c_end, n_src_tokens=len(S),
                                 o_len=h.o_end - h.o_start, c_len=h.c_end - h.c_start,
                                 type=h.type, op=h.type.split(":")[0], label="FP" if is_fp else "TP",
                                 matched_ref_idx=-1 if is_fp else match, false_feedback=int(is_fp),
                                 asr_err_window=int(ae_real), dd_err_window=int(de),
                                 asr_err_disfl_only=int(ae and not ae_real),
                                 # PRIMARY (v3): ASR omission of disfluencies is not an ASR error
                                 asr_induced=int(is_fp and ae_real), dd_induced=int(is_fp and de and not ae_real),
                                 gec_induced=int(is_fp and not ae_real and not de), mixed=int(is_fp and ae_real and de),
                                 # SENSITIVITY (literal definition: any A<->V mismatch, incl. omitted disfluencies)
                                 asr_err_window_lit=int(ae),
                                 asr_induced_lit=int(is_fp and ae), dd_induced_lit=int(is_fp and de and not ae),
                                 gec_induced_lit=int(is_fp and not ae and not de), mixed_lit=int(is_fp and ae and de)))
        for k, r in enumerate(ref_edits):
            if a.mode == "fluent":
                ae = de = ae_real = False
            else:
                spanV = span_in_target(r.o_start, r.o_end, len(M), vm.ap_yx)
                ae, de = asr_err_in(spanV, av.bad_y, av.gap_y), dd_err_V(spanV)
                ae_real = asr_err_real_V(spanV, av, manual_keep)
            missed = ref_matched[k] == 0
            ref_rows.append(dict(utt_id=uid, part=part_of(uid), utt_idx=i, ref_edit_idx=k, o_start=r.o_start, o_end=r.o_end,
                                 type=r.type, op=r.type.split(":")[0], n_matches=ref_matched[k],
                                 detected=int(not missed), missed=int(missed), asr_err_window=int(ae_real), dd_err_window=int(de),
                                 asr_err_disfl_only=int(ae and not ae_real),
                                 missed_asr=int(missed and ae_real), missed_dd=int(missed and de and not ae_real),
                                 missed_gec=int(missed and not ae_real and not de), mixed=int(missed and ae_real and de),
                                 asr_err_window_lit=int(ae),
                                 missed_asr_lit=int(missed and ae), missed_dd_lit=int(missed and de and not ae),
                                 missed_gec_lit=int(missed and not ae and not de), mixed_lit=int(missed and ae and de)))
    fn = {k: ref_tot[k] - tp[k] for k in ref_tot}

    norm = lambda d: {k: int(v) for k, v in d.items() if v}
    ver["counts_equal_official"] = {"tp": norm(tp) == norm(o_tp), "fp": norm(fp) == norm(o_fp), "fn": norm(fn) == norm(o_fn)}
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P, R, F05 = SE.compute_p_r_f05(T, F, N)   # official formula (float operation order matters for %.3f ties)
    txt = open(f"{rd}/gec/sgecf-scoring/result.txt").read()
    o = re.search(r"Overall results:\nP:\tR:\tF0\.5\n([\d.]+)\t([\d.]+)\t([\d.]+)", txt).groups()
    ver["overall_from_labels"] = {"tp": T, "fp": F, "fn_official": N, "P": round(P, 3), "R": round(R, 3), "F05": round(F05, 3)}
    ver["overall_result_txt"] = {"P": float(o[0]), "R": float(o[1]), "F05": float(o[2])}
    ver["overall_equal_result_txt"] = ("%.3f" % P, "%.3f" % R, "%.3f" % F05) == o
    per_type_ok = True
    for t, vals in re.findall(r"Edit Type:\t(\S+)\nP:\tR:\tF0\.5\n([\d.]+\t[\d.]+\t[\d.]+)", txt):
        tt, ff, nn = tp[t], fp[t], fn.get(t, 0)
        try:
            p_, r_, f_ = SE.compute_p_r_f05(tt, ff, nn)   # official formula, as print_results
            per_type_ok &= ("%.3f\t%.3f\t%.3f" % (p_, r_, f_)) == vals
        except ZeroDivisionError:
            pass
    ver["per_type_equal_result_txt"] = per_type_ok
    ver["missed_vs_official_fn"] = {"missed_ref_edits": sum(x["missed"] for x in ref_rows), "official_fn_total": N,
                                    "ref_edits_matched_more_than_once": sum(1 for x in ref_rows if x["n_matches"] > 1)}
    ver["alignment_diagnostics"] = dict(diag)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(hyp_rows).to_csv(f"{a.out_dir}/hyp_edit_labels_{a.ds}.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(f"{a.out_dir}/ref_edit_labels_{a.ds}.csv", index=False)
    pd.DataFrame({"utt_idx": range(len(ids)), "utt_id": ids}).to_csv(f"{a.out_dir}/utt_index_{a.ds}.csv", index=False)
    json.dump(ver, open(f"{a.out_dir}/b3_verification.json", "w"), indent=1)
    print(json.dumps(ver))
    ok = all(ver["align_identical"].values()) and all(ver["counts_equal_official"].values()) and ver["overall_equal_result_txt"] and per_type_ok
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
