#!/usr/bin/env python3
"""B2 step 3: "identity" disfluency detection for MANUAL fluent transcripts.

The fluent STM already has disfluencies removed, so DD is skipped. This script writes the two
files that baseline-pipeline/gec-tools/dd.py would write, in exactly the same formats, but with
every word kept:
  --fluent_ctm : tab-separated  id, 1, start, dur, word, prob   (numbers as Python floats)
  --gec_tsv    : header "ids data_flt sentence_flt"; data_flt = repr(list of (word, start, dur, prob))
Rows are ordered by phrase id (dd.py groups with pandas groupby, which sorts by id).

Input: the normalised phrase CTM from run-asr-to-task-input.sh --task gec (norm/gec.ctm),
whitespace separated: phrase_id chan start dur word prob.

Usage: python -I identity_dd.py --input_ctm norm/gec.ctm --fluent_ctm fluent.ctm --gec_tsv dd/pre-gec.tsv
"""
import argparse
import csv
import os
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ctm", required=True)
    ap.add_argument("--fluent_ctm", required=True)
    ap.add_argument("--gec_tsv", required=True)
    a = ap.parse_args()

    phrases = OrderedDict()
    for line in open(a.input_ctm, encoding="utf-8"):
        f = line.split()
        if len(f) < 6:
            continue
        phrases.setdefault(f[0], []).append((f[4], float(f[2]), float(f[3]), float(f[5])))

    os.makedirs(os.path.dirname(os.path.abspath(a.gec_tsv)), exist_ok=True)
    ids = sorted(phrases)  # pandas groupby order
    with open(a.fluent_ctm, "w", encoding="utf-8") as fc:
        for pid in ids:
            for w, s, d, p in phrases[pid]:
                fc.write(f"{pid}\t1\t{s}\t{d}\t{w}\t{p}\n")
    with open(a.gec_tsv, "w", encoding="utf-8", newline="") as ft:
        wr = csv.writer(ft, delimiter="\t", lineterminator="\n")
        wr.writerow(["ids", "data_flt", "sentence_flt"])
        for pid in ids:
            data = phrases[pid]
            wr.writerow([pid, repr(data), " ".join(w for w, *_ in data)])
    print(f"phrases={len(ids)} words={sum(len(v) for v in phrases.values())}")


if __name__ == "__main__":
    main()
