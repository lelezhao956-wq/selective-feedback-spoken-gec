#!/usr/bin/env python3
"""Stage D: repetition / hallucination counts of ASR output, per phrase of the official GEC task input (norm/gec.ctm, phrase IDs
<file>_PH<n>). Counts only (no text). Usage (any python3, corpus root):
  python3 -I d_asr_stats.py --ctm W=stageD/W_dev/gec/norm/gec.ctm --ctm small=stageB/B1_dev/gec/norm/gec.ctm \
     --ctm verbatim_ref=stageB/norm_verbatim_v4_dev/gec/norm/gec.ctm --out <json>
Per system: phrases, words; phrases with the same word >= 4 times in a row (the authors' criterion), and >= 8; phrases with a
2-4-word sequence repeated >= 4 times in a row (looping); longest run; words inside runs >= 4. The manual verbatim reference (markup
removed) is included as the natural rate of repetition in learner speech (its phrases are STM segments, so compare per 1,000 words).
"""
import argparse, json
from collections import OrderedDict
ap = argparse.ArgumentParser(); ap.add_argument("--ctm", action="append", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()


def phrases(p):
    P = OrderedDict()
    for l in open(p, errors="ignore"):
        f = l.split()
        if len(f) >= 5:
            P.setdefault(f[0], []).append(f[4].lower())
    return P


def runs(w):
    best = cur = 1 if w else 0; inrun = 0; r = 1
    for i in range(1, len(w) + 1):
        if i < len(w) and w[i] == w[i - 1]:
            r += 1
        else:
            if r >= 4:
                inrun += r
            best = max(best, r); r = 1
    return best, inrun


def ngram_loop(w, k=4):
    for n in (2, 3, 4):
        for s in range(len(w) - n * k + 1):
            g = w[s:s + n]
            if all(w[s + j * n:s + (j + 1) * n] == g for j in range(1, k)) and len(set(g)) > 1:
                return True
    return False


out = {}
for spec in a.ctm:
    name, p = spec.split("=", 1); P = phrases(p)
    n_w = sum(map(len, P.values())); r4 = r8 = loop = inrun = 0; longest = 0
    for w in P.values():
        b, ir = runs(w); longest = max(longest, b); inrun += ir; r4 += b >= 4; r8 += b >= 8; loop += ngram_loop(w)
    out[name] = dict(phrases=len(P), words=n_w, phrases_word_run_ge4=r4, phrases_word_run_ge8=r8, phrases_ngram_loop_ge4=loop,
                     longest_word_run=longest, words_in_runs_ge4=inrun,
                     run_ge4_per_1000_words=round(1000 * r4 / n_w, 3) if n_w else None)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out))
