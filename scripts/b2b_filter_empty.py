#!/usr/bin/env python3
"""B2b fix: remove phrases that DD reduced to nothing, before GEC and scoring (official code unchanged).

When the official dd.py removes every word of a phrase (all-disfluent phrase), it writes
  - a pre-gec.tsv row with data_flt = [] and an empty sentence_flt (pandas reads it as NaN -> the official
    gec-plus-align.py tokenizer raises ValueError), and
  - one placeholder line in fluent.ctm with empty word/start/duration/prob (the official SGECF builder reads CTMs with
    keep_default_na=False, so an empty start would turn the time column into strings, and the empty word would add
    an empty token to the hypothesis).
Removing both is equivalent to "the system output for this phrase is empty": an empty phrase has no words before or
after GEC, so it cannot produce a GEC edit. B1/B2 contain no such phrases (checked by this script's counts).
Every other line is copied byte-identically (checked).
Usage: python3 -I b2b_filter_empty.py SRC_GEC_DIR OUT_GEC_DIR
Writes OUT/dd/pre-gec.tsv, OUT/fluent.ctm, OUT/dd/filtered_empty_phrases.tsv (phrase IDs only); prints a JSON summary.
"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
L = open(f"{src}/dd/pre-gec.tsv").read().split("\n")
assert L[-1] == "" and L[0].split("\t") == ["ids", "data_flt", "sentence_flt"]
rows = L[1:-1]
assert all(len(r.split("\t")) == 3 for r in rows)
empty = [r.split("\t")[0] for r in rows if r.split("\t")[2] == ""]
assert all(r.split("\t")[1] == "[]" for r in rows if r.split("\t")[2] == "")
assert all(r.split("\t")[1] != "[]" for r in rows if r.split("\t")[2] != "")
E = set(empty)
keep = [r for r in rows if r.split("\t")[0] not in E]
with open(f"{out}/dd/pre-gec.tsv", "w") as f:
    f.write("\n".join([L[0]] + keep) + "\n")

C = open(f"{src}/fluent.ctm").read().split("\n")
assert C[-1] == ""
C = C[:-1]
drop = [c for c in C if c.split("\t")[0] in E]
assert len(drop) == len(E), "expected exactly one placeholder line per empty phrase"
assert all(c.split("\t")[2:] == ["", "", "", ""] for c in drop), "placeholder lines must carry no word/time"
ckeep = [c for c in C if c.split("\t")[0] not in E]
with open(f"{out}/fluent.ctm", "w") as f:
    f.write("\n".join(ckeep) + "\n")
assert all(c.split("\t")[4] != "" for c in ckeep), "no other empty words may remain"

with open(f"{out}/dd/filtered_empty_phrases.tsv", "w") as f:
    f.write("phrase_id\n" + "".join(i + "\n" for i in sorted(E)))
print(json.dumps(dict(src=src, out=out, phrases_total=len(rows), phrases_empty_removed=len(E),
                      phrases_kept=len(keep), fluent_ctm_lines_total=len(C), fluent_ctm_placeholder_lines_removed=len(drop),
                      fluent_ctm_lines_kept=len(ckeep))))
