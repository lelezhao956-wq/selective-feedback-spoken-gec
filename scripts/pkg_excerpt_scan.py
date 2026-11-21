#!/usr/bin/env python3
"""Excerpt scan of the repository package against the corpus manual transcripts (licence clause: excerpts of < 100 words).
For every text file of PACKAGE_DIR, finds maximal runs of consecutive words in which every 3-word window occurs in a manual
reference STM ({dev,train,eval,dev-subset} x {asr,fluent,gec}). A run of length L >= MIN (default 5) is reported by location
and length only; no text of the package line or of the corpus is printed. Runs >= 100 words violate the
licence and must be removed. Note: short runs are often ordinary English shared by chance (e.g. in code comments).
Usage: CORPUS_DIR=<corpus root> python3 -I pkg_excerpt_scan.py PACKAGE_DIR [MIN] > report.tsv   (exit 4 if any run >= 100)"""
import os, re, sys
C = os.path.join(os.environ["CORPUS_DIR"], "reference-materials", "stms")
D = sys.argv[1]; MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tri = set()
for ds in ["dev", "train", "eval", "dev-subset"]:
    for k in ["asr", "fluent", "gec"]:
        for l in open(f"{C}/{ds}-{k}.stm", errors="ignore"):
            if l.startswith(";;"): continue
            w = [x.lower() for x in l.split()[6:] if not x.startswith("<") and "IGNORE_TIME" not in x]
            tri.update(tuple(w[i:i + 3]) for i in range(len(w) - 2))
TOK = re.compile(r"[a-z']+")
longest = 0; n_runs = 0; files = 0
print("file\tline\trun_words")
for root, ds, fs in os.walk(D):
    for n in sorted(fs):
        p = os.path.join(root, n)
        try: text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError): continue
        files += 1
        low = text.lower(); toks = [(m.start(), m.end(), m.group()) for m in TOK.finditer(low)]
        w = [t[2] for t in toks]; hit = [tuple(w[i:i + 3]) in tri for i in range(len(w) - 2)]
        i = 0
        while i < len(hit):
            if not hit[i]: i += 1; continue
            j = i
            while j + 1 < len(hit) and hit[j + 1]: j += 1
            L = j - i + 3   # words covered by windows i..j
            if L >= MIN:
                line = text.count("\n", 0, toks[i][0]) + 1
                print(f"{os.path.relpath(p, D)}\t{line}\t{L}"); n_runs += 1; longest = max(longest, L)
            i = j + 1
print(f"# pkg_excerpt_scan: files={files} runs>={MIN}: {n_runs} longest={longest}", file=sys.stderr)
sys.exit(4 if longest >= 100 else 0)
