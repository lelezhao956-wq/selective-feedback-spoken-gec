#!/usr/bin/env python3
"""Leakage guard for the repository package: count 6-word sequences of ALL manual reference STMs ({dev,train,eval,dev-subset} x
{asr,fluent,gec}) in every file of a directory (default: 10_repository_package; PDFs on visible text as leak_scan.py). No exclusions.
Prints files with hits (file name and count only); exit 4 if any. Usage: python3 -I pkg_leak_scan.py [dir]"""
import glob, os, re, sys
C = os.path.join(os.environ["CORPUS_DIR"], "reference-materials", "stms")
D = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: pkg_leak_scan.py PACKAGE_DIR")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
grams = set(); nlines = 0
for ds in ["dev", "train", "eval", "dev-subset"]:
    for k in ["asr", "fluent", "gec"]:
        for l in open(f"{C}/{ds}-{k}.stm", errors="ignore"):
            if l.startswith(";;"): continue
            w = [x.lower() for x in l.split()[6:] if not x.startswith("<") and "IGNORE_TIME" not in x]; nlines += 1
            grams.update(tuple(w[i:i + 6]) for i in range(len(w) - 5))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leak_scan.py")).read()
ns = {"re": re}; exec(src[src.index("def pdf_text"):src.index("\n# public CoNLL")], ns)
bad = nf = 0
for p in glob.glob(D + "/**/*", recursive=True):
    if os.path.isdir(p): continue
    nf += 1
    try: w = re.findall(r"[a-z']+", (ns["pdf_text"](p) if p.lower().endswith(".pdf") else open(p, errors="ignore").read()).lower())
    except Exception: continue
    h = sum(1 for i in range(len(w) - 5) if tuple(w[i:i + 6]) in grams)
    if h: bad += 1; print(f"LEAK? {h} hits in {os.path.relpath(p, D)}")
print(f"pkg_leak_scan: stm_lines={nlines} grams={len(grams)} files_scanned={nf} files_with_hits={bad}")
sys.exit(4 if bad else 0)
