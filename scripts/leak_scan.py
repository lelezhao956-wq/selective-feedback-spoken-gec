#!/usr/bin/env python3
"""Leakage guard: count 6-word sequences from the manual reference STMs ({dev,train,dev-subset}-{asr,fluent,gec}.stm)
in every file of $EXP_DIR (excluding .git and 05_runs/*/outputs/ and 05_runs/D_*rehearsal/*/outputs/, which hold Whisper-derived model outputs that naturally
overlap with correct transcripts). Prints files with hits; exit 4 if any. Never deletes anything.
Usage: python3 -I leak_scan.py"""
import glob, os, re, sys
C = os.path.join(os.environ["CORPUS_DIR"], "reference-materials", "stms")
E = (os.environ.get("EXP_DIR") or __import__("sys").exit("EXP_DIR is not set: export EXP_DIR=<your work directory> (README, Setup)"))
grams = set()
for ds in ["dev", "train", "dev-subset"]:
    for k in ["asr", "fluent", "gec"]:
        for l in open(f"{C}/{ds}-{k}.stm", errors="ignore"):
            if l.startswith(";;"): continue
            w = [x.lower() for x in l.split()[6:] if not x.startswith("<") and "IGNORE_TIME" not in x]
            grams.update(tuple(w[i:i + 6]) for i in range(len(w) - 5))
def pdf_text(path):
    """Visible text of a PDF without external tools (no pdftotext/pypdf on this Mac): inflate the streams, keep only the
    strings shown by text operators (Tj, ', ", TJ) inside BT..ET blocks; glyph names (/one /two ...), font programs and
    other operators are dropped. Added 2026-11-11 (glyph-name false positives in matplotlib figure PDFs)."""
    import zlib
    raw = open(path, "rb").read(); out = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        d = m.group(1)
        try: d = zlib.decompress(d)
        except Exception: pass
        for bt in re.findall(rb"\bBT\b(.*?)\bET\b", d, re.S):
            for arr, lit in re.findall(rb"\[((?:\\.|[^\]\\])*)\]\s*TJ|\(((?:\\.|[^()\\])*)\)\s*(?:Tj|'|\")", bt, re.S):
                if arr: out.append(b"".join(re.findall(rb"\(((?:\\.|[^()\\])*)\)", arr)))
                else: out.append(lit)
    t = b" ".join(out)
    t = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8) & 255]), t)
    t = re.sub(rb"\\(.)", rb"\1", t)
    return t.decode("latin-1")


# public documents that contain no corpus content (exclusions decided by the authors): exact basenames, one per line, in the
# environment variable LEAK_SCAN_EXCLUDE_FILES (the study excluded one public guide-for-authors PDF kept in the working tree)
# public CoNLL-2014/NUCLE data (GRECO reproduction check; NOT corpus-derived; must never enter the repository package: NUCLE licence)
EXCLUDE_DIRS = ("/05_runs/2026-11-10_G_greco/repro_20261111/",)
EXCLUDE_FILES = {x for x in os.environ.get("LEAK_SCAN_EXCLUDE_FILES", "").split("\n") if x}
bad = 0
for p in glob.glob(E + "/**/*", recursive=True):
    if os.path.isdir(p) or os.path.basename(p) in EXCLUDE_FILES or any(d in p for d in EXCLUDE_DIRS) or "/.git/" in p or re.search(r"/05_runs/(?:D_[a-z]*rehearsal/)?[^/]+/outputs/", p): continue
    try: w = re.findall(r"[a-z']+", (pdf_text(p) if p.lower().endswith(".pdf") else open(p, errors="ignore").read()).lower())
    except Exception: continue
    h = sum(1 for i in range(len(w) - 5) if tuple(w[i:i + 6]) in grams)
    if h: bad += 1; print(f"LEAK? {h} hits in {p[len(E)+1:]}")
print(f"leak_scan: files_with_hits={bad}")
sys.exit(4 if bad else 0)
