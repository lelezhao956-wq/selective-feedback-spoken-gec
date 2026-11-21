#!/usr/bin/env python3
"""Build the repository package (<RESEARCH_DIR>/10_repository_package/) from the research working tree.
Research directory -> package directory (LAYOUT below): 04_scripts -> scripts, 03_configs -> configs, 06_results -> results,
01_environment -> environment; then 10_package_src/ is overlaid on the package root (README.md, LICENSE, docs/, and the
package versions of scripts / configs / environment files, which replace the copies of the same name).
Excludes *.orig backups, __pycache__, *.pyc, .DS_Store; 05_runs, 07_figures, 08_paper, 09_* and all corpus-derived data are never
copied. Anonymises the copies: user name, home path, and the research, scripts and corpus directories (-> $HOME, $EXP_DIR,
$SCRIPTS_DIR, $CORPUS_DIR; patterns derived at run time, none hard-coded) and the e-mail addresses given on the command line.
Writes MANIFEST.sha256. Never deletes anything; the target must be new and empty.
Usage: RESEARCH_DIR=<research tree> [CORPUS_DIR=<corpus root>] python3 -I pkg_build.py [e-mail ...]"""
import hashlib, os, re, shutil, sys
E = os.environ.get("RESEARCH_DIR") or sys.exit("RESEARCH_DIR is not set (the research working tree that contains 03_configs/, 04_scripts/, ...)")
E = os.path.abspath(E); OUT = f"{E}/10_repository_package"
LAYOUT = [("04_scripts", "scripts", False), ("03_configs", "configs", False), ("06_results", "results", False),
          ("01_environment", "environment", True), ("10_package_src", "", True)]   # (research dir, package dir, recursive)
HOME = os.path.expanduser("~"); USER = os.path.basename(HOME)
EMAILS = [x for x in sys.argv[1:]]   # e-mail addresses to anonymise, passed on the command line (not stored in this script)
assert os.path.isdir(OUT) and not os.listdir(OUT), "target must exist and be empty"
excl = lambda n: n.endswith(".orig") or n == "__pycache__" or n.endswith(".pyc") or n == ".DS_Store"
copied = {}   # package path -> source path (a later source replaces an earlier one: 10_package_src overrides)
for src, dst, rec in LAYOUT:
    S = f"{E}/{src}"
    for root, ds, fs in os.walk(S):
        ds[:] = [d for d in ds if not excl(d)] if rec else []
        for n in sorted(fs):
            if not excl(n):
                copied[os.path.normpath(f"{OUT}/{dst}/{os.path.relpath(f'{root}/{n}', S)}")] = f"{root}/{n}"
for p, s in copied.items():
    os.makedirs(os.path.dirname(p), exist_ok=True); shutil.copy2(s, p)
n_override = sum(1 for s in copied.values() if "/10_package_src/" in s)
# anonymise text files in place (copies only)
PROJ, SCD = os.path.basename(E), "04_scripts"
CORP = os.path.basename(os.path.normpath(os.environ.get("CORPUS_DIR", ""))) or None
bases = (HOME + "/Desktop", "~/Desktop", "$HOME/Desktop", HOME, "~", "$HOME")
loc = sorted([(f"{b}/{PROJ}/{SCD}/", "$SCRIPTS_DIR/") for b in bases], key=lambda x: -len(x[0]))
loc += sorted([(f"{b}/{PROJ}", "$EXP_DIR") for b in bases], key=lambda x: -len(x[0]))
if CORP: loc += sorted([(f"{b}/{CORP}", "$CORPUS_DIR") for b in bases], key=lambda x: -len(x[0]))
loc += [(PROJ, "EXP_DIR")]
rep = loc + [(HOME, "$HOME"), (USER, "USER")] + [(e, "ANON_EMAIL") for e in EMAILS]
n_changed = 0
for p in copied:
    try: s = open(p, encoding="utf-8").read()
    except UnicodeDecodeError: continue
    t = re.sub(r'"/private/tmp/[^"]*/scratchpad/([^"]+)"', r'os.path.expanduser("~/tmp/\1")', s)   # temporary session dirs
    for a, b in rep:
        t = t.replace(a, b)
    if t != s:
        open(p, "w", encoding="utf-8").write(t); n_changed += 1
with open(f"{OUT}/MANIFEST.sha256", "w") as fh:   # sha256 of every file of the package (after anonymisation)
    for p in sorted(copied):
        fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}\n")
print(f"copied {len(copied)} files ({n_override} from 10_package_src, overriding or adding); anonymised {n_changed}; MANIFEST.sha256 written")
