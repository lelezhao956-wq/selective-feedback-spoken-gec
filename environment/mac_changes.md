# macOS (Apple Silicon, arm64) changes

Machine: macOS, Apple Silicon (arm64). Changes are kept to what is needed to run on a machine without CUDA.
Every original file is backed up next to it as `*.orig` (`diff -u X.orig X` shows the change).

## Code / script changes

| File | Line | Change | Reason |
|---|---|---|---|
| `check-pipeline.sh` | 18 | Added `--cpu` to the end of the `run-asr-decode.sh` call | Without `--cpu`, `run-asr-decode.sh` forces `device='cuda'` (lines 108-118), so Whisper fails on a Mac without CUDA |
| `baseline-pipeline/gec-tools/gec-plus-align.py` | 154-155 | `torch.cuda.set_device(1)` now runs only when `torch.cuda.is_available()` is true | It was called unconditionally, so a torch build without CUDA throws right away; the next line already falls back to `cpu` |
| `baseline-pipeline/gec-tools/dd.py` | 137 | `torch.load(disfluency_model_path)` → `torch.load(disfluency_model_path, map_location=device)` | If the model was saved on CUDA, loading it on a machine without CUDA errors out unless it is remapped to the current device |
| `baseline-pipeline/sla-tools/inference_sla.py` | 65 | `torch.load(args.model)` → `torch.load(args.model, map_location=device)` | Same as above (`device` comes from `get_default_device()`, line 55) |
| `envs/sandi-all-env.sh` | 4 | `conda_env_path=$HOME/miniconda3` | Required by README-install-dataset.txt step 3 |
| `envs/sandi-dd-env.sh` | 4 | `conda_env_path=$HOME/miniconda3` | Same as above |

Note: `dd.py` and `inference_sla.py` already fall back to CPU when choosing a device; the only change there is adding map_location to torch.load. MPS was not enabled (CPU chosen), because the reference results come from CUDA and CPU fp32 is easier to compare against them.

## Environment differences (compared with envs/README.txt)

- Miniconda: `Miniconda3-latest-MacOSX-arm64.sh` (sha256 checked against the official listing), installed to `~/miniconda3` in `-b` mode, without changing shell rc files.
- Conda channel: both envs were created with `--override-channels -c conda-forge`, which avoids accepting the Anaconda default channels' ToS. The pip packages are unaffected.
- sandi-all: `openai-whisper==20231117` only ships as an sdist, and its setup.py needs `pkg_resources` (removed from setuptools >= 81).
  Workaround: `pip install "setuptools<81" wheel` into the env, `pip install --no-build-isolation openai-whisper==20231117`, then follow the README as written (`-r sandi-all.txt`, spacy en_core_web_sm, `numpy==1.24.3`). Every pinned version is unchanged.
- sandi-dd: see the section below.

## sandi-dd

`transformers==3.0.1` hard-pins `tokenizers==0.8.0rc4`, which only has a `macosx_10_10_x86_64` wheel and no arm64 wheel; building from source needs Rust. The other pinned packages (torch 1.13.1, numpy 1.24.1, pandas 2.0.3, scikit-learn 1.2.0) all have arm64 wheels.
Solution (decided by the authors, option a): **sandi-dd is an x86_64 environment running under Rosetta 2**. All versions are unchanged, and it still uses conda-forge only.

- Install Rosetta 2: `softwareupdate --install-rosetta --agree-to-license`
- `CONDA_SUBDIR=osx-64 conda create --override-channels -c conda-forge -n sandi-dd python=3.8`
- `conda activate sandi-dd && conda config --env --set subdir osx-64` (pins the env to osx-64 so later installs don't fall back to arm64)
- `pip install -r envs/sandi-dd.txt` (all x86_64 wheels, nothing built from source)
- Check: `python -c "import platform, torch, transformers, tokenizers; ..."` → `x86_64 1.13.1 3.0.1 0.8.0.rc4`
- Effect: the DD step (`dd.py`, called through `envs/sandi-dd-env.sh` in `run-gec.sh`) runs on CPU under Rosetta emulation, so it is slower than native arm64. sandi-all (ASR/SLA/GEC/scoring) is still native arm64.

## Known issue: macOS /bin/bash is 3.2

The run scripts use `#!/bin/bash`, and macOS ships GNU bash 3.2.57, which does not support the bash 4 `|&` syntax (pipe stdout+stderr together).
On the first run of check-pipeline.sh, every step that uses `|&` failed with `syntax error near unexpected token '&'`.
Affected places (8 in 5 files): run-asr-decode.sh:129, run-asr-to-task-input.sh:117/130/149, run-gec.sh:122/136, run-sla.sh:122, scoring-tools run-wer-scoring.sh:169.
Solution (decided by the authors): replace each `|&` with the equivalent `2>&1 |`. Every file is backed up as `*.orig`, and all pass a `/bin/bash -n` (3.2) syntax check.
The incomplete output of the failed first run was renamed to `dev-subset.failed-run1/` (not deleted).

## SCTK / tranfilt (WER scoring) binaries

The binaries that ship with it are Linux x86-64 ELF and cannot run on macOS (not even under Rosetta):
- `scoring-tools/wer-tools/sctk-2.4.9/bin/`: sclite, rfilter1, sctkUnit, sc_stats, rover, asclite (plus the matching executables and .o files under src/)
- `scoring-tools/wer-tools/tranfilt-1.14/rfilter1`: **this is the rfilter1 the WER pipeline actually uses**
  (run-wer-scoring.sh → glm_filter.py → hubscr07.sort.filt.pl → tranfilt-1.14/csrfilt.sh, `EXE_DIR=./scoring-tools/wer-tools/tranfilt-1.14`)

Rebuilding sctk-2.4.9:
- Backup: `cp -R sctk-2.4.9 sctk-2.4.9.orig` (`diff -rq` shows it identical)
- Toolchain: Command Line Tools at `/Library/Developer/CommandLineTools`, Apple clang 21.0.0, GNU Make 3.81
- `make clean` ✓, `make config` ✓, `make all` ✗
  - asclite compiled (`src/asclite/core/asclite` is an arm64 Mach-O)
  - stopped at `src/sclite/align.c`: clang 16+ treats implicit function declarations as errors by default
    `align.c:933 getpid`, `align.c:1070/1295/1438 unlink`: `call to undeclared function ... [-Wimplicit-function-declaration]`
- Final approach (decided by the authors, build settings only, no C code changed):
  1. Tried in `src/sclite`: `make all CFLAGS="-Os -include unistd.h"` (original default `-Os` + unistd.h). That fixed getpid/unlink, but
     `sclite.c:579 TEXT_set_lang_prof` (an sctk-internal function with no prototype) still raised an error → this approach failed.
  2. Fallback: `cd src/sclite && make clean && make all CFLAGS="-Os -Wno-error=implicit-function-declaration"` ✓
     Implicit declarations turned back into warnings: getpid, unlink, TEXT_set_lang_prof (defined as `int TEXT_set_lang_prof(char *)` in text.c:225).
     All have an int return and non-variadic arguments, matching an implicit declaration, so they are ABI-safe on arm64.
     CFLAGS was given only inside src/sclite; the other directories (asclite `-g -Os`, rfilter1 `OPTIONS=-DNEED_STRCMP=1`) kept their original defaults.
  3. Top-level `make all` ✓ (rfilter1 compiled, -Wpointer-sign warnings only)
  4. `make install prefix="$PWD"` ✓. The `config.status` in `src/asclite/core` is a stale leftover from the original distribution, with the relative path
     `prefix=./scoring-tools/wer-tools/sctk-2.4.9`, so a plain `make install` fails; it is overridden on the command line here and the makefile is not changed.
- Result: `file` shows all of sclite, rfilter1, sctkUnit, sc_stats, rover and asclite in bin/ as **Mach-O 64-bit executable arm64**
- Self-tests: `src/rfilter1 make check` gives 0 diff lines; `src/sclite make check` reports `ALL TESTS SUCCESSFULLY COMPLETED`

tranfilt-1.14:
- Backup: `cp -R tranfilt-1.14 tranfilt-1.14.orig` (identical)
- **Do not run `make` / `make clean` in tranfilt-1.14**: the makefile's `rfilter1` target depends on `scripts`, which regenerates `csrfilt.sh` and `ut_rfilt.pl`
  from the templates `csrfilt.` and `ut_rfilt.`, overwriting the project's scoring-logic changes (in csrfilt.sh, K Knill 2024-08-06 handling of hyphens in ctm fileIDs
  plus the `#!/bin/bash` shebang, and the relative EXE_DIR in ut_rfilt.pl).
- Ran only the makefile's compile line `gcc -o rfilter1 rfilter1.c` → **failed**:
  `rfilter1.c:29:5: error: second parameter of 'main' (argument array) must be of type 'char **'`
  Cause: `include/stdcenvp.h:27` has `#define char unsigned char`, which makes main's argv `unsigned char **`; clang treats this as a hard error (no -W switch); GCC only warns.
- Final approach (option 1, decided by the authors): `cd tranfilt-1.14 && gcc -ffreestanding -o rfilter1 rfilter1.c` (make not run) ✓, 123 warnings, 0 errors
  - In freestanding mode clang does not check main's signature; the side effect is that it also implies -fno-builtin, which only affects speed, not output.
  - `file`: Mach-O 64-bit executable arm64; sha256 e94fe560…fc11 (original ELF 2e2ee1c9…c5f, kept in tranfilt-1.14.orig/)
  - Compared with tranfilt-1.14.orig, `diff -rq` differs only in the rfilter1 binary; csrfilt.sh and ut_rfilt.pl are unchanged.
- Scoring check (A2): rescored dev-subset.reference/{asr,gec}/*.ctm with run-wer-scoring.sh into scoring-check/.
  After replacing the output path, .sys/.raw/.pra/.dtl/.lur match exactly; .sgml differs only in ref_fname path and creation_date;
  the hyp/ref .filt files rfilter1 outputs are byte-identical (cmp). Sum/Avg: ASR and GEC WER [corpus-derived value omitted; see the paper], same as the reference.

ELF scan (scoring-tools/ and baseline-pipeline/, excluding *.orig): only **`scoring-tools/wer-tools/tranfilt-1.14/rfilter1`** is left (including .o files).

## ffmpeg (needed by Whisper, not listed in envs/)

openai-whisper decodes audio through the `ffmpeg` command-line tool, but neither `envs/sandi-all.txt` nor `envs/sandi-all.yaml` lists it (the original Linux environment presumably used a system ffmpeg).
The first A3 run failed in the ASR step with `FileNotFoundError: 'ffmpeg'`.
- Install: `conda install -n sandi-all --override-channels -c conda-forge 'ffmpeg=4.4'` → ffmpeg 4.4.2 (gpl_hf4fd261_107).
  The dry run showed only new packages (ffmpeg and its C libraries); no existing package was upgraded, downgraded or removed.
- Lossless check: all corpus audio is 16 kHz / mono / 16-bit FLAC (afinfo; all 22 dev-subset files and 17 files sampled from dev-asr),
  so Whisper's `ffmpeg -ac 1 -ar 16000 -f s16le` needs no resampling.
  The PCM decoded by ffmpeg 4.4.2 and by macOS `afconvert` (LEI16@16000) is byte-identical for all 22 dev-subset files → the ffmpeg version cannot affect ASR input.

## 2026-11-11 06:45 — 04_scripts/b3_edit_labels.py (backup: b3_edit_labels.py.orig)
- Cause of the Q_dev labels failure (rc=3, 2026-11-10 22:31): per-type TP/FP/FN counts were equal to the official `errant_spoken`
  (`counts_equal_official` all true), but for one type (U:VERB: [corpus-derived value omitted; see the paper]) the self-check recomputed F0.5 as `1.25*p*r/(...)`
  whereas the official `compute_p_r_f05` computes `1.25*(p*r)/(...)`; the exact value lies on a `%.3f` rounding boundary ([corpus-derived value omitted; see the paper]), so the float operation order flips the
  `%.3f` rounding.
- The one reference edit matched twice (U:DET, `ref_edits_matched_more_than_once=1`) is NOT the cause: the official scorer itself counts
  tp per matching hypothesis edit (it breaks only the inner loop), so our counts already follow it; changing it to "count once" would make
  `counts_equal_official` false. Left unchanged; still reported as a diagnostic.
- Fix: the overall and per-type self-checks now call the official `SE.compute_p_r_f05` (unchanged module). Labels logic and the
  self-check rules (exact equality with result.txt, overall and per type) are unchanged — not relaxed.
- Check: rerun on B1 and W train/dev/eval into a scratch directory: hyp/ref labels, utt_index and b3_verification.json byte-identical
  to the committed files (6/6 sets).

## 2026-11-11 07:30 — 04_scripts/leak_scan.py (backup: leak_scan.py.pre_pdf.orig) and 04_scripts/e_explore.py (backup: e_explore.py.orig)
- leak_scan.py: .pdf files are scanned on their visible text (stdlib: inflated streams, Tj/'/"/TJ strings in BT..ET), not the raw bytes,
  to stop glyph-name false positives (/one /two ...). pdftotext/pypdf absent; nothing installed. Other file types unchanged.
- e_explore.py: E1 LOW-minus-HIGH contrast -> NA (skipped) when a band shows no edit at an operating point; previously TypeError.

## 2026-11-11 07:40 — 04_scripts/leak_scan.py (backup: leak_scan.py.pre_exclude.orig)
- EXCLUDE_FILES (exact basename): a public guide-for-authors PDF (decided by the authors; no corpus content). Scan: 0 files with hits.

## 2026-11-11 11:30 — 04_scripts/leak_scan.py (backup: leak_scan.py.pre_conll.orig)
- EXCLUDE_DIRS: 05_runs/2026-11-10_G_greco/repro_20261111/ (public CoNLL-2014/NUCLE GRECO reproduction files; decided by the authors).
