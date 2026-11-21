# Which Edits Should Learners See? Selective Feedback in Spoken Grammatical Error Correction

Code for the manuscript "Which Edits Should Learners See? Selective Feedback in Spoken Grammatical Error Correction"
(submitted to *Expert Systems with Applications*).

**What this repository contains**
- Code: `scripts/` (including the post hoc Appendix G scripts in `scripts/posthoc_appendix_g/`)
- Configurations: `configs/`
- Environment descriptions and portability patches: `environment/`
- Documentation: `docs/` (`DATA_LICENSE_NOTE.md`, `THIRD_PARTY_CODE.md`) and this README

**What it does not contain, and why**
- The Speak & Improve Corpus 2025 and anything derived from it at the level of utterances or edits (audio, transcripts, labels,
  features, model outputs, trained models). These are governed by the Speak & Improve Corpus 2025 licence; the corpus is
  available from Cambridge University Press & Assessment (CUP&A) under that licence.
- Aggregate results (`results/`) and the documents that quote them (`docs/AUDIT_REPORT.md`, `docs/F2_analysis_plan.md`,
  `docs/F_human_validation_plan.md`). They are corpus-derived and will be added once CUP&A approves their release.
- GRECO code and weights. Obtain them from the GRECO authors (https://github.com/nusnlp/greco, GPL-3.0); the `g_greco_*.py`
  scripts import GRECO at run time.

**Licences**: the code is released under the MIT licence (`LICENSE`; the few third-party lines listed in
`docs/THIRD_PARTY_CODE.md` are excepted). Documentation written by the authors is released under CC BY 4.0.

The original package README follows. Where it refers to `results/` or to the documents listed above, those files are not
part of this repository.

---

# Risk-aware selective feedback for spoken GEC — code and configurations

Supplementary repository for the manuscript "Which Edits Should Learners See? Selective Feedback in Spoken Grammatical Error
Correction" (under double-anonymized review; author information removed).

**Version 5.** Version 2 was checked in a pre-submission reproduction audit, run in a separate workspace with freshly created environments and following only the package documentation; the audit was carried out with an AI coding assistant (see the declaration of generative AI use in the manuscript). In the reproduction audit (its report, `docs/AUDIT_REPORT.md`, is not included in this repository (corpus licence; see the note at the top)) B1, the W eval and the
Q dev-subset were reproduced; 19 documentation defects were recorded, addressed in version 3 (appendix of that report); version 4 follows a
documentation check of version 3 (no output location is defaulted any more, Hugging Face cache configurable, generation steps of
every result file, B2/B2b, totals, downloads, licences); version 5 adds the corpus-licence note (`docs/DATA_LICENSE_NOTE.md`:
corpus-derived files, CUP&A approval), zero-context baseline patches, corrected corpus citations, neutral wording and an English
summary of the reproduction audit report, licence layers (code MIT; authors' documentation CC BY 4.0; results and corpus-derived statistics
without open licence, subject to CUP&A approval) and the list of third-party code lines. No result value was changed in any version; `results/` only gained files. `results/` is not included in this repository (corpus licence; see the note at the top).
The package checksum file `MANIFEST.sha256` is not included in this repository (it lists the withheld files).

## Overview: what it takes (Apple M5 Pro, 24 GB, macOS 26.5; measured wall-clock)
| step | command | time | needs |
|---|---|---|---|
| Setup: environments, official code, models | Setup 1–4 | ~16 min from local caches; downloads ≈ 35 GB extra (bandwidth) | conda, Xcode command-line tools |
| Check: official dev-subset run | `STEP=devsubset run_b1.sh` | < 1 min | |
| 1. B1 train+dev -> labels -> features -> CV -> frozen models | `run_b1.sh` | 2 h 32 min (pipelines 136 min, rest 16 min) | CPU |
| 2. B1 held-out eval (once) | `c7_driver.sh` | 1 h 50 min | CPU |
| 3. W chain dev/train/eval + models | `d_resume.sh W` | ≈ 7.5 h (eval only: ≈ 2.9 h) | Apple Silicon (mlx) for ASR |
| 4. Q chain dev/train/eval + models | `d_resume.sh Q` | ≈ 4 h | Apple Silicon (mlx); ≥ 16 GB RAM advisable for Qwen 7B 4-bit |
| 5. B2 / B2b diagnostic conditions | step 5 | ≈ 6 h (B2 56 min, B2b v1 2 h 26 min, B2b v2 2 h 40 min) | CPU |
| 6. Stage B/C side results | "Other result files" | ≈ 2 min | CPU |
| 7. Stage E exploratory | `e_run.sh`, `e_explore.py` | < 1 min | CPU |
| 8. GRECO comparison | `e_after_q.sh greco` | ≈ 1 h 40 min | CPU |
| 9. Summaries, leak scans | `pkg_results_summary.py`, `leak_scan.py` | < 2 min | |
| **Total** | | **≈ 24 h** of computing (steps 1–9; ≈ 18 h without B2/B2b) | |

Disk: corpus 17 GB (audio) + official models 6.6 GB; pretrained models 9.8 GB in the HF cache (Whisper-large-v3 3.1 GB, Qwen2.5-7B-4bit
4.3 GB, bert-base-uncased 0.4 GB, xlm-roberta-base 1.1 GB, deberta-v3-large 0.9 GB) + GRECO clone with checkpoint 1.8 GB; conda
environments ≈ 5.4 GB; outputs: run root ≈ 1.1 GB, `EXP_DIR` ≈ 0.4 GB. **Plan for ≈ 45 GB free.**
Platform: `caffeinate` / `pmset` (queue keep-awake) and mlx (W ASR, Qwen) are macOS / Apple Silicon only; the B1/B2 pipelines,
modelling and evaluation are plain CPU code (they should also run on Linux, not tested).

## What is included
| Folder | Contents |
|---|---|
| `configs/` | Pre-registered analysis plans and supplements (dated, written before the corresponding results; later amendments are marked inside the files), run configurations, and `final.yaml` (frozen configuration for the single held-out evaluation, SHA-256 of model artifacts and scripts, post-freeze note). These are historical records and are kept unchanged; see "Research directory names" below. `post_freeze_changes.md`: every change to a frozen script after the freeze, with diffs. |
| `scripts/` | All code written for this study (index below). `paths.sh` holds the path configuration. |
| `results/` | Aggregate results (JSON / CSV / Markdown): not included in this repository (corpus licence; see the note at the top). The index below still describes how each file is produced. |
| `environment/` | Conda exports, pip freezes, architecture info, system information, model file hashes (`model_hashes.sha256`), portability patches to the official baseline code (`mac_changes.md`, `patches/` with `patches/README.md`). |
| `docs/` | `AUDIT_REPORT.md` (the reproduction audit of version 2): not included in this repository (corpus licence; see the note at the top). `DATA_LICENSE_NOTE.md`: corpus licence terms, every corpus-derived file with its level of information, CUP&A approval status, excerpt scan. `THIRD_PARTY_CODE.md`: lines copied from the official baseline into the study scripts (location, source, licence). `F2_analysis_plan.md`, `F_human_validation_plan.md` (plans of the post hoc pre-submission analyses): not included in this repository (corpus licence; see the note at the top). |

### Research directory names
The configs, `mac_changes.md` and script comments were written in the research working tree and refer to its directory names.
They are kept as written (historical records); read them with this table:

| research tree | package / your machine |
|---|---|
| `03_configs/` | `configs/` |
| `04_scripts/` | `scripts/` (`$SCRIPTS_DIR`) |
| `06_results/` | published: `results/` (not included in this repository (corpus licence; see the note at the top)); your runs: `$EXP_DIR/06_results/` |
| `05_runs/`, `07_figures/` | `$EXP_DIR/05_runs/`, `$EXP_DIR/07_figures/` (created by the scripts; not distributed) |
| `01_environment/` | `environment/` |
| `08_paper/`, `09_lab_notebook.md`, `09_manuscript_zh/`, git commits (e.g. "commit e14da3c") | not distributed (manuscript notes, lab notebook, version control of the research tree) |
| the corpus directory, `stageB/`, `stageD/` | `$CORPUS_DIR` (inputs) and `$RUN_DIR/stageB`, `$RUN_DIR/stageD` (outputs) |

## What is NOT included, and why
- **Speak & Improve Corpus 2025** audio and manual annotations (Cambridge University Press & Assessment licence; not redistributable).
- **Per-edit labels, features and model outputs** (derived from the licensed annotations; they carry file IDs). Regenerate them.
- **Official S&I 2025 baseline code and models** (distributed with the corpus).
- **GRECO code and weights** (GPL-3.0) and **any CoNLL-2014 / NUCLE data** (own licence).
- **Pretrained models** (Hugging Face; see Models).
- `.orig` backups, `__pycache__`, run directories and logs.

## Setup
### 1. Paths (no path is hard-coded and no output location has a default: unset variables stop the scripts)
| variable | meaning | default |
|---|---|---|
| `EXP_DIR` | your work directory for research outputs: `05_runs/` (per-run derived data), `06_results/` (aggregate results), `07_figures/`; must exist; must not be the package | — **required** |
| `CORPUS_DIR` | root of the Speak & Improve Corpus 2025 (read-only inputs) | — **required** |
| `RUN_DIR` | run root with the official layout, built with `make_run_root.sh`; pipeline outputs and logs go to `$RUN_DIR/stageB`, `stageD`, ... (a warning is printed if it is the corpus directory itself) | — **required** |
| `SCRIPTS_DIR` | this package's `scripts/` directory (set by `paths.sh` inside the scripts; export it yourself for the commands in this README) | — |
| `CONDA_BASE` | conda installation | `$HOME/miniconda3` |
| `ENV_SANDI_ALL`, `ENV_SANDI_DD`, `ENV_ESWA_C`, `ENV_ESWA_D`, `ENV_ESWA_G` | conda environment names | `sandi-all`, `sandi-dd`, `eswa-c`, `eswa-d`, `eswa-g` |
| `HF_HUB_CACHE` (or `HF_HOME`) | Hugging Face hub cache holding the models | `${HF_HOME:-$HOME/.cache/huggingface}/hub` |
| `GRECO_DIR` | GRECO clone (GRECO comparison only) | `$HOME/greco_code` |
| `GRECO_SYSTEMS` | systems scored by `e_after_q.sh greco` | `B1 W Q` |
| `D_SETS`, `D_SKIP_RETRAIN` | W/Q queues: only some sets (e.g. `eval`); skip retraining | all sets; retrain |
| `GIT_COMMIT` | `1` lets `git_commit_locked.sh` snapshot-commit `$EXP_DIR` (only if it is itself a git work tree root) | unset: never commits |
| `LEAK_SCAN_EXCLUDE_FILES` | exact basenames (one per line) of public, non-corpus documents that `leak_scan.py` skips | unset: nothing skipped |
| `PATHS_NO_CORPUS`, `PATHS_NO_RUN` | internal: set by the few scripts that need no corpus / no run root | — |

```bash
export SCRIPTS_DIR=/path/to/package/scripts CORPUS_DIR=/path/to/sandi-corpus-2025 EXP_DIR=/path/to/work
mkdir -p $EXP_DIR
# official code (zips from the corpus distribution): unpack baseline-pipeline.zip and scoring-tools.zip into /path/to/official,
# copy envs/ and check-pipeline.sh from the corpus, apply environment/patches (group A of patches/README.md), rebuild SCTK and
# tranfilt as in environment/mac_changes.md
bash $SCRIPTS_DIR/make_run_root.sh /path/to/run_root /path/to/official   # symlinks: corpus inputs + patched official code
export RUN_DIR=/path/to/run_root HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```
Keep the published `results/` (not included in this repository (corpus licence; see the note at the top)) apart from `$EXP_DIR/06_results/`, where the scripts write their own files (`c7_driver.sh` stops if `$EXP_DIR/06_results/eval_results.json` is already present); compare the two folders afterwards.

### 2. Corpus (not included; cite as below)
- **Data set**: Cambridge University Press & Assessment (2025). *Speak & Improve Corpus 2025* [Data set]. Access:
  https://researchdatasets.cambridge.org/datasets/speak-and-improve-corpus-2025 (registration required; non-commercial use only,
  under the corpus licence agreement). The licensed distribution contains the download instructions, the official baseline
  (`baseline-pipeline.zip`, `scoring-tools.zip`, `envs/`, `check-pipeline.sh`) and the official models (DD, GEC, SLA).
  Derived items (e.g. models, data statistics) may be released only with prior approval of CUP&A: see `docs/DATA_LICENSE_NOTE.md`.
- **Corpus paper**: Knill, K., Nicholls, D., Gales, M. J. F., Qian, M., & Stroinski, P. (2025). *Speak & Improve Corpus 2025: an L2
  English Speech Corpus for Language Assessment and Feedback*. Cambridge University Press & Assessment.
  https://doi.org/10.17863/CAM.114333 (preprint: arXiv:2412.11986).
  SLaTE 2025 version: Knill, K. M., Nicholls, D., Gales, M. J. F., Qian, M., & Stroinski, P. (2025). Introducing the Speak & Improve
  Corpus 2025: an L2 English Speech Corpus for Language Assessment and Feedback. *Proc. 10th Workshop on Speech and Language
  Technology in Education (SLaTE 2025)*, 167–171. https://doi.org/10.21437/SLaTE.2025-34
- **Challenge and baseline paper**: Qian, M., Knill, K. M., Bannò, S., Tang, S., Karanasou, P., Gales, M. J. F., & Nicholls, D.
  (2025). Speak & Improve Challenge 2025. *Proc. SLaTE 2025*, 41–45. https://doi.org/10.21437/SLaTE.2025-9 (preprint, tasks and
  baseline systems: arXiv:2412.11985).

### 3. Environments
The environment names `eswa-c`, `eswa-d`, `eswa-g` (and the variables `ENV_ESWA_*`) are internal labels; they are not renamed
because frozen scripts and the archived environment exports refer to them (set `ENV_ESWA_*` to use other names).

| env | used for | how to create (all from conda-forge: `--override-channels -c conda-forge`) |
|---|---|---|
| `sandi-all` | official ASR, GEC, scoring; labels, BART scores, features | `conda env create -n sandi-all -f environment/conda_env_sandi-all.yaml` (osx-arm64, Python 3.10; always pass `-n`: the files' `prefix:` line is not portable), then as in `mac_changes.md`: `pip install "setuptools<81" wheel`, `pip install --no-build-isolation openai-whisper==20231117`, `pip install numpy==1.24.3` (whisper pulls a newer numpy), the remaining pins of `pip_freeze_sandi-all.txt`, spaCy `en_core_web_sm-3.8.0` (wheel; sha256 in the pip freeze), and `conda install ffmpeg=4.4` (missing from the export) |
| `sandi-dd` | official DD (BERT, transformers 3.0.1) | `CONDA_SUBDIR=osx-64 conda env create -n sandi-dd -f environment/conda_env_sandi-dd.yaml` (x86_64 under Rosetta 2, Python 3.8) + `pip_freeze_sandi-dd.txt` |
| `eswa-c` | modelling (LightGBM/LR), evaluation, stage E, side results | `conda env create -n eswa-c -f environment/conda_env_eswa-c.yaml` (Python 3.11, scikit-learn 1.5.2, lightgbm 4.5.0) |
| `eswa-d` | W chain ASR (mlx-whisper 0.4.2), Q chain (mlx-lm) | `conda_env_eswa-d.yaml` + `pip_freeze_eswa-d.txt` (Python 3.11) |
| `eswa-g` | GRECO scoring only | Python 3.10; torch 1.12.1, transformers 4.23.1, tokenizers 0.13.1, sentencepiece 0.1.97, numpy 1.23.3, pandas 1.5.0, protobuf 3.20.3, huggingface_hub 0.10.1 |

Helper scripts run with the system `python3` (standard library only; tested with 3.9.6).

### 4. Models (pinned revisions; `huggingface_hub` CLI; files go to `$HF_HUB_CACHE`)
```bash
huggingface-cli download mlx-community/whisper-large-v3-mlx --revision 49e6aa286ad60c14352c404340ded53710378a11          # 3.1 GB, W ASR
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit --revision c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed      # 4.3 GB, Q GEC + scores
huggingface-cli download bert-base-uncased --revision 86b5e0934494bd15c9632b12f734a8a67f723594                          # 0.4 GB, official DD tokenizer
huggingface-cli download xlm-roberta-base --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089                           # 1.1 GB, official SLA tokenizer
huggingface-cli download microsoft/deberta-v3-large --revision 64a8c8eab3e352a784c658aef62be1662607476f \
    config.json pytorch_model.bin spm.model tokenizer_config.json                                                        # 0.9 GB, GRECO only
echo 64a8c8eab3e352a784c658aef62be1662607476f > "$HF_HUB_CACHE/models--microsoft--deberta-v3-large/refs/main"            # offline loading by name
cd "$HF_HUB_CACHE" && shasum -a 256 -c /path/to/package/environment/model_hashes.sha256                                  # 29 files, all must be OK
```
Main weight files (sha256): Whisper `weights.npz` `05ff791c…`, Qwen `model.safetensors` `86110f36…`, DeBERTa `pytorch_model.bin`
`dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5`; all files in `environment/model_hashes.sha256`.
transformers 3.0.1 (`sandi-dd`) also reads the BERT vocabulary from its legacy cache `~/.cache/torch/transformers` (it downloads it
there on first use). After downloading, run offline (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`). A missing model stops the
script with "model not found" (W ASR, Qwen, GRECO); nothing is skipped silently.

Check the setup with the official dev-subset run (expected GEC WER and SGECF P/R/F0.5: [corpus-derived value omitted; see the paper]; compare with the official `dev-subset.reference/` outputs; also needed by the
Q rehearsal): `cd "$RUN_DIR" && STEP=devsubset bash $SCRIPTS_DIR/run_b1.sh` (< 1 min).

## Reproducing (in order)
1. **B1, train and dev** (main system): `cd "$RUN_DIR" && nohup caffeinate -ims bash $SCRIPTS_DIR/run_b1.sh > run_b1.log 2>&1 & disown`
   Steps (each resumable): B1 dev and train pipelines (official ASR -> DD -> GEC -> scoring; in parallel) -> `b_collect.sh`
   -> `b_norm_verbatim.sh` v3 and v4 (`--drop_markup`) -> `b3_edit_labels.py` (labels v3 and v4; v4 is primary) -> `b4_label_stats.py`
   (B1 only; the published label-stat files also contain B2/B2b, step 5) -> `c1_bart_score.py` -> `c1_features.py` (labels v4)
   -> `c_model.py cv` and `evaluate` (5-fold grouped CV with inner model selection; OOF metrics; `results/C2_cv_*.json`, not included here) ->
   `c6_final.py` (frozen models) -> `c6b_addendum.py` (risk_lr, operating points).
   Check: sha256 of `$EXP_DIR/05_runs/2026-11-10_C6_final/{risk_lgbm.txt,gec_lr.pkl,asr_lr.pkl,type_prior.json,risk_lr.pkl}` =
   `configs/final.yaml` (`artifact_sha256`, `addendum_before_eval.risk_lr.sha256`).
2. **B1 held-out eval, once**: `cd "$RUN_DIR" && nohup caffeinate -ims bash $SCRIPTS_DIR/c7_driver.sh > c7_driver.log 2>&1 & disown`
   -> `$EXP_DIR/06_results/eval_results.json` (refuses to run twice).
3. **W chain** (Whisper-large-v3 ASR -> official DD/GEC): `bash $SCRIPTS_DIR/d_resume.sh W`. Needs the B1 outputs of the same set
   (`stageB/B1_<set>`, `stageB/norm_verbatim_v4_<set>`; for eval run step 2 first; checked at the start of each set). Eval only:
   `D_SETS=eval D_SKIP_RETRAIN=1 bash $SCRIPTS_DIR/d_resume.sh W` -> `D_W_eval_frozen.json`.
   Measured: ASR 60/36/55 min (dev/train/eval), DD 48/31/47 min, GEC 60/36/63 min, labels/BART/features a few minutes; models 1 min.
4. **Q chain** (B1 DD output -> Qwen2.5-7B-Instruct 3-shot GEC): `bash $SCRIPTS_DIR/d_resume.sh Q` (waits for a running W queue);
   dev-subset rehearsal: `bash $SCRIPTS_DIR/d_resume.sh Q_rehearsal` (~1.5 min; needs the Setup check run).
   Measured: generation 44/31/44 min, Qwen teacher-forced scores 22/13/19 min, BART scores 5/3/5 min, align+score+labels ~3 min per set;
   M_Q (retrain + eval once, primary and BART-feature variants) 41 s + 39 s.
5. **B2 / B2b diagnostic conditions** (manual transcripts instead of ASR; they explain where B1 loses accuracy, they are not
   systems of the paper): B2 = manual *fluent* transcript -> GEC; B2b = manual *verbatim* transcript -> official DD -> GEC
   (v1: with transcription markup; v2, used in `_v4` files: markup removed). Needs the norm_verbatim outputs of step 1.
   ```bash
   cd "$RUN_DIR"; source $CONDA_BASE/etc/profile.d/conda.sh
   for ds in dev train; do   # B2 (34 / 22 min)
     bash $SCRIPTS_DIR/b_pipeline2.sh fluent $ds reference-materials/flists.flac/$ds-gec.tsv stageB/B2_$ds > stageB/B2_$ds.log 2>&1
     bash $SCRIPTS_DIR/b_collect.sh fluent 2026-11-10_B2_$ds stageB/B2_$ds stageB/B2_$ds.log
     ( conda activate $ENV_SANDI_ALL && python $SCRIPTS_DIR/b3_edit_labels.py --run_dir stageB/B2_$ds --ds $ds --mode fluent \
         --verbatim_norm_ctm stageB/norm_verbatim_$ds/gec/norm/gec.ctm --out_dir $EXP_DIR/05_runs/2026-11-10_B2_$ds/labels )
   done
   for ds in dev train; do   # B2b v1: the official GEC step stops on phrases that DD empties (exit 100, expected) ...
     bash $SCRIPTS_DIR/b_pipeline2.sh verbatim $ds reference-materials/flists.flac/$ds-gec.tsv stageB/B2b_$ds > stageB/B2b_$ds.log 2>&1
   done
   bash $SCRIPTS_DIR/b2b_driver.sh     # ... b2b_regec.sh reuses that DD output, filters empty phrases, reruns GEC + scoring, labels (67 min)
   bash $SCRIPTS_DIR/b2b_v2_driver.sh  # B2b v2 (markup-free verbatim input; 97 / 63 min), labels with the v4 reference
   for ds in dev train; do           # label statistics with the three conditions (published files; verified byte-identical)
     python3 -I $SCRIPTS_DIR/b4_label_stats.py --ds $ds --cond B1=$EXP_DIR/05_runs/2026-11-10_B1_$ds/labels \
       --cond B2b=$EXP_DIR/05_runs/2026-11-10_B2b_$ds/labels --cond B2=$EXP_DIR/05_runs/2026-11-10_B2_$ds/labels --out $EXP_DIR/06_results/label_stats_$ds.csv
     python3 -I $SCRIPTS_DIR/b4_label_stats.py --ds $ds --cond B1=$EXP_DIR/05_runs/2026-11-10_B1_$ds/labels_v4 \
       --cond B2b=$EXP_DIR/05_runs/2026-11-10_B2b_${ds}_v2/labels --cond B2=$EXP_DIR/05_runs/2026-11-10_B2_$ds/labels --out $EXP_DIR/06_results/label_stats_v4_$ds.csv
   done
   python3 -I $SCRIPTS_DIR/b_checkpoint_table.py; python3 -I $SCRIPTS_DIR/b_checkpoint_table.py --v4
   ```
   (`b_queue.sh` is the overnight queue that originally ran B1 train, B2 and B2b v1 in one go; it waits for the PID of a running
   v1 B1 dev pipeline and is kept for the record.)
6. **Other result files** (see "Results: index and origin").
7. **Stage E (exploratory, added after the eval)**: `bash $SCRIPTS_DIR/e_run.sh` (B1, W, W_frozen) and
   `$CONDA_BASE/envs/$ENV_ESWA_C/bin/python -I $SCRIPTS_DIR/e_explore.py --system Q` (each < 20 s; figures in `$EXP_DIR/07_figures/`).
8. **GRECO comparison (added after the C7 eval; Q added after the eval)** — see GRECO; `bash $SCRIPTS_DIR/e_after_q.sh greco`.
9. **Summaries and leak scans**: `python3 -I $SCRIPTS_DIR/pkg_results_summary.py` (numbers only, from the run outputs);
   `python3 -I $SCRIPTS_DIR/leak_scan.py` scans `$EXP_DIR` for 6-grams of the manual references in `$CORPUS_DIR/reference-materials/stms`
   (excluding `05_runs/*/outputs/` and the basenames listed, one per line, in `LEAK_SCAN_EXCLUDE_FILES`); `python3 -I $SCRIPTS_DIR/pkg_excerpt_scan.py <package dir>`
   reports runs of >= 5 consecutive words shared with the manual references (location and length only; >= 100 words = licence violation);
   `python3 -I $SCRIPTS_DIR/pkg_leak_scan.py <package dir>` scans a package against all references,
   no exclusions. Exit code 4 = hits. The public test fixtures inside SCTK (`scoring-tools/wer-tools/sctk-2.4.9/...`) are known false
   positives when a scan root contains the official scoring tools. (`pkg_build.py` builds this package from the research tree; authors only.)

The dev-subset (`reference-materials/flists.flac/dev-subset-gec.tsv`) has **[corpus-derived value omitted; see the paper] audio files**, split by the official scorer into [corpus-derived value omitted; see the paper]
scoring segments and by the GEC step into [corpus-derived value omitted; see the paper] phrases (the file count noted in `results/official_baseline_comparison.json`, not included here,
is wrong; the file is left unchanged).

## Results: index and origin (`results/`, not included in this repository (corpus licence; see the note at the top))
| file | contents | produced by |
|---|---|---|
| `reproduction.json` | stage A3: official baseline check on the dev-subset (own CPU ASR and the official ASR output) | **Compiled by hand** (no script): from the outputs of the official `check-pipeline.sh` (own ASR) and `scripts/a3_downstream_check.sh` (official Whisper CTM), compared with the official reference outputs shipped with the corpus (`dev-subset.reference/`): sclite Sum/Avg rows, SGECF `result.txt` (also rescored with `run-sgecf-scoring.sh --edit_detail`), SLA scores; word differences and byte identity of the intermediate files checked with `diff`/`cmp`. Re-check: run both, compare with `dev-subset.reference/`. |
| `official_baseline_comparison.json` | B1 dev vs the official baseline (overall, sub/del/ins, per type); official eval values | **Compiled by hand** (no script): official values transcribed from the challenge paper (`speak-and-improve-challenge-2025.pdf`, Tables 5–6, Dev) and the SLaTE 2025 overview (Tables 4–5, taken to be Eval), both in the corpus `reference-materials/docs/`; ours from `$EXP_DIR/05_runs/2026-11-10_B1_dev/metrics.json` (step 1). Also confirmed by the reproduction audit. |
| `b_checkpoint_summary{,_v4}.{csv,md}` | stage B: B1, B2b, B2 per dataset (metrics + label stats) | `python3 -I scripts/b_checkpoint_table.py [--v4]` after step 5 (`--v4`: labels v4, B2b v2) |
| `label_stats{,_v4}_{dev,train}.csv` | edit-label statistics, conditions B1, B2b, B2 (v4 = primary label definition; files without `_v4` are superseded versions kept for transparency) | `b4_label_stats.py`, commands in step 5 |
| `oracle_upper_bounds{,_v4}.json` | B1 SGECF if false feedback were withheld perfectly | `python3 -I scripts/c5_oracle.py > $EXP_DIR/06_results/oracle_upper_bounds.json` and `python3 -I scripts/c5_oracle.py labels_v4 > …_v4.json` (after step 1) |
| `C2_cv_evaluation.json`, `C2_cv_cv_info.json`, `C2_cv_coverage_curves.json` | stage C2 cross-validation (OOF) with bootstrap CIs, folds, chosen hyper-parameters, coverage curves | step 1 (`c_model.py cv/evaluate`); copied by `pkg_results_summary.py` |
| `eval_results.json` | B1 held-out eval (C7): points, 95% CIs, pre-specified paired tests, operating points, calibration, per FP cause | step 2 |
| `eval_posthoc_equal_precision.json` | post-hoc (unplanned, after the eval): coverage at equal precision | `$CONDA_BASE/envs/$ENV_ESWA_C/bin/python -I scripts/c7b_posthoc_equal_precision.py` (after step 2; < 10 s) |
| `fig4_precision_at_coverage_eval.csv` | data of Figure 4 (from `eval_results.json`) | `pkg_results_summary.py` |
| `asr_conf_diagnostics.json` | ASR-confidence diagnostics (manuscript Section 5.2.3; no retraining) | `$CONDA_BASE/envs/$ENV_ESWA_C/bin/python -I scripts/asr_conf_diagnostics.py` (after step 1) |
| `D_W_eval_frozen.json`, `D_W_eval_retrained.json`, `D_Q_eval_retrained{,_bartfeat}.json` | stage D: W and Q systems | steps 3, 4 |
| `system_metrics_summary.json`, `qwen_score_summary.json` | ASR/GEC WER, SGECF, edits for B1/W/Q; Qwen score means and format errors | `pkg_results_summary.py` |
| `E_*_exploratory.json` | stage E exploratory analyses | step 7 |
| `GRECO_{B1,W,Q}_eval.json` | GRECO comparison (includes the B1 OOF point estimates of risk_lgbm) | step 8 (written to `$EXP_DIR/06_results/`) |
| `F2_ablation_trdev.json`, `F2_lopo_trdev.json`, `F2_ablation_eval_posthoc.json` | **post hoc** (after the eval): feature-group ablation, leave-one-task-out validation, eval supplement (manuscript Section 5.3.5, Table 12, Appendix C) | `f2_ablation.py` (see the scripts index) |
| `F_human_eval.json` | **post hoc**: human review of the false-feedback labels, aggregate statistics only (manuscript Section 5.4, Table 13) | `f_human_eval.py --inter --inter2` (needs the human labels, which are not distributed) |

Verification of these commands (package version 4): `b_checkpoint_table.py` (both), `c5_oracle.py` (both), `b4_label_stats.py`
(4 files) and `c7b_posthoc_equal_precision.py` re-run on the original run outputs reproduce the published files byte for byte;
`asr_conf_diagnostics.py` reproduces all 32 numbers (only its embedded script hash differs, because the script's path line changed).

## Scripts index (`scripts/`)
| script | role |
|---|---|
| `paths.sh`, `make_run_root.sh` | path configuration; run-root builder |
| `run_b1.sh` | step 1 driver (calls `b_pipeline2.sh`, `b_collect.sh`, `b_metrics.py`, `b_norm_verbatim.sh`, `stm_to_ctm.py`, `b3_edit_labels.py`, `b4_label_stats.py`, `c1_bart_score.py`, `c1_features.py`, `c_model.py`, `c6_final.py`, `c6b_addendum.py`) |
| `c7_driver.sh`, `c7_eval.py` | step 2 |
| `c7b_posthoc_equal_precision.py`, `c5_oracle.py`, `asr_conf_diagnostics.py`, `b_checkpoint_table.py`, `pkg_results_summary.py` | side results (results index) |
| `c5_analysis.py`, `c5_ablation_boot.py` (with `c_model.py ablate`) | C5 planned side analyses (per test part, label-definition sensitivity, feature-group ablation); outputs in `$EXP_DIR/05_runs/2026-11-10_C5_ablation/`, not in `results/` |
| `c5_b2b_residual_disfl.py`, `c5_disfl_run.sh` | B2b residual-disfluency counts (outputs in `05_runs/2026-11-10_C5_extra`) |
| `d_resume.sh`, `d_queue_w.sh`, `d_queue_q.sh`, `d_check.sh` | steps 3–4 (queues and a status check); `d_queue.sh` = earlier combined rehearsal queue, kept for the record |
| `d_asr.py`, `d_ctm_check.py`, `d_asr_stats.py`, `b2b_filter_empty.py`, `d_llm_gec.py`, `d_qwen_prompt.py`, `d_llm_align.py`, `d_qwen_score.py`, `d_model.py` | called by the queues (`d_qwen_prompt_v2.py`: earlier prompt, kept for the record) |
| `b_pipeline2.sh`, `identity_dd.py`, `b2b_regec.sh`, `b2b_v2.sh`, `b2b_driver.sh`, `b2b_v2_driver.sh`, `b_queue.sh` | B1/B2/B2b pipelines and drivers (step 5; `identity_dd.py` = the DD step of B2, whose fluent transcripts need no disfluency removal); `b_pipeline.sh` = v1 pipeline, superseded by `b_pipeline2.sh` |
| `a3_downstream_check.sh` | stage A check: official-ASR downstream run (`reproduction.json`) |
| `e_run.sh`, `e_explore.py`, `e_after_q.sh` | stage E and the GRECO runner |
| `g_greco_score.py`, `g_greco_model.py`, `g_greco_validate.py`, `g_greco_conll_eval.py` | GRECO comparison |
| `leak_scan.py`, `pkg_leak_scan.py`, `pkg_excerpt_scan.py`, `pkg_build.py`, `git_commit_locked.sh` | leak scans, package builder (authors), optional snapshot commits |

### Pre-submission supplementary analyses (post hoc)
These analyses were added **after the single held-out (test set) evaluation**, before submission; they are post hoc and not part
of the pre-registered plan. Their plans were written before the corresponding results: `docs/F2_analysis_plan.md` (ablation and
leave-one-task-out) and `docs/F_human_validation_plan.md` (human review, with two dated addenda); both are not included in this repository (corpus licence; see the note at the top).

| script | role |
|---|---|
| `f2_ablation.py` | Feature-group ablation and leave-one-task-out (leave-one-part-out) validation (manuscript Section 5.3.5, Table 12, Appendix C). Input: the train+dev features and labels of B1 (`$EXP_DIR/05_runs/2026-11-10_C1_features/features_{train,dev}.csv`, `utt_counts_*.csv`; the target `false_feedback` is a column of the features), the C2 folds (`05_runs/2026-11-10_C2_cv/`), the C5 outputs (reference only) and, for the post hoc eval supplement, the B1 eval features and `eval_scores_frozen.csv` (written by `c7b_posthoc_equal_precision.py`). LightGBM with the final fixed parameters (no inner tuning). Output: `$EXP_DIR/06_results/F2_ablation_trdev.json`, `F2_lopo_trdev.json`, `F2_ablation_eval_posthoc.json`; per-edit scores in `05_runs/2026-11-12_F2/`. Run after steps 1–2 and the C5 side analyses: `$CONDA_BASE/envs/$ENV_ESWA_C/bin/python -I -B scripts/f2_ablation.py` (about 2 min); built-in checks: the refitted full model and GEC-only LR reproduce the frozen eval scores. |
| `f_make_sheet.py`, `f_make_sheet_r2.py` | Build the blind annotation sheets of the human review (manuscript Section 5.4): a stratified sample of 200 eval edits (2 × 2 strata, seed 20261112), a 50-item sheet for round 1 of the second annotator and 50 new items for round 2 (seed 20261113). The sheets contain corpus text, so the sheets, the answer keys and the human labels are **not distributed** with the code repository for the manuscript. Kept as run, with the research-time paths (they do not use `paths.sh`). |
| `f_human_eval.py` | Computes the statistics of Table 13 from the human labels (weighted by stratum size; stratified bootstrap; inter-annotator agreement and Cohen's κ with `--inter --inter2`). Because the human labels are not distributed, this script **cannot be run** from the code repository for the manuscript; the aggregate results are given in Table 13 of the manuscript (and in `results/F_human_eval.json` where `results/` is distributed). |

### Appendix G analyses (post hoc, added during revision): `scripts/posthoc_appendix_g/`
These scripts were written and run after the single held-out (test set) evaluation, during the revision (2026-11-19); they are
post hoc. They are copied as run: the path constants at the top of each script point to the research machine's layout (`~/...`;
`common.py` also imports `c_model.py` from the research tree), so edit them before running. They read per-edit features, labels
and frozen-model scores that are not distributed (corpus licence); their outputs (per-edit scores and aggregate JSON) are not
included either.
In `h_editconf.py` and `i_lu_proxy.py`, the variable `S1` originally pointed to an old folder on the authors' machine; to
reproduce, set it to the `scripts/` directory of this repository.

| script | role | manuscript |
|---|---|---|
| `common.py` | shared paths, metrics and the response-grouped bootstrap (same seed and procedure as `c_model.py`) | used by the scripts below |
| `asr_spline.py` | nonlinear ASR baseline: cubic B-spline logistic regression on `asr_conf_win_min`, selected on development data, refitted on train+dev and frozen | Tables G.1 and G.6 |
| `dd_convert.py`, `native_dd.py`, `dd_export.py`, `dd_export_native.py`, `dd_reconcile.py`, `dd_validate_ctm.py` | DD class posteriors computed with the archived DD checkpoint (no upstream output changed), with consistency checks against the archived fluent CTM | inputs of Tables G.2–G.3 |
| `lu_inputs.py` | aligns the DD posteriors with the B1 edits (inputs of the Lu-style rules) | inputs of Tables G.2–G.3 |
| `lu_weighted.py` | Lu-style weighted rules: development-selected weights and thresholds, plus the published weights | Tables G.2, G.3 and G.6 |
| `lu_weighted_primary_v1.py` | first Lu-style run, before the published-weights variant was added; kept for the record | see script header |
| `cross_qian.py` | Qian-style edit_avg / edit_min rules on the large-v3 and Qwen candidates; B1-transferred and own-development thresholds | Tables G.4, G.5 and G.6 |
| `ablation_precision_check.py` | re-evaluates the archived out-of-fold ablation scores to recover the four-decimal values (no refitting) | Table G.7 |
| `h_editconf.py`, `p0_stats.py` | Qian-style edit_avg / edit_min rules on B1, and the structural input check run before it | Section 5.3.4; see script header |
| `i_lu_proxy.py` | earlier Lu-inspired proxy (without DD posteriors), superseded by the Lu-style analysis above | see script header |

Table G.8 reuses the existing GRECO comparison (`g_greco_model.py`, step 8) and Table G.9 the existing ablation (`f2_ablation.py`).

## GRECO (Qorib & Ng, EMNLP 2023) — obtaining and running
- Code: `git clone https://github.com/nusnlp/greco $GRECO_DIR && git -C $GRECO_DIR checkout 8e509aaacf4e5ec3276d3427371471fd82a7f58b` (GPL-3.0). `models.py` is used unchanged.
- Checkpoint: `curl -L -o $GRECO_DIR/models/checkpoint.bin https://huggingface.co/nusnlp/GRECO/resolve/main/checkpoint/checkpoint.bin`,
  1,748,823,771 bytes, sha256 `1a11360c12bdfb20660555fe1e4a71e28b4c21e0acd0b4077d8af80a8f096884` (checked by `e_after_q.sh`).
- Base LM: `microsoft/deberta-v3-large` (Models). `e_after_q.sh greco` stops with an error if the env, the clone, the checkpoint
  (or its sha256) or the DeBERTa snapshot is missing, and also if the Q queue lock file `$EXP_DIR/05_runs/D_state/queue_Q.lock`
  is missing (run step 4 first).
- Device: CPU. torch 1.12.1's MPS backend lacks an int64 op used by DeBERTa-v2. The official `score()` passes no attention mask, so
  `g_greco_score.py` batches only pairs of identical token length (no padding; verified identical to one-by-one scoring).
- Validation before the full run: dev-subset mapping check (all edits scored, 3 scores each, no NaN, token counts equal), and the
  authors' public example (CoNLL-2014 system outputs shipped in the GRECO repo; not redistributed here): with the repo's own
  `score_all.py` + `correlation.py` (default mode) Spearman rho 0.4152 (paper Table 2: 0.415) — reproduced; re-ranking with the README
  `rerank.py` F0.5 33.78 vs 54.40 in the paper — not reproduced, cause not identified. We use only GRECO's quality scores, never its
  re-ranking. Helper scripts: `g_greco_validate.py`, `g_greco_conll_eval.py`.
- Full run: dev-subset gate, then per system train/dev/eval scoring -> `g_greco_model.py`, eval once ->
  `$EXP_DIR/06_results/GRECO_<system>_eval.json`. Measured (CPU, nice 10): B1 31 min, W 28 min, Q 33 min of scoring + < 1 min modelling each; resumable at 200-utterance chunks.

## Changes to frozen scripts after the freeze
`configs/final.yaml` (`post_freeze_notes`, written for version 3; historical, unchanged) and `configs/post_freeze_changes.md`
(current, with the version-4 hashes and diffs). In short: `b3_edit_labels.py` was changed on 2026-11-11 only in its self-check
(P/R/F0.5 with the official `compute_p_r_f05`; labelling logic and outputs unchanged — B1 and W labels re-generated byte-identical);
in versions 3–4 `b3_edit_labels.py`, `c1_features.py`, `c_model.py` only had their path constant changed (now without default).
The reproduction audit confirmed B1 labels, label statistics, OOF metrics, frozen-model hashes and eval results identical to `results/` (not included in this repository (corpus licence; see the note at the top)); re-running
C2–C6 with the version-4 scripts again gives byte-identical frozen models.

## Licence
- **Code** (`scripts/`): MIT (see `LICENSE`), except the few lines copied from the official baseline listed in
  `docs/THIRD_PARTY_CODE.md`, which remain with their owner.
- **Documentation written by the authors** (this README, `docs/`, the text of `configs/` and `environment/`): CC BY 4.0, except
  the corpus-derived statistics they quote and the third-party lines in `environment/patches/` (official baseline: no licence
  stated, only the changed lines are given; NIST SCTK: public domain; see `environment/patches/README.md`).
- **`results/` and all corpus-derived statistics** wherever they appear: **no open licence is granted.** Subject to the Speak &
  Improve Corpus 2025 licence; released subject to CUP&A approval (requested 2026-11-11). See `docs/DATA_LICENSE_NOTE.md`.
- GRECO code (GPL-3.0), the corpus and the per-edit derived labels, features and model outputs are not included in this package; each remains under its own licence (GRECO: GPL-3.0; corpus and everything derived from its annotations: the Cambridge University Press & Assessment corpus licence). The GRECO-related scripts call a separately obtained GRECO clone and contain none of its source; running them together with GRECO is subject to the GPL-3.0.
- Not included either, each under its own licence: the official S&I 2025 baseline code and models (corpus licence); Whisper-large-v3
  (MIT), Qwen2.5-7B-Instruct (Apache-2.0), bert-base-uncased (Apache-2.0), xlm-roberta-base (MIT), deberta-v3-large (MIT) — see each
  model card at `https://huggingface.co/<repo>`.
