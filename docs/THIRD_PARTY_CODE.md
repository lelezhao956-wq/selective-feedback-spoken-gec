# Third-party code lines inside the study scripts

The study scripts below contain a few lines copied verbatim from the official Speak & Improve Challenge 2025 baseline (shipped
with the corpus as `baseline-pipeline.zip` and `scoring-tools.zip`). They are listed here; the scripts themselves are frozen
(their hashes are part of the frozen configuration or of the verified reproduction) and are not changed. The lines are not
reproduced in this document.

**Licence of the source files**: no licence file or licence terms were found in the official zips or the corpus distribution
(checked 2026-11-11). `gec-plus-align.py` and `dd.py` carry only the header line "Copyright: S&I Challenge 2024";
`spoken_errant.py` carries no copyright or licence line. These lines therefore remain with their owner (named in the headers only
as "S&I Challenge 2024") and are **not** covered by the MIT licence
of `scripts/`; the MIT licence applies to the rest of each file.

| study script (package) | line | source file (official zip) | source line | what it is |
|---|---|---|---|---|
| `scripts/b3_edit_labels.py` | 90 | `scoring-tools/sgecf-tools/spoken_errant.py` | 101 | one statement of the edit-extraction logic, kept identical so that labels match the official scorer |
| `scripts/b3_edit_labels.py` | 98 | `scoring-tools/sgecf-tools/spoken_errant.py` | 110 | as above |
| `scripts/d_llm_align.py` | 27 | `baseline-pipeline/gec-tools/gec-plus-align.py` | 193 | CTM-writing code after the GEC step (docstring of `d_llm_align.py`: "imported/copied from the official script, unchanged") |
| `scripts/d_llm_align.py` | 28 | `baseline-pipeline/gec-tools/gec-plus-align.py` | 195 | as above |
| `scripts/d_llm_align.py` | 29 | `baseline-pipeline/gec-tools/gec-plus-align.py` | 196 | as above |
| `scripts/d_llm_align.py` | 32 | `baseline-pipeline/gec-tools/dd.py` | 171 | as above |
| `scripts/d_llm_align.py` | 33 | `baseline-pipeline/gec-tools/gec-plus-align.py` | 202 | as above |
| `scripts/d_llm_align.py` | 34 | `baseline-pipeline/gec-tools/gec-plus-align.py` | 204 | as above (found by the shorter-line check below) |

**How the lines were found**: every non-comment line of >= 40 characters of the official `.py`/`.sh` files (703 distinct lines)
was compared, after stripping white space, with every line of the package; outside `environment/patches/` only the 7 lines with
source lines 101, 110, 193, 195, 196, 171 and 202 matched. A second pass with >= 25 characters added `d_llm_align.py:34`; its
other matches are generic one-liners (standard imports, `if __name__ == "__main__":`, a `split(maxsplit=5)` call) and are not listed.
`scripts/g_greco_score.py` re-implements the two-line `get_file_data()` reader of `spoken_errant.py` (stated in a comment; not a
verbatim copy). The official portability patches are covered in `environment/patches/README.md`.
