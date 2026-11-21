#!/usr/bin/env python3
"""B2 step 1: convert a manual (fluent) STM into a Whisper-like "pre-norm" CTM so that the
OFFICIAL run-asr-to-task-input.sh --task gec (phrase markers + ctm_norm) can be applied unchanged.

- Only files listed in --flist (the {ds}-gec.tsv list) are kept, in flist order.
- STM header lines (';;') and IGNORE_TIME_SEGMENT_IN_SCORING segments are dropped
  (the SGECF scorer removes hypothesis words falling inside those segments anyway).
- STM segments carry no word times: the n words of a segment [start, end) get uniform
  durations (end-start)/n and consecutive start times, formatted with 2 decimals (as in Whisper CTMs).
- Confidence column is set to 1.00 (manual transcript).
- Punctuation and case are kept, because add_phrase_markers.py uses '!?.' to split phrases and
  ctm_norm.py does the lower-casing / punctuation removal, exactly as for ASR input.
- --phrase_unit segment (default): if the last word of an STM segment does not end in [.?!], a '.'
  is appended to it, so that the official add_phrase_markers.py starts a new phrase at every STM
  segment boundary (STM segments = the unit of the manual GEC annotation). The appended '.' is
  removed again by ctm_norm.py, so the words seen by GEC are unchanged.
  --phrase_unit punct: no marker added (phrases only where the manual transcript has [.?!]).
- --drop_in_ignore_of STM2 (used for the verbatim condition B2b): additionally drop every word whose uniform-time
  midpoint falls inside an IGNORE_TIME_SEGMENT_IN_SCORING segment of STM2 (the fluent STM). Needed because the
  verbatim STM marks far fewer segments IGNORE than the fluent/GEC STMs (dev 359 vs 1591) and is segmented
  differently (746/1591 fluent IGNORE segments have identical bounds in the verbatim STM), so the scorer's
  time-based filter would otherwise keep verbatim words in regions excluded from GEC scoring.

- --drop_markup (added 2026-11-10, used for B2b v2 and the label reference V v4): drop transcription markup tokens of the
  manual STMs: bracketed filled-pause markers '(%...%)', bracketed partial words '(...)' (all end in '-'), and bare '%...%'
  markers. The official ctm_norm.py removes the same phenomena from Whisper output (remove_hesitation: um/er/...;
  remove_partial_words: 'wo-'), but it does not recognise the STM's bracketed encoding. Word times of all other words
  are computed BEFORE dropping, so they are identical to a run without the option.

Usage: python -I stm_to_ctm.py --stm STM --flist TSV --out CTM [--phrase_unit segment|punct] [--drop_markup]
"""
import argparse
import re

MARKUP = re.compile(r"\(%[^)]*%\)|\([^)]*\)|%[^%\s]+%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stm", required=True)
    ap.add_argument("--flist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrase_unit", choices=["segment", "punct"], default="segment")
    ap.add_argument("--drop_in_ignore_of", default=None)
    ap.add_argument("--drop_markup", action="store_true")
    a = ap.parse_args()

    order = [l.split("\t")[0].strip() for l in open(a.flist, encoding="utf-8") if l.strip()]
    keep = set(order)
    segs = {}
    n_ignored = n_segs = 0
    for line in open(a.stm, encoding="utf-8"):
        if line.startswith(";;") or not line.strip():
            continue
        parts = line.split(maxsplit=5)
        fid, start, end = parts[0], float(parts[3]), float(parts[4])
        rest = parts[5] if len(parts) > 5 else ""
        # rest = "<label> words..."
        if rest.startswith("<"):
            close = rest.find(">")
            words = rest[close + 1:].split()
        else:
            words = rest.split()
        if fid not in keep:
            continue
        if words == ["IGNORE_TIME_SEGMENT_IN_SCORING"]:
            n_ignored += 1
            continue
        if not words:
            continue
        n_segs += 1
        segs.setdefault(fid, []).append((start, end, words))

    ign = {}
    if a.drop_in_ignore_of:
        for line in open(a.drop_in_ignore_of, encoding="utf-8"):
            if line.startswith(";;") or not line.strip():
                continue
            p = line.split(maxsplit=5)
            if p[0] in keep and len(p) > 5 and "IGNORE_TIME_SEGMENT_IN_SCORING" in p[5]:
                ign.setdefault(p[0], []).append((float(p[3]), float(p[4])))
    n_words = n_marked = n_dropped_ign = n_markup = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for fid in order:
            for start, end, words in sorted(segs.get(fid, []), key=lambda s: s[0]):
                dur = (end - start) / len(words)
                kept = []
                for i, w in enumerate(words):
                    mid = start + (i + 0.5) * dur
                    if any(s0 <= mid < e0 for s0, e0 in ign.get(fid, [])):
                        n_dropped_ign += 1
                        continue
                    if a.drop_markup and MARKUP.fullmatch(w.strip(".?!,")):
                        n_markup += 1
                        continue
                    kept.append((start + i * dur, w))
                if not kept:
                    continue
                if a.phrase_unit == "segment" and kept[-1][1][-1] not in "!?.":
                    kept[-1] = (kept[-1][0], kept[-1][1] + ".")
                    n_marked += 1
                for t, w in kept:
                    fo.write(f"{fid} 1 {t:.2f} {dur:.2f} {w} 1.00\n")
                    n_words += 1
    missing = [f for f in order if f not in segs]
    print(f"files_in_flist={len(order)} files_with_words={len(order) - len(missing)} "
          f"segments_used={n_segs} ignore_segments_dropped={n_ignored} words={n_words} "
          f"phrase_unit={a.phrase_unit} segment_end_markers_added={n_marked} dropped_in_ignore_of={n_dropped_ign} markup_dropped={n_markup}")


if __name__ == "__main__":
    main()
