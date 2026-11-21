#!/usr/bin/env python3
"""Stage D-GEC: teacher-forced scores of the Q GEC output under Qwen2.5-7B-Instruct itself (the GEC model of chain Q), computed as
c1_bart_score.py does for BART, and written in the SAME per-phrase JSONL schema so that c1_features.py builds the GEC features unchanged.
Run in the eswa-d env from the corpus root (offline):
  python -I d_qwen_score.py --run_dir stageD/Q_dev --out <dir>/qwen_scores_dev.jsonl
Per phrase (rows of <run_dir>/gec/dd/pre-gec.tsv = the GEC input; output = the words of the phrase in <run_dir>/gec/gec.ctm):
  context = the frozen 3-shot prompt (d_qwen_prompt.py; shared prefix KV cache, trimmed back after every pass) + the phrase;
  pass 1 (output as forced assistant reply, then <|im_end|>): per output word sum / min of token log-probs, mean entropy of the
         next-token distribution, all-tokens-equal-argmax flag; seq_lp (incl. <|im_end|>), eos_lp, bos_lp (first token);
  pass 2 (copy: the input as forced reply): per input word sum / min log-prob; seq_copy_lp.
BART-specific fields are fixed: n_chunks 1, midword_cut 0 (no 126-token chunking). Tokens map to words through the tokenizer
offsets (leading space skipped). Diagnostics as in C1. Numbers only.
"""
import argparse, json, os, sys
from collections import OrderedDict
import mlx.core as mx
import pandas as pd
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_qwen_prompt as QP

REV = "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
repo = os.path.join((os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")), f"models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/{REV}")
if not os.path.isdir(repo): raise SystemExit(f"model not found: {repo} (download it first: README, Models; cache = HF_HUB_CACHE or HF_HOME/hub)")
model, tok = load(repo)
hf = tok._tokenizer
END = hf.convert_tokens_to_ids("<|im_end|>")
PRE, SUFFIX = QP.build(tok)
cache = make_prompt_cache(model); model(mx.array(PRE)[None], cache=cache); mx.eval([c.state for c in cache]); P0 = cache[0].offset

pre = pd.read_csv(f"{a.run_dir}/gec/dd/pre-gec.tsv", sep="\t", keep_default_na=False)
g = pd.read_csv(f"{a.run_dir}/gec/gec.ctm", sep="\t", header=None, keep_default_na=False)
gw = OrderedDict()
for pid, w in zip(g[0], g[4]):
    gw.setdefault(pid, []).append(str(w))
if a.limit:
    pre = pre.head(a.limit)


def word_ids(text, words):
    enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, p = [], 0
    for w in words:
        spans.append((p, p + len(w))); p += len(w) + 1
    wid = []
    for s, e in enc["offset_mapping"]:
        while s < e and text[s] == " ":
            s += 1
        wid.append(next((i for i, (a0, a1) in enumerate(spans) if a0 <= s < a1), -2))
    return enc["input_ids"], wid


def forced(suffix, tgt_ids):
    toks = suffix + tgt_ids + [END]
    logits = model(mx.array(toks)[None], cache=cache)[0].astype(mx.float32)
    trim_prompt_cache(cache, cache[0].offset - P0)
    s = len(suffix)
    lg = logits[s - 1:s - 1 + len(tgt_ids) + 1]             # predictions for every target token and <|im_end|>
    logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    tgt = mx.array(tgt_ids + [END])
    lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
    ent = -(mx.exp(logp) * logp).sum(-1)
    am = mx.argmax(logp, axis=-1) == tgt
    mx.eval(lp, ent, am)
    return lp.tolist(), ent.tolist(), am.tolist()


def per_word(nw, lp, ent, am, wid):
    s = [0.0] * nw; mn = [0.0] * nw; en = [[] for _ in range(nw)]; ag = [1] * nw; cov = [0] * nw
    for l, e, m, k in zip(lp, ent, am, wid):
        if k < 0:
            continue
        s[k] += l; mn[k] = min(mn[k], l) if cov[k] else l; en[k].append(e); ag[k] &= int(m); cov[k] += 1
    return s, mn, [sum(x) / len(x) if x else float("nan") for x in en], ag, cov


diag = dict(phrases=len(pre), chunked_phrases=0, midword_cuts=0, roundtrip_fail=0, words_uncovered=0, tokens_unmapped=0,
            tgt_tokens=0, tgt_tokens_argmax=0, empty_tgt=0, scorer="qwen2.5-7b-instruct-4bit teacher forcing (3-shot prompt)")
with open(a.out, "w") as f:
    for n, (pid, src) in enumerate(zip(pre["ids"], pre["sentence_flt"])):
        src = str(src); sw = src.split(" "); tw = gw.get(pid, []); tt = " ".join(tw)
        suf = SUFFIX(src)
        t_ids, t_wid = word_ids(tt, tw) if tw else ([], [])
        s_ids, s_wid = word_ids(src, sw)
        lp, en, am = forced(suf, t_ids)
        lp2, en2, am2 = forced(suf, s_ids)
        if tw and hf.decode(t_ids) != tt:
            diag["roundtrip_fail"] += 1
        diag["tokens_unmapped"] += sum(1 for w in t_wid if w == -2)
        diag["tgt_tokens"] += len(lp); diag["tgt_tokens_argmax"] += sum(am); diag["empty_tgt"] += int(not tw)
        s, mn, e, ag, cov = per_word(len(tw), lp[:-1], en[:-1], am[:-1], t_wid)
        s2, mn2, _, _, cov2 = per_word(len(sw), lp2[:-1], en2[:-1], am2[:-1], s_wid)
        diag["words_uncovered"] += sum(1 for c in cov if c == 0) + sum(1 for c in cov2 if c == 0)
        rec = dict(id=pid, n_chunks=1, n_src=len(sw), n_tgt=len(tw), midword_cut=0, t_lp=s, t_min=mn, t_ent=e, t_agree=ag,
                   s_lp=s2, s_min=mn2, seq_lp=sum(lp), seq_copy_lp=sum(lp2), eos_lp=lp[-1], bos_lp=lp[0])
        f.write(json.dumps(rec) + "\n")
        if n % 500 == 0:
            print(f"phrase {n} / {len(pre)}", flush=True)
diag["argmax_agreement"] = round(diag["tgt_tokens_argmax"] / max(diag["tgt_tokens"], 1), 4)
json.dump(diag, open(a.out.replace(".jsonl", ".diag.json"), "w"), indent=1)
print(json.dumps(diag))
