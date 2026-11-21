#!/usr/bin/env python3
"""C1a: teacher-forced (forced-decoding) scores of the EXISTING B1 GEC output with the same BART model.
No new GEC generation. Run with the sandi-all env from the corpus root:
  python c1_bart_score.py --run_dir stageB/B1_dev --out <dir>/bart_scores_dev.jsonl [--limit N]

Per phrase (rows of <run_dir>/gec/dd/pre-gec.tsv = the exact GEC input; GEC output = the words of that phrase in
<run_dir>/gec/gec.ctm, which gec-plus-align.py writes as sentence_gec.split() in order):
  - the official 126-BART-token chunking of the input is replicated (gec-plus-align.process_and_split_sentences);
    for multi-chunk phrases the output words are split between chunks with the same SequenceMatcher alignment that
    gec-plus-align uses (flag `chunked`), because the per-chunk outputs were concatenated before the CTM was written;
  - pass 1 (GEC output as forced target): per output word sum / min of BART token log-probs, mean entropy of the
    predictive distribution, and whether every token equals the argmax (greedy agreement);
  - pass 2 (copy, input as forced target): per input word sum / min log-prob of reproducing the word unchanged.
Diagnostics per phrase: BPE round trip of the output string, token->word coverage, argmax agreement.
Only numbers are written (no words).
"""
import argparse
import json
import math
from collections import OrderedDict
from difflib import SequenceMatcher

import pandas as pd
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--gec_model", default="sandi-models/gec/gec_model")
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.gec_model)
model = BartForConditionalGeneration.from_pretrained(a.gec_model).eval()

pre = pd.read_csv(f"{a.run_dir}/gec/dd/pre-gec.tsv", sep="\t")
g = pd.read_csv(f"{a.run_dir}/gec/gec.ctm", sep="\t", header=None, keep_default_na=False)
gw = OrderedDict()
for pid, w in zip(g[0], g[4]):
    gw.setdefault(pid, []).append(str(w))
if a.limit:
    pre = pre.head(a.limit)


def chunks_of(src):
    enc = tok(src, return_offsets_mapping=True, add_special_tokens=False)
    ids, sp = enc["input_ids"], enc["offset_mapping"]
    out = []
    for i in range(0, len(ids), 126):
        out.append((sp[i:i + 126][0][0], sp[i:i + 126][-1][1]))
    return out


def word_spans(words):
    spans, p = [], 0
    for w in words:
        spans.append((p, p + len(w)))
        p += len(w) + 1
    return spans


def split_tgt(src_words, tgt_words, src_word_bounds):
    """Split tgt words at the src word boundaries using the SequenceMatcher alignment (as in gec-plus-align)."""
    m = SequenceMatcher(None, src_words, tgt_words)
    s2t = {}
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        for i in range(i1, i2):
            s2t[i] = j1 + min(i - i1, max(j2 - j1, 0))
    cuts = [s2t.get(b, len(tgt_words)) for b in src_word_bounds]
    parts, prev = [], 0
    for c in cuts + [len(tgt_words)]:
        parts.append(tgt_words[prev:c]); prev = c
    return parts


items = []   # (phrase_id, chunk_no, src_text, tgt_words, src_words, src_word_offset, tgt_word_offset)
meta = {}
for pid, src in zip(pre["ids"], pre["sentence_flt"]):
    src = str(src)
    sw = src.split(" ")
    tw = gw.get(pid, [])
    ch = chunks_of(src)
    meta[pid] = dict(n_chunks=len(ch), n_src=len(sw), n_tgt=len(tw), midword_cut=0)
    if len(ch) == 1:
        items.append((pid, 0, src, tw, sw, 0, 0))
        continue
    sspan = word_spans(sw)
    bounds = []
    for (c0, c1) in ch[1:]:
        b = next((k for k, (s0, s1) in enumerate(sspan) if s0 >= c0), len(sw))
        if not any(s0 == c0 for s0, _ in sspan):
            meta[pid]["midword_cut"] = 1
        bounds.append(b)
    tparts = split_tgt(sw, tw, bounds)
    sb = [0] + bounds + [len(sw)]
    toff = 0
    for k in range(len(ch)):
        s_words = sw[sb[k]:sb[k + 1]]
        items.append((pid, k, " ".join(s_words), tparts[k], s_words, sb[k], toff))
        toff += len(tparts[k])


def tok_word_map(text, words):
    enc = tok(text, return_offsets_mapping=True)
    spans = word_spans(words)
    wid = []
    for (s, e) in enc["offset_mapping"]:
        if s == e:
            wid.append(-1)       # special tokens
            continue
        k = next((i for i, (a0, a1) in enumerate(spans) if a0 <= s < a1), -2)
        wid.append(k)
    return enc["input_ids"], wid


@torch.no_grad()
def forced(src_texts, tgt_texts, tgt_words_list):
    enc = tok(src_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    lab, maps = [], []
    for t, w in zip(tgt_texts, tgt_words_list):
        ids, wid = tok_word_map(t, w)
        lab.append(ids); maps.append(wid)
    L = max(len(x) for x in lab)
    labels = torch.full((len(lab), L), -100, dtype=torch.long)
    for i, x in enumerate(lab):
        labels[i, :len(x)] = torch.tensor(x)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
    logp = torch.log_softmax(out.logits.float(), dim=-1)
    res = []
    for i, x in enumerate(lab):
        n = len(x)
        lp = logp[i, :n].gather(1, torch.tensor(x).unsqueeze(1)).squeeze(1)
        ent = -(logp[i, :n].exp() * logp[i, :n]).sum(-1)
        am = logp[i, :n].argmax(-1) == torch.tensor(x)
        res.append((lp.tolist(), ent.tolist(), am.tolist(), maps[i]))
    return res


def per_word(nw, lp, ent, am, wid):
    s = [0.0] * nw; mn = [0.0] * nw; en = [[] for _ in range(nw)]; ag = [1] * nw; cov = [0] * nw
    for l, e, m, k in zip(lp, ent, am, wid):
        if k < 0:
            continue
        s[k] += l; mn[k] = min(mn[k], l) if cov[k] else l; en[k].append(e); ag[k] &= int(m); cov[k] += 1
    return s, mn, [sum(x) / len(x) if x else float("nan") for x in en], ag, cov


acc = {}
order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
diag = dict(phrases=len(pre), chunked_phrases=sum(1 for v in meta.values() if v["n_chunks"] > 1),
            midword_cuts=sum(v["midword_cut"] for v in meta.values()), roundtrip_fail=0, words_uncovered=0,
            tokens_unmapped=0, tgt_tokens=0, tgt_tokens_argmax=0, empty_tgt=0)
for b in range(0, len(order), a.batch):
    idx = order[b:b + a.batch]
    batch = [items[i] for i in idx]
    src_t = [x[2] for x in batch]
    tgt_t = [" ".join(x[3]) for x in batch]
    r1 = forced(src_t, tgt_t, [x[3] for x in batch])
    r2 = forced(src_t, src_t, [x[4] for x in batch])
    for (pid, k, st, tw, sw, soff, toff), (lp, en, am, wid), (lp2, en2, am2, wid2), tt in zip(batch, r1, r2, tgt_t):
        if tok.decode(tok(tt)["input_ids"], skip_special_tokens=True) != tt:
            diag["roundtrip_fail"] += 1
        diag["tokens_unmapped"] += sum(1 for w in wid if w == -2)
        diag["tgt_tokens"] += len(lp); diag["tgt_tokens_argmax"] += sum(am)
        diag["empty_tgt"] += int(len(tw) == 0)
        s, mn, e, ag, cov = per_word(len(tw), lp, en, am, wid)
        s2, mn2, _, _, cov2 = per_word(len(sw), lp2, en2, am2, wid2)
        diag["words_uncovered"] += sum(1 for c in cov if c == 0) + sum(1 for c in cov2 if c == 0)
        eos_lp = lp[-1] if lp else float("nan")
        acc.setdefault(pid, []).append(dict(k=k, soff=soff, toff=toff, t_lp=s, t_min=mn, t_ent=e, t_agree=ag,
                                            s_lp=s2, s_min=mn2, eos_lp=eos_lp, bos_lp=lp[0] if lp else float("nan"), seq_lp=sum(lp), seq_copy_lp=sum(lp2)))
    if (b // a.batch) % 50 == 0:
        print(f"batch {b // a.batch} / {math.ceil(len(order) / a.batch)}", flush=True)

with open(a.out, "w") as f:
    for pid in pre["ids"]:
        parts = sorted(acc[pid], key=lambda x: x["k"])
        rec = dict(id=pid, **meta[pid], t_lp=[], t_min=[], t_ent=[], t_agree=[], s_lp=[], s_min=[],
                   seq_lp=sum(p["seq_lp"] for p in parts), seq_copy_lp=sum(p["seq_copy_lp"] for p in parts),
                   eos_lp=parts[-1]["eos_lp"], bos_lp=sum(p["bos_lp"] for p in parts))
        for p in parts:
            for key in ["t_lp", "t_min", "t_ent", "t_agree", "s_lp", "s_min"]:
                rec[key] += p[key]
        assert len(rec["t_lp"]) == meta[pid]["n_tgt"] and len(rec["s_lp"]) == meta[pid]["n_src"]
        f.write(json.dumps(rec) + "\n")
diag["argmax_agreement"] = round(diag["tgt_tokens_argmax"] / max(diag["tgt_tokens"], 1), 4)
json.dump(diag, open(a.out.replace(".jsonl", ".diag.json"), "w"), indent=1)
print(json.dumps(diag))
