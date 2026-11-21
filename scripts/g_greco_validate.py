#!/usr/bin/env python3
"""GRECO sanity check on the authors' public example (repo data, no corpus): CoNLL-2014 official system outputs
(data/conll-official/texts, 13 systems incl. INPUT; source data/conll-source.txt). Re-ranks the first N source sentences exactly as
rerank.py (dedup hypotheses in system order, pick the highest GRECO score) and compares the pick with the authors' released
outputs/CoNLL-2014_GRECO.out; also system-level Spearman of the mean GRECO score vs each system's official F0.5 (m2 reports).
Scoring: unpadded length-bucketed batches (as g_greco_score.py). Env eswa-g, offline. Prints numbers only.
  python -I g_greco_validate.py --greco_dir <clone> --checkpoint <bin> --lm <snapshot> --n 300 --out <json> [--device mps|cpu]"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
for k in ["greco_dir", "checkpoint", "lm", "out"]:
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--n", type=int, default=300); ap.add_argument("--device", default="mps"); ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, a.greco_dir)
import torch  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402
from models import GRECO  # noqa: E402
dev = torch.device(a.device); torch.manual_seed(20261110)
model = GRECO(a.lm); model.load_state_dict(torch.load(a.checkpoint, map_location="cpu")); model = model.to(dev).eval(); model.device = dev
MAXLEN = model.config.max_position_embeddings - 2
T = f"{a.greco_dir}/data/conll-official/texts"
systems = {s: open(f"{T}/{s}").readlines() for s in os.listdir(T)}
systems["INPUT"] = open(f"{a.greco_dir}/data/conll-source.txt").readlines()
src = [x.strip() for x in systems["INPUT"]][:a.n]
released = open(f"{a.greco_dir}/outputs/CoNLL-2014_GRECO.out").read().split("\n")[:a.n]
cache = {}


def score(pairs):
    need = [p for p in dict.fromkeys(pairs) if p not in cache]; by = {}
    for p in need:
        by.setdefault(min(len(model.tokenizer(p[0], p[1])["input_ids"]), MAXLEN), []).append(p)
    for ps in by.values():
        for i in range(0, len(ps), a.batch):
            b = ps[i:i + a.batch]
            with torch.no_grad():
                out = model.score([x[0] for x in b], [x[1] for x in b])
            cache.update({p: float(v) for p, v in zip(b, out.reshape(-1).tolist())})
    return [cache[p] for p in pairs]


t0 = time.time()
agree = 0; sys_scores = {s: [] for s in systems}; neg = 0
for i, s in enumerate(src):
    order = {}
    for name in systems:
        order.setdefault(systems[name][i].strip(), []).append(name)
    hyps = list(order)
    sc = score([(s, h) for h in hyps]); neg += sum(v < 0 for v in sc)
    best = sorted(zip(sc, hyps), reverse=True, key=lambda x: x[0])[0][1]
    agree += int(best == released[i].strip())
    for h, v in zip(hyps, sc):
        for name in order[h]:
            sys_scores[name].append(v)
el = time.time() - t0
f05 = {}
for s in systems:
    rep = open(f"{a.greco_dir}/data/conll-official/reports/{s}.m2-report").read().strip().split("\n")
    g = [l for l in rep if l.startswith("F_0.5") and ":" in l]   # final corpus-level line of the m2scorer report
    f05[s] = float(g[-1].split(":")[1]) if g else None
names = [s for s in systems if f05[s] is not None]
m = [sum(sys_scores[s]) / len(sys_scores[s]) for s in names]; f = [f05[s] for s in names]
res = dict(n_sentences=len(src), n_pairs=len(cache), device=str(dev), rerank_agreement_with_released=agree / len(src), n_agree=agree,
           negative_scores=neg, system_level_spearman=float(spearmanr(m, f)[0]), system_level_pearson=float(pearsonr(m, f)[0]),
           n_systems=len(names), seconds=round(el, 1), s_per_pair=round(el / max(len(cache), 1), 4),
           mean_score_by_system={s: round(v, 4) for s, v in zip(names, m)}, official_f05=f05,
           commit=os.popen(f"git -C {a.greco_dir} rev-parse HEAD").read().strip())
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
