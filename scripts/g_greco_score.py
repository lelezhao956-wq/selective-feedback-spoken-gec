#!/usr/bin/env python3
"""GRECO comparison (added after the C7 eval; plan 03_configs/2026-11-10_GRECO_plan_supplement.yaml). Per-edit GRECO scores.
GRECO (Qorib & Ng, EMNLP 2023; github.com/nusnlp/greco, GPL-3.0) = DeBERTa-v3-large quality estimator trained on WRITTEN GEC data;
used here zero-shot on spoken-learner transcripts (cross-domain). Env eswa-g (torch), offline. Numbers only.
  python -I g_greco_score.py --greco_dir <clone> --checkpoint <checkpoint.bin> --run_dir <corpus run dir> --labels <labels_v4 dir>
         --ds <ds> --out <csv> [--limit N]
Inputs per utterance (row i of <labels>/utt_index_<ds>.csv): the official ERRANT input lines (<run_dir>/gec/sgecf-scoring/errant/
flt_ctm.txt = system fluent source S, gec_ctm.txt = GEC hypothesis G; spaCy tokens joined by spaces, as used by the official scorer).
Per hypothesis edit e (hyp_edit_labels_<ds>.csv: o_start/o_end in S, c_start/c_end in G):
  greco_single = q(S, S with only e applied) - q(S, S)          (gain of e alone; zero-shot risk = -greco_single)
  greco_loo    = q(S, G) - q(S, G with e reverted)              (contribution of e inside the full hypothesis)
  greco_sent   = q(S, G)                                        (sentence-level quality of the full hypothesis)
q = GRECO.score(source, hypothesis) of the official models.py (unchanged). Resumable: rows already in <out> are kept.
"""
import argparse, csv, hashlib, json, math, os, sys, time
import pandas as pd
ap = argparse.ArgumentParser()
for k in ["greco_dir", "checkpoint", "run_dir", "labels", "ds", "out"]:
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--subset_flist", default=""); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
ap.add_argument("--chunk", type=int, default=200)
ap.add_argument("--lm", default=os.path.join((os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")), "models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"))
a = ap.parse_args()
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, a.greco_dir)
for _p, _what in [(a.lm, "deberta-v3-large snapshot"), (a.checkpoint, "GRECO checkpoint"), (os.path.join(a.greco_dir, "models.py"), "GRECO clone")]:
    if not os.path.exists(_p): raise SystemExit(f"{_what} not found: {_p} (README, GRECO)")
import torch  # noqa: E402
from models import GRECO  # noqa: E402  (official, unchanged)
CORPUS = (os.environ.get("RUN_DIR") or __import__("sys").exit("RUN_DIR is not set: build a run root with make_run_root.sh and export RUN_DIR (README, Setup)"))


def get_file_data(fname):   # = spoken_errant.get_file_data (official; not imported because it needs errant/spaCy)
    with open(fname) as fh:
        return [x.strip("\n") for x in fh.readlines()]


dev = torch.device("cpu" if a.device == "auto" else a.device)   # auto = cpu: torch 1.12.1 MPS lacks int64 neg used by DeBERTa-v2 (2026-11-11)
torch.manual_seed(20261110)
model = GRECO(a.lm); model.load_state_dict(torch.load(a.checkpoint, map_location="cpu")); model = model.to(dev); model.eval()
model.device = dev   # official score() sends input_ids to self.device (hard-coded cpu/cuda); attribute set here, models.py unchanged
MAXLEN = model.config.max_position_embeddings - 2
ed = f"{a.run_dir}/gec/sgecf-scoring/errant"
S_lines, G_lines = get_file_data(f"{ed}/flt_ctm.txt"), get_file_data(f"{ed}/gec_ctm.txt")
ids = list(pd.read_csv(f"{a.labels}/utt_index_{a.ds}.csv")["utt_id"]); pos = {u: i for i, u in enumerate(ids)}
lab = pd.read_csv(f"{a.labels}/hyp_edit_labels_{a.ds}.csv")
if a.subset_flist:   # e.g. reference-materials/flists.flac/dev-subset-gec.tsv (utt_id = file ID)
    keep = set(pd.read_csv(a.subset_flist, sep="\t", header=None)[0].astype(str).str.split().str[0]); lab = lab[lab["utt_id"].isin(keep)]
if a.limit:
    lab = lab[lab["utt_id"].isin(list(dict.fromkeys(lab["utt_id"]))[:a.limit])]
done = set()
if os.path.exists(a.out):
    done = {(r["utt_id"], int(r["edit_idx"])) for r in csv.DictReader(open(a.out))}
diag = dict(n_edits=len(lab), n_utts=int(lab["utt_id"].nunique()), kept=len(done), token_count_mismatch=0, empty_source=0, device=str(dev),
            pairs_scored=0, pairs_truncated=0, nonfinite_rows=0, utts_scored_now=0)
q_cache = {}


def q(pairs):
    """GRECO scores for (src, hyp) pairs, cached. The official score() passes no attention mask, so padding inside a batch would
    change the scores: pairs are therefore batched only with pairs of identical token length (no padding = same as one by one)."""
    need = [p for p in dict.fromkeys(pairs) if p not in q_cache]
    by_len = {}
    for p in need:
        n = len(model.tokenizer(p[0], p[1])["input_ids"])
        diag["pairs_truncated"] += int(n > MAXLEN)
        by_len.setdefault(min(n, MAXLEN), []).append(p)
    for n, ps in by_len.items():
        for i in range(0, len(ps), a.batch):
            b = ps[i:i + a.batch]
            with torch.no_grad():
                out = model.score([x[0] for x in b], [x[1] for x in b])
            for p, v in zip(b, out.reshape(-1).tolist()):
                q_cache[p] = float(v)
            diag["pairs_scored"] += len(b)
    return [q_cache[p] for p in pairs]


new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
f = open(a.out, "a", newline=""); w = csv.writer(f)
if new:
    w.writerow(["utt_id", "edit_idx", "greco_single", "greco_loo", "greco_sent"])
t0 = time.time(); n = 0
todo = []
for u, g in lab.groupby("utt_id", sort=False):
    g = g[[(u, int(k)) not in done for k in g["edit_idx"]]]
    if not g.empty:
        todo.append((u, g))
# chunks of --chunk utterances: all their pairs are scored together in length buckets (larger unpadded batches), then the rows of the
# chunk are written and flushed (resumable at chunk granularity)
for c0 in range(0, len(todo), a.chunk):
    items = []
    for u, g in todo[c0:c0 + a.chunk]:
        S = S_lines[pos[u]].split(" "); G = G_lines[pos[u]].split(" ")
        diag["token_count_mismatch"] += int((g["n_src_tokens"] != len(S)).any())
        src, hyp = " ".join(S), " ".join(G)
        pairs = [(src, src), (src, hyp)]
        for e in g.itertuples(index=False):
            pairs.append((src, " ".join(S[:e.o_start] + G[e.c_start:e.c_end] + S[e.o_end:])))
            pairs.append((src, " ".join(G[:e.c_start] + S[e.o_start:e.o_end] + G[e.c_end:])))
        items.append((u, g, pairs))
    q([p for _, _, ps in items for p in ps])
    for u, g, pairs in items:
        sc = q(pairs); q_ss, q_sg = sc[0], sc[1]; diag["utts_scored_now"] += 1
        for j, e in enumerate(g.itertuples(index=False)):
            row = [sc[2 + 2 * j] - q_ss, q_sg - sc[3 + 2 * j], q_sg]
            diag["nonfinite_rows"] += int(not all(map(math.isfinite, row)))
            w.writerow([u, int(e.edit_idx)] + row); n += 1
    f.flush(); q_cache.clear()
    print(f"chunk {c0 // a.chunk + 1}/{(len(todo) + a.chunk - 1) // a.chunk} utts {min(c0 + a.chunk, len(todo))}/{len(todo)} "
          f"{round(time.time() - t0)} s", flush=True)
f.close()
el = time.time() - t0
diag.update(scored_now=n, seconds=round(el, 1), s_per_edit=round(el / max(n, 1), 4), s_per_utt=round(el / max(diag["utts_scored_now"], 1), 4),
            s_per_pair=round(el / max(diag["pairs_scored"], 1), 4), commit=os.popen(f"git -C {a.greco_dir} rev-parse HEAD").read().strip(),
            checkpoint_sha256=hashlib.sha256(open(a.checkpoint, "rb").read()).hexdigest())
json.dump(diag, open(a.out.replace(".csv", ".diag.json"), "w"), indent=1); print(json.dumps(diag))
