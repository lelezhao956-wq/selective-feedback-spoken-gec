#!/usr/bin/env python3
"""Stage D-GEC: LLM GEC (Qwen2.5-7B-Instruct, 4-bit MLX) on the official DD output of B1 (pre-gec.tsv), one call per phrase.
v2 (2026-11-10, decided by the authors): 3-shot prompt from d_qwen_prompt.py (frozen), the shared prefix is computed once and its KV cache
reused (trimmed back after every phrase); raw output and format flags saved. v1 (zero-shot): d_llm_gec.py.orig.
Run in the eswa-d env from the corpus root (offline):
  HF_HUB_OFFLINE=1 python -I d_llm_gec.py --input stageB/B1_dev/gec/dd/pre-gec.tsv --out_dir stageD/Q_dev/llm
3-shot minimal-edit prompt (system message + three self-written examples from d_qwen_prompt.py, chat template; the
zero-shot v1 is kept only as d_llm_gec.py.orig, not distributed), greedy decoding (deterministic), max_tokens = 2 * words + 20.
Post-processing to the input convention of the official GEC step (lower case, no punctuation except apostrophes; checked on
B1 dev: the DD output and the BART output contain no upper case and almost no punctuation): lower-case, drop characters
other than letters, digits, apostrophes and spaces (a hyphen or slash becomes a space), collapse spaces, first line only.
Empty result -> the input phrase unchanged (counted). Saves per phrase: output, number of generated tokens, mean and min
token log-probability (greedy token under the model), truncation flag, seconds. Resume: one JSON line per phrase, appended
and flushed; complete lines in <out_dir>/gen.jsonl and in <out_dir>.partial-*/gen.jsonl (copied, never moved) are reused.
Progress: number of lines in gen.jsonl. Prints one JSON summary line; writes <out_dir>/llm_gec.tsv (ids, sentence_gec).
"""
import argparse, glob, json, os, re, sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hashlib
import d_qwen_prompt as QP
PROMPT_ID = hashlib.sha256(json.dumps([QP.SYS, QP.SHOTS]).encode()).hexdigest()[:12]

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"; REV = "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
rows = [l.rstrip("\n").split("\t") for l in open(a.input)]
assert rows[0] == ["ids", "data_flt", "sentence_flt"], rows[0][:3]
rows = rows[1:][: a.limit or None]
os.makedirs(a.out_dir, exist_ok=True)
G = f"{a.out_dir}/gen.jsonl"


def read_done(p):
    out = {}
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l); out[(r["i"], r["ids"])] = r
            except Exception:
                pass   # incomplete last line after a crash
    return out


done = read_done(G)
part = {}
for pp in sorted(glob.glob(f"{a.out_dir}.partial-*/gen.jsonl")):
    for k, r in read_done(pp).items():
        if r.get("prompt_id") == PROMPT_ID:      # only lines of the same prompt version
            part.setdefault(k, r)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ").replace("/", " "))).strip()
repo = os.path.join((os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")), f"models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/{REV}")
if not os.path.isdir(repo): raise SystemExit(f"model not found: {repo} (download it first: README, Models; cache = HF_HUB_CACHE or HF_HOME/hub)")
model, tok = load(repo)
PRE, SUFFIX = QP.build(tok)
cache = make_prompt_cache(model)
model(mx.array(PRE)[None], cache=cache); mx.eval([c.state for c in cache])
P0 = cache[0].offset
n_fmt = 0
n_kept = n_reused = n_gen = n_empty = n_trunc = 0; T = 0.0
with open(G, "a") as f:
    for i, r in enumerate(rows):
        k = (i, r[0])
        if k in done and done[k].get("input") == r[2] and done[k].get("prompt_id") == PROMPT_ID:
            n_kept += 1; continue
        if k in part and part[k].get("input") == r[2] and part[k].get("prompt_id") == PROMPT_ID:
            f.write(json.dumps(part[k]) + "\n"); f.flush(); n_reused += 1; continue
        n = len(r[2].split()); mt = 2 * n + 20
        pr = SUFFIX(r[2])
        t = time.time(); text = ""; lps = []; ng = 0
        for resp in stream_generate(model, tok, prompt=pr, max_tokens=mt, prompt_cache=cache):
            text += resp.text; ng = resp.generation_tokens
            if resp.logprobs is not None and resp.token is not None:
                lps.append(float(resp.logprobs[resp.token]))
        e = time.time() - t; T += e
        trim_prompt_cache(cache, cache[0].offset - P0); assert cache[0].offset == P0
        raw = text.strip(); lines = [x for x in raw.split("\n") if x.strip()]
        out = norm(lines[0]) if lines else ""
        empty = not out
        if empty:
            out = r[2]; n_empty += 1
        nw_out = len(out.split()) if not empty else 0
        fmt = dict(multiline=len(lines) > 1, empty=empty, truncated=ng >= mt, changed_by_norm=bool(lines) and norm(lines[0]) != lines[0].strip(),
                   length_ratio_out_of_range=not empty and not (0.5 <= nw_out / max(n, 1) <= 2.0))
        n_fmt += any(fmt.values())
        n_trunc += ng >= mt; n_gen += 1
        rec = dict(i=i, ids=r[0], prompt_id=PROMPT_ID, input=r[2], raw=raw, output=out, format=fmt, format_error=any(fmt.values()), empty_fallback=empty,
                   gen_tokens=ng, truncated=ng >= mt,
                   lp_mean=sum(lps) / len(lps) if lps else None, lp_min=min(lps) if lps else None, sec=round(e, 3))
        f.write(json.dumps(rec) + "\n"); f.flush()
allr = {k: r for k, r in read_done(G).items() if r.get("prompt_id") == PROMPT_ID}
assert all((i, r[0]) in allr for i, r in enumerate(rows)), "missing phrases"
with open(f"{a.out_dir}/llm_gec.tsv", "w") as f:
    f.write("ids\tsentence_gec\n")
    for i, r in enumerate(rows):
        f.write(f"{r[0]}\t{allr[(i, r[0])]['output']}\n")
fe = sum(1 for r in allr.values() if r.get("format_error"))
summ = dict(prompt=f"3-shot ({QP.VERSION}, prompt_id {PROMPT_ID})", format_errors=fe, format_error_rate=round(fe / max(len(rows), 1), 4),
            phrases=len(rows), kept=n_kept, reused_from_partial=n_reused, generated=n_gen, empty_fallback=n_empty, truncated=n_trunc,
            gen_s=round(T, 1), s_per_phrase=round(T / n_gen, 3) if n_gen else None, model=MODEL, rev=REV)
json.dump(summ, open(f"{a.out_dir}/llm_summary.json", "w"), indent=1)
print(json.dumps(summ))
