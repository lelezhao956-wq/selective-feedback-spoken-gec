"""F1: build blind human-validation annotation sheet (B1 chain, eval). Prints counts only, never corpus text."""
import os, json, glob, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
H = os.path.expanduser("~/mnt/EXP_DIR"); C = os.path.expanduser("~/mnt/sandi-corpus-2025-2")
R = f"{H}/05_runs/2026-11-10_C7_B1_eval"; E = f"{C}/stageB/B1_eval/gec/sgecf-scoring/errant"
OUT = f"{H}/11_human_validation"; os.makedirs(OUT, exist_ok=True)
SEED, PER = 20261112, 50
h = pd.read_csv(f"{R}/labels_v4/hyp_edit_labels_eval.csv"); s = pd.read_csv(f"{R}/eval_scores_frozen.csv")
d = h.merge(s[["utt_id", "edit_idx", "s_risk_lgbm", "s_risk_lr", "s_gec_lr"]], on=["utt_id", "edit_idx"])
assert len(d) == len(h) == len(s)
N = len(d); k = N - N // 2  # coverage 0.5 on eval, coverage-based endpoint as in the frozen protocol
def kept_by(col):  # higher score = higher risk (check direction below)
    r = d[col].rank(method="first"); return (r <= k).astype(int)
# direction check: risk score should be positively associated with false feedback
for c in ["s_risk_lgbm", "s_gec_lr"]:
    assert d.groupby("false_feedback")[c].mean().loc[1] > d.groupby("false_feedback")[c].mean().loc[0], c
d["kept_risk"] = kept_by("s_risk_lgbm"); d["kept_gec"] = kept_by("s_gec_lr")
d["cell"] = d.false_feedback.map({1: "FF", 0: "OK"}) + "_" + d.kept_risk.map({1: "kept", 0: "withheld"})
pop = d.cell.value_counts().to_dict()
lines = {n: open(f"{E}/{n}.txt").read().split("\n") for n in ["flt_ctm", "gec_ctm", "flt_stm"]}
rng = np.random.default_rng(SEED)
samp = pd.concat([g.loc[rng.choice(g.index, PER, replace=False)] for _, g in d.groupby("cell")])
samp = samp.loc[rng.permutation(samp.index)].reset_index(drop=True)
samp["item_id"] = [f"A{i+1:03d}" for i in range(len(samp))]
flac = {os.path.basename(p)[:-5]: os.path.relpath(p, C) for p in glob.glob(f"{C}/data/flac/eval/*/*.flac")}
def src_cause(r):
    return "asr" if r.asr_induced else "dd" if r.dd_induced else "gec" if r.gec_induced else ("mixed" if r.mixed else "none")
samp["cause"] = samp.apply(src_cause, axis=1)
red = InlineFont(color="C00000", b=True)
def marked(tokens, a, b):
    pre, mid, post = " ".join(tokens[:a]), " ".join(tokens[a:b]) or "∅", " ".join(tokens[b:])
    return CellRichText([TextBlock(InlineFont(), pre + " " if pre else ""), TextBlock(red, "【" + mid + "】"),
                         TextBlock(InlineFont(), " " + post if post else "")])
mism = 0; rows = []
for _, r in samp.iterrows():
    S = lines["flt_ctm"][r.utt_idx].split(" "); T = lines["gec_ctm"][r.utt_idx].split(" ")
    mism += int(len(S) != r.n_src_tokens)
    rows.append(dict(item_id=r.item_id, said=lines["flt_stm"][r.utt_idx], inp=marked(S, r.o_start, r.o_end),
                     outp=marked(T, r.c_start, r.c_end),
                     edit=f'{" ".join(S[r.o_start:r.o_end]) or "∅"}  →  {" ".join(T[r.c_start:r.c_end]) or "∅"}',
                     audio=flac.get(r.utt_id, "")))
assert mism == 0, mism
def book(path, rws, title):
    wb = Workbook(); ws = wb.active; ws.title = "标注"
    hdr = ["编号", "说话人实际说的（人工转写，已去掉口误/重复）", "系统输入（ASR+DD 之后）", "系统修改后（红色为本条修改）",
           "本条修改", "判断：1 合理 / 0 不合理 / ? 无法判断", "备注（可选）", "音频（本地回听，可选）"]
    ws.append(hdr)
    for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDEBF7"); c.alignment = Alignment(wrap_text=True, vertical="top")
    for x in rws:
        ws.append([x["item_id"], x["said"], "", "", x["edit"], None, None, x["audio"]])
        rr = ws.max_row; ws.cell(rr, 3).value = x["inp"]; ws.cell(rr, 4).value = x["outp"]
        for c in ws[rr]: c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(rr, 6).fill = PatternFill("solid", fgColor="FFF2CC")
    for col, w in zip("ABCDEFGH", [8, 50, 50, 50, 22, 16, 24, 30]): ws.column_dimensions[col].width = w
    dv = DataValidation(type="list", formula1='"1,0,?"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"F2:F{ws.max_row}")
    ws.freeze_panes = "B2"
    g = wb.create_sheet("说明")
    for t in [title, "", "只看第 4 列红色【】里的这一处修改，判断：把它展示给学习者是否合理。",
              "1 = 合理：说话人确实犯了这个错，且改法正确（改法不同于你自己会写的也算合理，只要语法正确、符合原意）。",
              "0 = 不合理：说话人其实没说错（多见于识别错误：比较第 2 列与第 3 列）；或改错了、改了意思、没必要改。",
              "? = 无法判断：信息不足。尽量少用。",
              "第 2 列是人工转写，代表说话人真正说的话；第 3 列是系统听到的。两者不同的地方，常常就是系统误纠的原因。",
              "∅ 表示空：原句此处无词（插入），或改后此处无词（删除）。",
              "句子已按分词显示（如 do n't、it 's），这不是错误，不必理会。标点与大小写差异不计。",
              "一次只看一条，不要回头改前面的判断；可以分几次完成。",
              "本文件含语料原文：只在本机查看，不要上传或发送给任何人或任何在线工具。"]:
        g.append([t])
    g.column_dimensions["A"].width = 120
    wb.save(path)
book(f"{OUT}/annotation_sheet.xlsx", rows, "人工复核标注表（200 条）")
# re-annotation 50: stratified 12/13 per cell, reshuffled, new ids
rs = pd.concat([g.sample(n=13 if i < 2 else 12, random_state=SEED) for i, (_, g) in enumerate(samp.groupby("cell"))])
rs = rs.sample(frac=1, random_state=SEED + 1).reset_index(drop=True); rs["re_id"] = [f"R{i+1:02d}" for i in range(len(rs))]
byid = {x["item_id"]: x for x in rows}
book(f"{OUT}/reannotation_50.xlsx", [dict(byid[i], item_id=j) for i, j in zip(rs.item_id, rs.re_id)], "复标表（50 条，一周后盲标，不要翻看第一次的结果）")
key = samp[["item_id", "utt_id", "part", "edit_idx", "type", "op", "cell", "false_feedback", "kept_risk", "kept_gec",
            "cause", "s_risk_lgbm", "s_risk_lr", "s_gec_lr"]]
key.to_csv(f"{OUT}/key_DO_NOT_OPEN.csv", index=False)
rs[["re_id", "item_id"]].to_csv(f"{OUT}/key_reannotation_DO_NOT_OPEN.csv", index=False)
json.dump({"seed": SEED, "per_cell": PER, "N_eval_edits": N, "coverage_k_kept": int(k), "population_per_cell": pop,
           "sample_cause_counts": key.cause.value_counts().to_dict(), "sample_part_counts": key.part.value_counts().to_dict()},
          open(f"{OUT}/sampling.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"pop": pop, "cause": key.cause.value_counts().to_dict(), "part": key.part.value_counts().to_dict(),
                  "audio_found": int((pd.Series([x["audio"] for x in rows]) != "").sum()), "token_mismatch": mism}, ensure_ascii=False))
