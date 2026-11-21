"""F1 round 2 for annotator 2: 50 NEW items (not in annotator2_50), stratified 12/13 per cell, seed 20261113,
blank labels, clarified instructions. Copies rows from annotation_sheet.xlsx (labels removed). Prints counts only."""
import os, pandas as pd, openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
V = os.path.expanduser("$EXP_DIR/11_human_validation")
if not os.path.isdir(V): V = os.path.expanduser("~/mnt/EXP_DIR/11_human_validation")
k = pd.read_csv(f"{V}/key_DO_NOT_OPEN.csv"); used = set(pd.read_csv(f"{V}/key_reannotation_DO_NOT_OPEN.csv").item_id)
rest = k[~k.item_id.isin(used)]
pick = pd.concat([g.sample(n=13 if i < 2 else 12, random_state=20261113) for i, (_, g) in enumerate(rest.groupby("cell"))])
pick = pick.sample(frac=1, random_state=20261114).reset_index(drop=True); pick["s_id"] = [f"S{i+1:02d}" for i in range(len(pick))]
src = openpyxl.load_workbook(f"{V}/annotation_sheet.xlsx", rich_text=True)["标注"]
row_of = {src.cell(r, 1).value: r for r in range(2, src.max_row + 1)}
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "标注"
for c in range(1, 9): ws.cell(1, c).value = src.cell(1, c).value
for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDEBF7"); c.alignment = Alignment(wrap_text=True, vertical="top")
for i, (iid, sid) in enumerate(zip(pick.item_id, pick.s_id), start=2):
    r = row_of[iid]
    for c in range(1, 9): ws.cell(i, c).value = src.cell(r, c).value
    ws.cell(i, 1).value = sid; ws.cell(i, 6).value = None; ws.cell(i, 7).value = None
    for c in ws[i]: c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.cell(i, 6).fill = PatternFill("solid", fgColor="FFF2CC")
for col, w in zip("ABCDEFGH", [8, 50, 50, 50, 22, 16, 24, 30]): ws.column_dimensions[col].width = w
dv = DataValidation(type="list", formula1='"1,0,?"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"F2:F{ws.max_row}")
ws.freeze_panes = "B2"
g = wb.create_sheet("说明")
for t in ["第二轮标注表（50 条，均为新条目）", "",
          "判断标准：这条修改【针对学习者实际说出的话】是否合理、值得展示给学习者。",
          "每条请按两步判断：",
          "第一步：对照第 2 列（人工转写＝学习者实际说的话）和第 3 列（系统听到的话），看红色【】修改处附近两者是否一致。",
          "  · 如果不一致（系统听错了），而这条修改是在纠正系统听错的词、学习者其实并没有说错 → 0。",
          "  · 例（自编）：学习者说 she likes it，系统听成 she like it，修改为 like → likes。修改本身语法正确，但学习者没有犯这个错 → 0。",
          "第二步：如果修改处附近两者一致，再判断修改本身：",
          "  · 学习者确实错了，改法正确（即使和你自己的改法不同） → 1；",
          "  · 改错了、改变了原意、或原句本来就可以接受 → 0。",
          "? = 无法判断，尽量少用。标点、大小写和分词形式（如 do n't）不计。",
          "独立完成：不要与他人讨论，不要查看其他标注表或答案文件。",
          "本文件含语料原文：只在本机查看，不要上传、发送或输入任何在线工具（包括 AI 聊天工具）。"]:
    g.append([t])
g.column_dimensions["A"].width = 120
wb.save(f"{V}/annotator2_round2_50.xlsx")
pick[["s_id", "item_id"]].to_csv(f"{V}/key_round2_DO_NOT_OPEN.csv", index=False)
print({"n": len(pick), "cells": pick.cell.value_counts().to_dict(), "cause": pick.cause.value_counts().to_dict(), "overlap_with_round1": int(pick.item_id.isin(used).sum())})
