#!/usr/bin/env python3
"""Stage D-GEC: align the LLM GEC output with the DD output (timestamps) and write gec.ctm exactly as the official
baseline-pipeline/gec-tools/gec-plus-align.py does after its BART step (same align_dataset() and the same CTM-writing
code, imported/copied from the official script, unchanged). The official 126-token sentence splitting is not needed (the LLM
corrects the whole phrase), so the phrase-level rows go straight to alignment.
Run in the sandi-all env from the corpus root:
  python d_llm_align.py --pre_gec stageD/Q_dev/gec/dd/pre-gec.tsv --llm stageD/Q_dev/llm/llm_gec.tsv --gec_ctm stageD/Q_dev/gec/gec.ctm
"""
import argparse, importlib.util, json, os
from ast import literal_eval
import pandas as pd

spec = importlib.util.spec_from_file_location("gpa", os.path.abspath("baseline-pipeline/gec-tools/gec-plus-align.py"))
import sys
sys.path.insert(0, os.path.abspath("baseline-pipeline/gec-tools"))
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)   # official module (main() is not called)
ap = argparse.ArgumentParser()
ap.add_argument("--pre_gec", required=True); ap.add_argument("--llm", required=True); ap.add_argument("--gec_ctm", required=True)
a = ap.parse_args()
df = pd.read_csv(a.pre_gec, sep="\t", keep_default_na=False)
df["data_flt"] = df["data_flt"].apply(literal_eval)
q = pd.read_csv(a.llm, sep="\t", keep_default_na=False)
assert list(q["ids"]) == list(df["ids"]), "phrase order/ids differ"
df["sentence_gec"] = q["sentence_gec"].values
# --- from here identical to the official main() after GEC ---
df = G.align_dataset(df)
df['data_gec'] = df['data_gec'].apply(lambda x: [t[1:] for t in x])
gec_tuple = pd.DataFrame(df['data_gec'].explode())
pre_ctm = pd.merge(left=df[['ids']], right=gec_tuple, right_index=True, left_index=True).reset_index(drop=True)
ctm = pd.concat([pre_ctm[['ids']], pd.DataFrame(pre_ctm['data_gec'].to_list(), columns=['start', 'duration', 'prob', 'word'])], axis=1)
ctm['channel'] = 1
ctm = ctm[['ids', 'channel', 'start', 'duration', 'word', 'prob']]
ctm = ctm.loc[ctm['word'].notna()].reset_index(drop=True)
ctm['prob'] = ctm['prob'].fillna(0.5)
ctm.to_csv(a.gec_ctm, sep='\t', header=None, index=None)
changed = int((df["sentence_gec"] != df["sentence_flt"]).sum())
print(json.dumps(dict(phrases=len(df), phrases_changed=changed, gec_ctm_words=len(ctm))))
