"""Resolve greedy word/start pointer ambiguities by checking complete kept CTM tuples in order."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
O=Path(__file__).resolve().parents[1];COR=Path('~/Desktop/sandi-corpus-2025-2')
for ds in ['dev','eval']:
 p=pd.read_csv(COR/f'stageB/B1_{ds}/gec/norm/gec.ctm',sep=r'\s+',header=None,keep_default_na=False,engine='python');f=pd.read_csv(COR/f'stageB/B1_{ds}/gec/fluent.ctm',sep=r'\s+',header=None,keep_default_na=False,engine='python');D={x['phrase_id']:x for x in map(json.loads,open(O/f'results/dd_posteriors_{ds}.jsonl'))};fg={str(pid):g for pid,g in f.groupby(0)};bad=[];amb=[];nw=0
 for pid,g in p.groupby(0):
  pid=str(pid);sel=g.loc[np.array(D[pid]['predicted_class'])==0].reset_index(drop=True);old=fg.get(pid,f.iloc[:0]).reset_index(drop=True);nw+=len(sel)
  same=len(sel)==len(old)and np.array_equal(sel[[0,1,4]].values,old[[0,1,4]].values)and np.allclose(sel[[2,3,5]].to_numpy(dtype=float),old[[2,3,5]].to_numpy(dtype=float),rtol=0,atol=1e-12)
  if not same:bad.append(pid)
  if g.duplicated([2,4],keep=False).any():amb.append(pid)
 assert not bad,(ds,bad)
 meta=json.load(open(O/f'results/dd_posteriors_{ds}.metadata.json'));meta.update({'full_kept_ctm_tuple_verification':{'all_phrases_exact':True,'rows':nw,'bad_phrases':0,'compared_fields':'phrase_id/channel/word/start/duration/incoming_ASR_confidence in original within-phrase order','phrases_with_ambiguous_word_start_keys':len(amb)},'upstream_predictions_recomputed_match_archived_fluent':True,'matching_note':'Earlier 2-indicator disagreements per split came from greedy word/start back-pointers. Reconstructing full kept CTM tuples reproduces archived outputs exactly; no archived decision or candidate set changed. Posterior mapping uses reconstructed kept indices, not the ambiguous pointer.','ctm_validation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()});(O/f'results/dd_posteriors_{ds}.metadata.json').write_text(json.dumps(meta,indent=2));print(ds,json.dumps(meta['full_kept_ctm_tuple_verification']),flush=True)
