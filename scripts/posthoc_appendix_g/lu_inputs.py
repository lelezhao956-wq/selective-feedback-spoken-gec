"""Align new true DD posteriors to existing B1 edits; no transcript content is exported."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parents[1];R=Path('~/Desktop/11.9剑桥');COR=Path('~/Desktop/sandi-corpus-2025-2')
sys.path.insert(0,str(COR/'scoring-tools/sgecf-tools'));import create_files_for_errant as CF
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readctm(path):
 d=pd.read_csv(path,sep=r'\s+',header=None,keep_default_na=False,engine='python');d['Base_ID']=d[0].str.rsplit('_',n=1).str[0];d['widx']=d.groupby(0).cumcount();return d
def filtered(d,stm):
 ign=CF.load_stm(str(stm));ign=ign[ign.Transcription=='IGNORE_TIME_SEGMENT_IN_SCORING'];iw=CF.get_non_scorable_words(ign,d);return d.loc[~d.index.isin(iw.index)]
summary={}
for ds in ['dev','eval']:
 rd=COR/f'stageB/B1_{ds}/gec';meta=json.load(open(O/f'results/dd_posteriors_{ds}.metadata.json'));assert meta['upstream_predictions_recomputed_match_archived_fluent'],meta
 pp={x['phrase_id']:x for x in map(json.loads,open(O/f'results/dd_posteriors_{ds}.jsonl'))};pre=readctm(rd/'norm/gec.ctm');fltall=readctm(rd/'fluent.ctm');gall=readctm(rd/'gec.ctm');stm=COR/f'reference-materials/stms/{ds}-fluent.stm';flt=filtered(fltall,stm);prec=filtered(pre,stm);gec=filtered(gall,COR/f'reference-materials/stms/{ds}-gec.stm')
 pb={pid:g for pid,g in pre.groupby(0)};aidx={};keepidx={pid:set()for pid in pb}
 for pid,g in fltall.groupby(0):
  raw=pb[pid];selected=np.flatnonzero(np.array(pp[str(pid)]['predicted_class'])==0);assert len(selected)==len(g)
  for j,row in zip(selected,g.itertuples(index=False,name=None)):
   assert str(raw.iloc[j,4])==str(row[4]) and np.allclose(raw.iloc[j,[2,3,5]].to_numpy(dtype=float),np.array([row[2],row[3],row[5]],dtype=float),rtol=0,atol=1e-12)
   aidx[(pid,int(row[-1]))]=int(j);keepidx[pid].add(int(j))
 # Actual archived decision confidence: kept=class0, deleted=class1.
 def ddconf(pid,i):
  z=pp[str(pid)];assert i<len(z['class_posterior']);lab=0 if i in keepidx[pid]else 1;return float(z['class_posterior'][i][lab])
 ld=R/f'05_runs/2026-11-10_B1_dev/labels_v4'if ds=='dev'else R/'05_runs/2026-11-10_C7_B1_eval/labels_v4'
 fp=R/'05_runs/2026-11-10_C1_features/features_dev.csv'if ds=='dev'else R/'05_runs/2026-11-10_C7_B1_eval/features/features_eval.csv'
 bp=R/'05_runs/2026-11-10_C1_features/bart_scores_dev.jsonl'if ds=='dev'else R/'05_runs/2026-11-10_C7_B1_eval/features/bart_scores_eval.jsonl';bart={z['id']:z for z in map(json.loads,open(bp))}
 lab=pd.read_csv(ld/f'hyp_edit_labels_{ds}.csv');feat=pd.read_csv(fp);ids=pd.read_csv(ld/f'utt_index_{ds}.csv').utt_id.tolist();off=pd.read_csv(rd/'sgecf-scoring/errant/flt_ctm.txt',sep='\t',header=None,keep_default_na=False)[0].tolist();fu={k:g for k,g in flt.groupby('Base_ID')};pu={k:g for k,g in prec.groupby('Base_ID')};gu={k:g for k,g in gec.groupby('Base_ID')};uc={};bad=0
 for i,uid in enumerate(ids):
  if uid not in fu:continue
  fr=fu[uid];words=list(map(str,fr[4]));s=' '.join(words);doc=CF.nlp.make_doc(s);st=' '.join(x.text for x in doc);assert st==off[i],(ds,uid,'scorer token sequence mismatch')
  starts=np.cumsum([0]+[len(w)+1 for w in words[:-1]]);t2w=np.searchsorted(starts,[t.idx for t in doc],side='right')-1;raw=pu[uid]
  asr=float(raw[5].min());ddsent=min(ddconf(row[0],int(row[-1]))for row in raw.itertuples(index=False,name=None))
  gvals=[bart[row[0]]['t_min'][int(row[-1])]for row in gu[uid].itertuples(index=False,name=None)]if uid in gu else[];gsent=float(np.exp(min(gvals)))if gvals else 1.0
  uc[uid]={'fr':fr,'t2w':t2w,'ASR':asr,'DD_sentence':ddsent,'GEC_sentence':gsent}
 rows=[];empty=0
 for e in lab.itertuples(index=False):
  z=uc[e.utt_id];fr=z['fr'];ws=sorted(set(z['t2w'][e.o_start:e.o_end]));vals=[ddconf(fr.iloc[w,0],aidx[(fr.iloc[w,0],int(fr.iloc[w]['widx']))])for w in ws];ddedit=min(vals)if vals else 1.0;empty+=int(not vals)
  rows.append({'utt_id':e.utt_id,'edit_idx':e.edit_idx,'ASR_sentence':z['ASR'],'DD_edit':ddedit,'DD_sentence':z['DD_sentence'],'GEC_sentence':z['GEC_sentence']})
 out=feat[['utt_id','edit_idx','group','false_feedback','op','gec_lp_c_min']].merge(pd.DataFrame(rows),on=['utt_id','edit_idx'],validate='1:1');assert len(out)==len(feat)and np.array_equal(out.false_feedback,feat.false_feedback)
 out['GEC_edit']=np.exp(out.gec_lp_c_min);out.loc[out.op=='U','GEC_edit']=1.0;out=out.drop(columns=['gec_lp_c_min']);assert np.isfinite(out[['ASR_sentence','DD_edit','DD_sentence','GEC_edit','GEC_sentence']].to_numpy()).all()
 out.to_csv(O/f'results/lu_inputs_{ds}.csv',index=False);summary[ds]={'n_edits':len(out),'dd_empty_source_defaults_1':empty,'scorer_alignment_all_matched':True,'true_DD_class_posteriors':True,'DD_archived_kept_decisions_match':True,'minimum_ASR_over_scored_preDD_not_postDD':True,'input_sha256':{str(p):sha(p)for p in [fp,bp,ld/f'hyp_edit_labels_{ds}.csv',rd/'norm/gec.ctm',rd/'fluent.ctm']}};print(ds,json.dumps(summary[ds]),flush=True)
(O/'results/lu_input_validation.json').write_text(json.dumps(summary,indent=2))
