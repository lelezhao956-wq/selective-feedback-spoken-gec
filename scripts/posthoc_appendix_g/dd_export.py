"""New DD-only posterior scoring from archived checkpoint; no upstream output is modified.
Uses original chunk/first-wordpiece decision conventions and checks kept words against archived fluent CTM.
Original dd.py discarded its probability list (and applied softmax on sequence dim); this export uses class dim=-1.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd,torch
from transformers import BertTokenizerFast
O=Path(__file__).resolve().parents[1];COR=Path('~/Desktop/sandi-corpus-2025-2');MODEL=COR/'sandi-models/dd/dd_model'
torch.set_num_threads(4);t0=time.time();model=torch.load(str(MODEL),map_location='cpu');assert not model.training;model.eval();tok=BertTokenizerFast.from_pretrained('bert-base-uncased',is_pretokenized=True,local_files_only=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();allmeta={}
for ds in ['dev','eval']:
 prepath=COR/f'stageB/B1_{ds}/gec/norm/gec.ctm';fltpath=COR/f'stageB/B1_{ds}/gec/fluent.ctm';pre=pd.read_csv(prepath,sep=r'\s+',header=None,keep_default_na=False,engine='python');flt=pd.read_csv(fltpath,sep=r'\s+',header=None,keep_default_na=False,engine='python')
 groups={str(pid):g for pid,g in pre.groupby(0)};expected={str(pid):g for pid,g in flt.groupby(0)};chunks=[];preds={};probs={};counts={'phrases':len(groups),'chunks':0,'words':len(pre),'chunk_label_count_mismatch':0,'kept_decision_mismatches':0,'unassigned_words':0}
 for pid,g in groups.items():
  sentence=' '.join(map(str,g[4]));enc=tok(sentence,return_offsets_mapping=True,add_special_tokens=False);offset=enc['offset_mapping'];ids=enc['input_ids'];preds[pid]=[];probs[pid]=[]
  for i in range(0,len(ids),126):
   span=offset[i:i+126];chunks.append((pid,sentence[span[0][0]:span[-1][1]]))
 counts['chunks']=len(chunks)
 for start in range(0,len(chunks),24):
  batch=chunks[start:start+24];x=tok([s.split()for _,s in batch],is_pretokenized=True,return_offsets_mapping=True,padding='max_length',truncation=True,max_length=128,return_tensors='pt')
  with torch.no_grad():lg=model(x['input_ids'],attention_mask=x['attention_mask'])[0];pr=torch.softmax(lg,dim=-1);lab=torch.argmax(lg,dim=-1)
  pp=pr.cpu().numpy();ll=lab.cpu().numpy();off=x['offset_mapping'].numpy()
  for j,(pid,s)in enumerate(batch):
   mask=(off[j,:,0]==0)&(off[j,:,1]!=0);pv=pp[j,mask];lv=ll[j,mask];preds[pid].extend(lv.tolist());probs[pid].extend(pv.tolist())
  if start%240==0:print(ds,'chunks',min(start+24,len(chunks)),'/',len(chunks),'seconds',round(time.time()-t0,1),flush=True)
 out=O/f'results/dd_posteriors_{ds}.jsonl'
 with out.open('w')as f:
  for pid,g in groups.items():
   n=len(g);lab=preds[pid][:n];pp=probs[pid][:n];counts['chunk_label_count_mismatch']+=int(len(preds[pid])!=n);counts['unassigned_words']+=max(0,n-len(lab));want=np.zeros(n,dtype=bool);j=0
   if pid in expected:
    for row in expected[pid].itertuples(index=False,name=None):
     while j<n and not(str(g.iloc[j,4])==str(row[4])and abs(float(g.iloc[j,2])-float(row[2]))<1e-6):j+=1
     assert j<n,(ds,pid,'fluent subsequence failed');want[j]=True;j+=1
   have=np.zeros(n,dtype=bool);have[:len(lab)]=np.array(lab)==0;counts['kept_decision_mismatches']+=int(np.count_nonzero(have!=want))
   f.write(json.dumps({'phrase_id':pid,'n_words':n,'predicted_class':lab,'class_posterior':pp})+'\n')
 meta={'dataset':ds,'new_dd_scoring_only':True,'counts':counts,'seconds':time.time()-t0,'model_sha256':sha(MODEL),'input_sha256':{str(p):sha(p)for p in [prepath,fltpath]},'posterior_axis':'classes, dim=-1; original discarded probability list is not reused','upstream_predictions_recomputed_match_archived_fluent':counts['kept_decision_mismatches']==0,'output_sha256':sha(out),'script_sha256':sha(__file__)}
 allmeta[ds]=meta;(O/f'results/dd_posteriors_{ds}.metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
(O/'results/dd_export_summary.json').write_text(json.dumps(allmeta,indent=2))
