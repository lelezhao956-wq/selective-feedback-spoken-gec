"""Recheck native batching disagreements in the exact historical single-phrase runtime.
Preserves raw native posterior output separately and never changes archived decisions.
"""
from pathlib import Path
import json,torch,pandas as pd,numpy as np,hashlib,shutil,time
from transformers import BertTokenizerFast
O=Path(__file__).resolve().parents[1];COR=Path('~/Desktop/sandi-corpus-2025-2');torch.set_num_threads(4)
m=torch.load(str(COR/'sandi-models/dd/dd_model'),map_location='cpu');assert not m.training;m.eval();tok=BertTokenizerFast.from_pretrained('bert-base-uncased',is_pretokenized=True,local_files_only=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for ds in ['dev','eval']:
 f=O/f'results/dd_posteriors_{ds}.jsonl';meta=json.load(open(O/f'results/dd_posteriors_{ds}.metadata.json'));shutil.copyfile(f,O/f'results/dd_posteriors_{ds}.native.jsonl');shutil.copyfile(O/f'results/dd_posteriors_{ds}.metadata.json',O/f'results/dd_posteriors_{ds}.native.metadata.json')
 records=[json.loads(s)for s in open(f)];D={z['phrase_id']:z for z in records};pre=pd.read_csv(COR/f'stageB/B1_{ds}/gec/norm/gec.ctm',sep=r'\s+',header=None,keep_default_na=False,engine='python');flt=pd.read_csv(COR/f'stageB/B1_{ds}/gec/fluent.ctm',sep=r'\s+',header=None,keep_default_na=False,engine='python');fg={str(pid):g for pid,g in flt.groupby(0)};wanted={};bad=[]
 for pid,g in pre.groupby(0):
  pid=str(pid);n=len(g);keep=np.zeros(n,dtype=bool);j=0
  if pid in fg:
   for row in fg[pid].itertuples(index=False,name=None):
    while j<n and not(str(g.iloc[j,4])==str(row[4])and abs(float(g.iloc[j,2])-float(row[2]))<1e-6):j+=1
    assert j<n;keep[j]=True;j+=1
  wanted[pid]=keep
  if not np.array_equal(np.array(D[pid]['predicted_class'])==0,keep):bad.append((pid,g))
 rec=[]
 for pid,g in bad:
  s=' '.join(map(str,g[4]));enc=tok(s,return_offsets_mapping=True,add_special_tokens=False);off=enc['offset_mapping'];ids=enc['input_ids'];labs=[];probs=[]
  for st in range(0,len(ids),126):
   a=off[st:st+126];chunk=s[a[0][0]:a[-1][1]];x=tok(chunk.split(),is_pretokenized=True,return_offsets_mapping=True,padding='max_length',truncation=True,max_length=128,return_tensors='pt')
   with torch.no_grad():lg=m(x['input_ids'],attention_mask=x['attention_mask'])[0];pr=torch.softmax(lg,dim=-1).squeeze(0).numpy();cl=lg.argmax(-1).squeeze(0).numpy()
   offsets=x['offset_mapping'].squeeze(0).numpy();mask=(offsets[:,0]==0)&(offsets[:,1]!=0);labs.extend(cl[mask].tolist());probs.extend(pr[mask].tolist())
  labs=labs[:len(g)];probs=probs[:len(g)];native=D[pid];left=int(np.count_nonzero((np.array(labs)==0)!=wanted[pid]));dist=float(np.max(np.abs(np.array(probs)-np.array(native['class_posterior']))));rec.append({'phrase_id':pid,'native_mismatches':int(np.count_nonzero((np.array(native['predicted_class'])==0)!=wanted[pid])),'historical_runtime_mismatches':left,'max_posterior_diff':dist});D[pid]={'phrase_id':pid,'n_words':len(g),'predicted_class':labs,'class_posterior':probs}
 mism=sum(int(np.count_nonzero((np.array(D[pid]['predicted_class'])==0)!=mask))for pid,mask in wanted.items())
 with f.open('w')as out:
  for z in records:out.write(json.dumps(D[z['phrase_id']])+'\n')
 meta.update({'native_batch_disagreements':meta['counts']['kept_decision_mismatches'],'historical_runtime_rechecks':rec,'remaining_archived_decision_mismatches':mism,'upstream_predictions_recomputed_match_archived_fluent':mism==0,'output_sha256':sha(f),'reconciliation_script_sha256':sha(__file__)})
 (O/f'results/dd_posteriors_{ds}.metadata.json').write_text(json.dumps(meta,indent=2));print(ds,json.dumps(meta),flush=True)
