import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import time
t0=time.time();dp=RUN/'2026-11-10_C1_features/features_dev.csv';du=dp.parent/'utt_counts_dev.csv';bd=pd.read_csv(dp);bu=pd.read_csv(du);bc,_=qian(bd);bthr={k:select_threshold(c,bd,bu)for k,c in bc.items()};save('cross_qian_B1_thresholds.json',bthr)
for system in ['W','Q']:
 dvp=RUN/f'2026-11-10_D_{system}_dev/features/features_dev.csv';up=dvp.parent/'utt_counts_dev.csv';dev=pd.read_csv(dvp);ud=pd.read_csv(up);cv,vdev=qian(dev);own={k:select_threshold(c,dev,ud)for k,c in cv.items()}
 save(f'cross_qian_{system}_threshold_freeze.json',{'B1_dev':bthr,'own_dev':own,'selection_uses_test_labels':False,'input_hashes':{str(p):sha(p)for p in [dvp,up]}})
 ep=RUN/f'2026-11-10_D_{system}_eval/features/features_eval.csv';eu=ep.parent/'utt_counts_eval.csv';d=pd.read_csv(ep);u=pd.read_csv(eu);cq,vt=qian(d);S={};checks={};refdata={}
 for setting in (['frozen','retrained']if system=='W'else['retrained']):
  root=RUN/'2026-11-10_C6_final'if setting=='frozen'else RUN/f'2026-11-10_D_{system}_model/05_runs/2026-11-10_C6_final'
  ev=json.load(open(R/f'06_results/D_{system}_eval_{setting}.json'));scores,art=aligned_scores(d,root,ev);refdata[setting]=ev
  for k,s in scores.items():
   name=setting+'_'+k;S[name]=s;pt=C.metrics(d.false_feedback.to_numpy(),s,int(u.n_ref_edits.sum()));checks[name]={'artifact_hashes_match':all(art.values()),'point_metric_maxdiff':max(abs(pt[m]-ev['point'][k][m])for m in MET)}
   assert checks[name]['point_metric_maxdiff']<1e-12,checks
 for k,c in cq.items():S[k]=1-c
 pairs=[]
 for k in S:
  if k.endswith('risk_lgbm')or k.endswith('gec_lr'):
   for rule in cq:pairs.append((k,rule))
 ops={}
 for scope,thresholds in [('B1_transfer',bthr),('own_dev',own)]:
  for k,c in cq.items():ops[scope+'_'+k]={'confidence':c,'threshold':thresholds[k]['threshold']}
 out=boot(d,u,S,pairs,ops=ops);out['operating_points']={k:{'threshold':v['threshold'],**op(d.false_feedback.to_numpy(),v['confidence'],int(u.n_ref_edits.sum()),v['threshold'])}for k,v in ops.items()}
 out.update({'system':system,'status':'completed Qian-style adaptation; same geometric/token-min proxy as B1, not original end-to-end reproduction','structural_empty_validation':{'dev':vdev,'test':vt},'B1_dev_thresholds':bthr,'own_dev_thresholds':own,'frozen_model_replication':checks,'input_sha256':{str(p):sha(p)for p in [dvp,up,ep,eu]},'script_sha256':sha(__file__),'seconds':time.time()-t0})
 pd.DataFrame({'utt_id':d.utt_id,'edit_idx':d.edit_idx,**S}).to_csv(OUT/f'results/cross_qian_{system}_test_scores.csv',index=False)
 save(f'cross_qian_{system}_results.json',out);print(json.dumps({k:out[k]for k in ['system','status','n_edits','n_ref','valid_resamples','point','ci95','paired','operating_points']},indent=2),flush=True)
