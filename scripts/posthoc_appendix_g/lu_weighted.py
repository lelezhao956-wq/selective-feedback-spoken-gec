import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import time
t0=time.time();d=pd.read_csv(OUT/'results/lu_inputs_dev.csv');u=pd.read_csv(RUN/'2026-11-10_C1_features/utt_counts_dev.csv');nr=int(u.n_ref_edits.sum());y=d.false_feedback.to_numpy();configs={};Sdev={};grid=[]
def logcomponents(d,variant):return np.log(np.clip(d[['ASR_sentence','DD_'+variant,'GEC_'+variant]].to_numpy(),1e-12,1))
def sweep(s,y,nref):
 # confidence here is weighted log-confidence, using documented >= t - 1e-9.
 order=np.argsort(s,kind='stable');ss=s[order];tp=np.r_[0,np.cumsum((1-y)[order])];cuts=np.unique(ss);ix=np.searchsorted(ss,cuts-1e-9,side='left');n=len(s)-ix;tps=tp[-1]-tp[ix];p=tps/n;r=tps/nref;f=1.25*p*r/(.25*p+r+1e-300);best=np.flatnonzero(f==f.max())[0]
 return {'threshold_log_confidence':float(cuts[best]),'F05':float(f[best]),'P':float(p[best]),'R':float(r[best]),'coverage':float(n[best]/len(s))}
for variant in ['edit','sentence']:
 X=logcomponents(d,variant);cand=[]
 for a in range(11):
  for b in range(11-a):
   w=np.array([a,b,10-a-b])/10;s=X@w;r=sweep(s,y,nr);cand.append({'weights':w.tolist(),**r})
 best=sorted(cand,key=lambda z:(-round(z['F05'],14),-z['coverage'],z['weights']))[0];configs[variant]=best;grid.append({'variant':variant,'candidates':cand})
fixed=logcomponents(d,'sentence')@np.array([.3,.4,.3]);configs['sentence_published']={'weights':[.3,.4,.3],**sweep(fixed,y,nr),'weights_source':'Lu2022 Figure4; only threshold selected on B1 dev'}
save('lu_weight_freeze.json',{'dev_only_selection':True,'configs':configs,'grid':grid,'selection_metric':'SGECF F0.5; simplex step0.1; distinct log-confidence cutoffs; ties greater coverage then lexicographic weights','original_published_sentence_weights':[.3,.4,.3],'original_OTHER_exclusion_not_applied':'all original B1 candidates retained for matched comparison','empty_edit_side':1.0,'confidence_floor':1e-12,'fit_or_selection_on_test_labels':False});print('L U FROZEN BEFORE TEST',json.dumps(configs),flush=True)
e=pd.read_csv(OUT/'results/lu_inputs_eval.csv');feature=pd.read_csv(RUN/'2026-11-10_C7_B1_eval/features/features_eval.csv');assert np.array_equal(e[['utt_id','edit_idx']].values,feature[['utt_id','edit_idx']].values);u=pd.read_csv(RUN/'2026-11-10_C7_B1_eval/features/utt_counts_eval.csv');scores=pd.read_csv(RUN/'2026-11-10_C7_B1_eval/eval_scores_frozen.csv');m=e[['utt_id','edit_idx']].merge(scores,on=['utt_id','edit_idx'],validate='1:1');assert len(m)==len(e)
S={k:m['s_'+k].to_numpy()for k in ['risk_lgbm','gec_lr']};ops={}
for v,c in configs.items():
 lc=logcomponents(e,'sentence'if v=='sentence_published'else v)@np.array(c['weights']);name='lu_'+v;S[name]=-lc;ops[name]={'confidence':lc,'threshold':c['threshold_log_confidence']}
out=boot(e,u,S,[(a,b)for a in ['risk_lgbm','gec_lr']for b in ['lu_edit','lu_sentence','lu_sentence_published']],ops=ops);N=int(u.n_ref_edits.sum());out['operating_points']={k:{'threshold_log_confidence':v['threshold'],**op(e.false_feedback.to_numpy(),v['confidence'],N,v['threshold'])}for k,v in ops.items()};out.update({'status':'completed adapted Lu weighted rule with newly scored true DD posteriors; not a faithful reproduction of original LIN system or OTHER-excluded scoring','configs':configs,'input_validation':json.load(open(OUT/'results/lu_input_validation.json')),'script_sha256':sha(__file__),'seconds':time.time()-t0})
pd.DataFrame({'utt_id':e.utt_id,'edit_idx':e.edit_idx,**S}).to_csv(OUT/'results/lu_weighted_test_scores.csv',index=False);save('lu_weighted_results.json',out);print(json.dumps({k:out[k]for k in ['status','configs','n_edits','valid_resamples','point','ci95','paired','operating_points']},indent=2),flush=True)
