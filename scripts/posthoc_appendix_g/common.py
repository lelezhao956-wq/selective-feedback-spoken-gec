from pathlib import Path
import os,sys,json,hashlib
import numpy as np,pandas as pd
R=Path('~/Desktop/11.9剑桥');RUN=R/'05_runs';OUT=Path(__file__).resolve().parents[1];SEED=20261110
os.environ['EXP_DIR']=str(R);sys.path.insert(0,str(R/'04_scripts'));import c_model as C
assert C.SEED==SEED
MET=['AUROC','AURC','P@0.5']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pc(x):return [float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]
def aligned_scores(d,root,expected):
 import lightgbm as lgb,pickle
 root=Path(root);fit=json.load(open(root/'final_fit.json'));art={f:sha(root/f)==v for f,v in fit['artifact_sha256'].items()};assert all(art.values()),art
 cols=fit['feature_columns'];x=d[cols].copy()
 for c,lev in fit['models']['risk_lgbm']['categories'].items():x[c]=x[c].astype(str).astype(pd.CategoricalDtype(lev))
 scores={'risk_lgbm':lgb.Booster(model_file=str(root/'risk_lgbm.txt')).predict(x,num_threads=1),'gec_lr':pickle.load(open(root/'gec_lr.pkl','rb')).predict_proba(d[C.GROUPS['GEC']])[:,1]}
 for k,s in scores.items():
  # Point comparison validated later using correct n_ref, not a new model fit.
  assert len(s)==len(d) and np.isfinite(s).all()
 return scores,art
def boot(d,u,scores,pairs,nboot=1000,ops=None):
 y=d.false_feedback.to_numpy();ng=u.groupby('group').n_ref_edits.sum();N=int(ng.sum());groups=np.array(sorted(set(ng.index)|set(d.group)));gi={g:np.flatnonzero(d.group.to_numpy()==g)for g in groups};point={k:C.metrics(y,v,N)for k,v in scores.items()};B={k:{m:[]for m in MET}for k in scores};BO={}
 if ops:BO={k:{m:[]for m in ['coverage','P','R','F05']}for k in ops}
 rng=np.random.RandomState(SEED);invalid=0
 for j in range(nboot):
  gs=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([gi[g]for g in gs]);nr=int(sum(ng.get(g,0)for g in gs));yb=y[ix]
  if len(np.unique(yb))<2:invalid+=1;continue
  for k,s in scores.items():
   z=C.metrics(yb,s[ix],nr)
   for m in MET:B[k][m].append(z[m])
  if ops:
   for k,o in ops.items():
    z=op(yb,o['confidence'][ix],nr,o['threshold'])
    for m in BO[k]:BO[k][m].append(z[m])
 ci={k:{m:pc(a)for m,a in v.items()}for k,v in B.items()};tests={}
 for a,b in pairs:
  z={}
  for m in MET:
   diff=np.array(B[a][m])-np.array(B[b][m]);z[m]={'diff':point[a][m]-point[b][m],'ci95':pc(diff),'two_sided_p_descriptive':float(min(1,2*min(np.mean(diff<=0),np.mean(diff>=0))))}
  tests[a+'_minus_'+b]=z
 return {'seed':SEED,'n_boot_requested':nboot,'valid_resamples':nboot-invalid,'invalid_resamples':invalid,'n_groups_including_zero_edit':len(groups),'n_edits':len(d),'n_ref':N,'point':point,'ci95':ci,'paired':tests,'operating_ci95':{k:{m:pc(v)for m,v in z.items()}for k,z in BO.items()},'bootstrap_note':'Paired test-response resampling; percentile CI; original c_model.metrics and deterministic tie permutation; descriptive post hoc comparisons outside original Holm family'}
def qian(d):
 a=d.asr_conf_o_mean.to_numpy().copy();b=d.asr_conf_o_min.to_numpy().copy();c=np.exp(d.gec_lp_c_mean.to_numpy());e=np.exp(d.gec_lp_c_min.to_numpy());M=d.op.eq('M').to_numpy();U=d.op.eq('U').to_numpy()
 assert (np.isnan(a)==M).all()and(np.isnan(b)==M).all()and(np.isnan(c)==U).all()and(np.isnan(e)==U).all()
 a[M]=1;b[M]=1;c[U]=1;e[U]=1
 z={'edit_avg':(a+c)/2,'edit_min':np.minimum(b,e)}
 assert all(np.isfinite(v).all()for v in z.values())
 return z,{'insertions_empty_fluent_side':int(M.sum()),'deletions_empty_corrected_side':int(U.sum()),'other_missing':0}
def op(y,c,nref,t):
 keep=c>=t-1e-9;n=int(keep.sum());tp=int(((1-y)*keep).sum());p=tp/n if n else float('nan');rr=tp/nref
 return {'coverage':n/len(y),'n_shown':n,'n_matched_shown':tp,'P':p,'R':rr,'F05':1.25*p*rr/(.25*p+rr)if n and p+rr else 0.0}
def select_threshold(conf,d,u):
 n=int(u.n_ref_edits.sum());y=d.false_feedback.to_numpy();rows=[{'threshold':i/100,**op(y,conf,n,i/100)}for i in range(101)]
 return max(rows,key=lambda z:(z['F05'],-z['threshold']))
def save(name,obj):
 p=OUT/'results'/name;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2));return p
