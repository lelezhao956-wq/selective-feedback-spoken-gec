import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle,itertools,time
t0=time.time();trp=RUN/'2026-11-10_C1_features/features_train.csv';dvp=RUN/'2026-11-10_C1_features/features_dev.csv'
tr=pd.read_csv(trp);dv=pd.read_csv(dvp);col=['asr_conf_win_min'];grid=[]
def model(k,c):return make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=k,degree=3,knots='quantile',include_bias=False,extrapolation='constant'),StandardScaler(),LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=2000,random_state=SEED))
for k,c in itertools.product([3,5,7],[.01,.1,1,10]):
 m=model(k,c);m.fit(tr[col],tr.false_feedback);auc=roc_auc_score(dv.false_feedback,m.predict_proba(dv[col])[:,1]);grid.append({'knots':k,'C':c,'dev_AUROC':float(auc)})
best=sorted(grid,key=lambda z:(-round(z['dev_AUROC'],12),z['knots'],z['C']))[0];pool=pd.concat([tr,dv],ignore_index=True);m=model(best['knots'],best['C']);m.fit(pool[col],pool.false_feedback)
mp=OUT/'results/asr_spline_frozen.pkl';mp.write_bytes(pickle.dumps(m))
freeze={'structure':'one scalar confidence; imputer -> cubic B-spline -> standard scaler -> L2 LR','selection':best,'candidate_dev_scores':grid,'final_n':len(pool),'fill_value':float(m[0].statistics_[0]),'spline_basis_dimension':m[1].n_features_out_,'model_sha256':sha(mp),'inputs_train_dev':{str(p):sha(p)for p in [trp,dvp]},'test_data_read_during_fitting_or_selection':False}
save('asr_spline_freeze.json',freeze);print('FROZEN_BEFORE_TEST',json.dumps(freeze),flush=True)
ep=RUN/'2026-11-10_C7_B1_eval/features/features_eval.csv';up=ep.parent/'utt_counts_eval.csv';sp=RUN/'2026-11-10_C7_B1_eval/eval_scores_frozen.csv';d=pd.read_csv(ep);u=pd.read_csv(up);s=pd.read_csv(sp)
merged=d[['utt_id','edit_idx','false_feedback']].merge(s,on=['utt_id','edit_idx'],suffixes=('_feature','_score'),validate='1:1');assert len(merged)==len(d)
assert np.array_equal(merged.false_feedback_feature,d.false_feedback)and np.array_equal(merged.false_feedback_score,d.false_feedback)
fit=json.load(open(RUN/'2026-11-10_C6_final/final_fit.json'));S={'asr_conf':1-d.asr_conf_win_min.fillna(fit['asr_conf_fill']).to_numpy(),'asr_spline':m.predict_proba(d[col])[:,1],'gec_lr':merged.s_gec_lr.to_numpy(),'risk_lgbm':merged.s_risk_lgbm.to_numpy()}
pairs=list(itertools.combinations(S,2));out=boot(d,u,S,pairs);old=json.load(open(R/'06_results/eval_results.json'));checks={}
for k in ['asr_conf','gec_lr','risk_lgbm']:
 checks[k]={'point':max(abs(out['point'][k][v]-old['point'][k][v])for v in MET)<1e-12,'ci':max(abs(out['ci95'][k][v][i]-old['ci95'][k][v][i])for v in MET for i in [0,1])<1e-12}
assert all(all(x.values())for x in checks.values()),checks
out.update({'status':'completed post hoc univariate nonmonotonic baseline','frozen':freeze,'original_metric_replication':checks,'input_sha256':{str(p):sha(p)for p in [trp,dvp,ep,up,sp]},'script_sha256':sha(__file__),'seconds':time.time()-t0})
pd.DataFrame({'utt_id':d.utt_id,'edit_idx':d.edit_idx,**S}).to_csv(OUT/'results/asr_spline_test_scores.csv',index=False)
save('asr_spline_results.json',out);print(json.dumps({k:out[k]for k in ['status','valid_resamples','original_metric_replication','point','ci95','paired']},indent=2),flush=True)
