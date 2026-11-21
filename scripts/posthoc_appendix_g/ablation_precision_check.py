"""Re-evaluate archived OOF scores only; no fitting. Recover pre-rounding precision."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
p=RUN/'2026-11-12_F2/F2_oof_trdev.csv';d=pd.read_csv(p)
u=pd.concat([pd.read_csv(RUN/f'2026-11-10_C1_features/utt_counts_{s}.csv')for s in ['train','dev']],ignore_index=True)
scores={k[2:]:d[k].to_numpy()for k in d if k.startswith('s_')}
out=boot(d,u,scores,[(k,'all')for k in scores if k!='all']);out['input_sha256']={str(p):sha(p),**{str(RUN/f'2026-11-10_C1_features/utt_counts_{s}.csv'):sha(RUN/f'2026-11-10_C1_features/utt_counts_{s}.csv')for s in ['train','dev']}}
old=json.load(open(R/'06_results/F2_ablation_trdev.json'))['ablation'];checks=[]
for k,z in out['paired'].items():
 key=k.replace('_minus_',' - ')
 for m in ['AURC','P@0.5']:
  a=old['diff'][key][m];checks.append(round(z[m]['diff'],4)==a['diff']);checks.extend(round(x,4)==y for x,y in zip(z[m]['ci95'],a['ci95']))
assert all(checks),(len(checks),sum(checks))
out['all_rounded4_archived_differences_and_CIs_match']=True
out['note']='Full precision recovered from archived OOF predictions using original bootstrap; no model fitted; avoids second rounding of four-decimal JSON.'
save('ablation_full_precision_check.json',out)
for key,m in [('-EDIT_minus_all','AURC'),('only_ASR_minus_all','P@0.5'),('only_GEC_minus_all','P@0.5')]:print(key,m,out['paired'][key][m],flush=True)
print('ARCHIVED_ROUND4_REPLICATION',len(checks),'checks passed; valid resamples',out['valid_resamples'],flush=True)
