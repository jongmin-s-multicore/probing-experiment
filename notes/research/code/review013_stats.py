import json, numpy as np
R='/home/jaekyungkim/mylab/experiments/013_size_use_erasure/'
d=json.load(open(R+'results.json')); C=d['stage1']['conditions']
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
splits=['train','interp','interp_heldhalf','extrap','extrap_heldhalf']
for who in ['trained','random_init']:
  for m in ['leace_scalar','level_mean_match']:
    for met in ['motion','total']:
      P(f'\n## {who} {m} metric={met}  (delta_use [lo,hi] | du/gap | size_inc | rand_inc | n)')
      for cond in C:
        for sp in [q for q in splits if q in C[cond]['seeds']['0'][who][m]]:
          row=[]
          for s in '012':
            x=C[cond]['seeds'][s][who][m][sp][met]
            du=x['delta_use']
            row.append(f"{du['point']:+.4f}[{du['ci_lo']:+.4f},{du['ci_hi']:+.4f}] r={x['delta_use_over_gap']:+.3f} si={x['increase_size']['point']:.4f} ri={x['increase_rand_mean']['point']:.4f}")
          P(cond,sp,'n=',x['n_episodes'],'\n   '+'\n   '.join(row))
# recompute verdict
P('\n## recompute verdict (train, trained)')
for m in ['leace_scalar','level_mean_match']:
  for met in ['motion','total']:
    res={}
    for cond in C:
      du=[];gap=[];pos=0
      for s in '012':
        x=C[cond]['seeds'][s]['trained'][m]['train'][met]
        du.append(x['delta_use']['point']);gap.append(x['gap']['point'])
        pos+= x['delta_use']['ci_lo']>0
      res[cond]=(np.mean(du),np.mean(gap),pos,du)
    b=res['multi_60_ind_160'];a=res['single_ind']
    P(m,met,'b mean du %.4f gap %.4f ratio %.3f seeds_pos %d | a mean du %.4f | c1 %s c2 %s c3 %s'%(b[0],b[1],b[0]/b[1],b[2],a[0],b[2]>=2,b[0]>=.1*b[1],b[0]>a[0]))
# random direction spread
P('\n## rand-direction spread (train, motion & total), trained: mean, sd, min, max across 20 dirs; size increase; z of size vs rand')
for m in ['leace_scalar','level_mean_match']:
  for met in ['motion','total']:
    for cond in C:
      for s in '012':
        x=C[cond]['seeds'][s]['trained'][m]['train'][met]
        r=np.array(x['rand_increase_by_direction']);si=x['increase_size']['point']
        P(m,met,cond,s,'rand mean %.4f sd %.4f min %.4f max %.4f | size %.4f | size-max %.4f | rank of size among 21: %d'%(r.mean(),r.std(ddof=1),r.min(),r.max(),si,si-r.max(),(r>=si).sum()+1))
# between-seed sd of du
P('\n## seed sd of delta_use (train motion), b')
for m in ['leace_scalar','level_mean_match']:
  for met in ['motion','total']:
    for cond in C:
      v=[C[cond]['seeds'][s]['trained'][m]['train'][met]['delta_use']['point'] for s in '012']
      ci=[(C[cond]['seeds'][s]['trained'][m]['train'][met]['delta_use']['ci_hi']-C[cond]['seeds'][s]['trained'][m]['train'][met]['delta_use']['ci_lo'])/2 for s in '012']
      P(m,met,cond,'mean %.4f sd %.4f  mean CI halfwidth %.4f'%(np.mean(v),np.std(v,ddof=1),np.mean(ci)))
# readout
P('\n## readout R2 (late, linear), mean over seeds, per cond/method/split, depth0..6')
RO=d['erasure_check_readout_r2']
for cond in RO:
  for meth in RO[cond]['0']:
    for sp in ['train','interp','extrap']:
      arr=np.array([[RO[cond][s][meth]['r2_by_split_depth'][sp][str(k)] for k in range(7)] for s in '012'])
      P(cond,meth,sp,np.round(arr.mean(0),3).tolist(),'seed sd max %.3f'%arr.std(0,ddof=1).max())
open(R+'analysis/review013_stats_tables.txt','w').write('\n'.join(out))
