import json, numpy as np
R='/home/jaekyungkim/mylab/experiments/013_size_use_erasure/'
d=json.load(open(R+'results.json')); C=d['stage1']['conditions']
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
P('## decomposition motion = |e-delta|^2, total=|e|^2 => 2<e,delta> = total + |delta|^2 - motion  (per-element units; |delta|^2 per element = disp_sq_mean/384 ASSUMED)')
for m in ['leace_scalar','level_mean_match']:
  for cond in C:
    for s in '012':
      x=C[cond]['seeds'][s]['trained'][m]['train']['abs_means_over_episodes']
      row=[]
      for k in ['size','rand_mean']:
        tot=x['mean_total'][k];mot=x['mean_motion'][k];ds=x['mean_disp_sq_mean'][k]
        row.append((k,round(tot,4),round(mot,4),round(ds,3),round(ds/384,4),round((tot+ds/384-mot)/2,4)))
      P(m,cond,s,'persist',round(x['mean_persist']['none'],4),'none',round(x['mean_total']['none'],4),row)
P('\n## seed-level t tests (n=3 per cond), train, trained')
for m in ['leace_scalar','level_mean_match']:
  for met in ['motion','total']:
    g=lambda c:[C[c]['seeds'][s]['trained'][m]['train'][met]['delta_use']['point'] for s in '012']
    b,a=g('multi_60_ind_160'),g('single_ind')
    sb,sa=np.var(b,ddof=1)/3,np.var(a,ddof=1)/3
    tw=(np.mean(b)-np.mean(a))/np.sqrt(sa+sb)
    dfw=(sa+sb)**2/(sa**2/2+sb**2/2)
    n=3;mb=np.mean(b);se=np.std(b,ddof=1)/np.sqrt(n);q=2.919986
    P(m,met,'b-a diff %.4f welch t %.2f df %.2f (no scipy; p not computed) | b mean/se t=%.2f | 90%% t-int(df2) of b mean [%.4f, %.4f]'%(mb-np.mean(a),tw,dfw,mb/se,mb-q*se,mb+q*se))
P('\n## random_init: (a) vs (b) identical?  max abs diff in train delta_use over seeds')
for m in ['leace_scalar','level_mean_match']:
  for met in ['motion','total']:
    P(m,met,max(abs(C['single_ind']['seeds'][s]['random_init'][m]['train'][met]['delta_use']['point']-C['multi_60_ind_160']['seeds'][s]['random_init'][m]['train'][met]['delta_use']['point']) for s in '012'))
P('\n## motion_norm_none trained vs random_init (train)')
for cond in C:
  for s in '012':
    P(cond,s,'trained none %.3f | RI none %.3f size %.3f'%(C[cond]['seeds'][s]['trained']['leace_scalar']['train']['motion']['motion_norm_none']['point'],C[cond]['seeds'][s]['random_init']['leace_scalar']['train']['motion']['motion_norm_none']['point'],C[cond]['seeds'][s]['random_init']['leace_scalar']['train']['motion']['motion_norm_size']['point']))
P('\n## split-wise B-vs-A sign agreement (trained, b), LMM & LEACE; cond (a) interp sign by seed')
P('## stage2'); 
u=C['multi_60_ind_160']['seeds']['0']
print(json.dumps(u['stage2_level_mean_delta_z'])[:700])
for cond in C:
  for s in '012':
    P(cond,s,json.dumps(C[cond]['seeds'][s]['stage2_level_mean_delta_z'])[:400])
open(R+'analysis/review013_stats_tables2.txt','w').write('\n'.join(out))
