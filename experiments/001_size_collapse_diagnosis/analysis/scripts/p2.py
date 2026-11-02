import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
def get(s,pool,mdl,proto,typ,sub='all'):
    return runs[s]['probe']['pools'][pool][mdl][proto][typ][sub]
def f(x): return 'NA' if x is None else f"{x:.3f}"
for pool,mdl in [('mean','ridge'),('grid2','ridge'),('mean','mlp')]:
  print('=====',pool,mdl)
  for proto in ['strat_cv','logo','gt_transfer','upper','heldout_size']:
    for typ in ['gt','tf1','ol','rvit','pix']:
        row=[]
        for s in S:
            try: d=get(s,pool,mdl,proto,typ)
            except KeyError: row=None;break
            row.append((d['r2_overall']['mean'],d['r2_within_group']['mean'],d['acc_cls']['mean'],d['acc_cls_pm1']['mean'],d['slope']['mean'],d['slope_group_means']['mean']))
        if row is None: continue
        a=np.array([[np.nan if v is None else v for v in x] for x in row],float)
        print(f"{proto:12s}{typ:5s} r2 {np.round(a[:,0],3)} r2w {np.round(a[:,1],3)} acc {np.round(a[:,2],3)} pm1 {np.round(a[:,3],3)} slope {np.round(a[:,4],3)} slopeGM {np.round(a[:,5],3)}")
  try:
    c=get('exp001_cs_baseline_single_s0',pool,mdl,'constant_mean','gt')
    print('const', {k:(v['mean'] if isinstance(v,dict) else None) for k,v in c.items() if k not in ('bias_log_area_by_group','n_episodes')})
  except KeyError as e: print('no const',e)
