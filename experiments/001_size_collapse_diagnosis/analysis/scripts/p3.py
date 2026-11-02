import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
def get(s,pool,mdl,proto,typ,sub='all'): return runs[s]['probe']['pools'][pool][mdl][proto][typ][sub]
print("probe seed sd (within train seed) vs CI half-width vs across-training-seed sd; pool mean ridge")
for proto,typ in [('strat_cv','gt'),('strat_cv','rvit'),('strat_cv','pix'),('strat_cv','tf1'),('strat_cv','ol'),('gt_transfer','tf1'),('gt_transfer','ol'),('upper','ol'),('logo','ol'),('heldout_size','ol')]:
  for met in ['r2_overall','r2_within_group','acc_cls','slope']:
    try:
        ds=[get(s,'mean','ridge',proto,typ)[met] for s in S]
    except KeyError: continue
    if ds[0]['mean'] is None: continue
    means=np.array([d['mean'] for d in ds]); sds=np.array([d['sd'] for d in ds]); hw=np.array([(d['ci_hi']-d['ci_lo'])/2 for d in ds])
    print(f"{proto:12s}{typ:5s}{met:16s} mean {means.mean():.3f}  probeSeed sd(avg) {sds.mean():.4f}  CI halfwidth(avg) {hw.mean():.4f} CI0 [{ds[0]['ci_lo']:.3f},{ds[0]['ci_hi']:.3f}]  trainSeed sd {means.std(ddof=1):.4f}")
# differences gt - rvit, gt-pix on strat_cv ridge/mlp/grid2: point & CI-overlap
print()
for pool,mdl in [('mean','ridge'),('grid2','ridge'),('mean','mlp')]:
  for met in ['r2_overall','r2_within_group','acc_cls','acc_cls_pm1']:
    g=get(S[0],pool,mdl,'strat_cv','gt')[met]; 
    out=[f"{pool}/{mdl} {met}: gt {g['mean']:.3f}[{g['ci_lo']:.3f},{g['ci_hi']:.3f}]"]
    for c in ['rvit','pix']:
        x=get(S[0],pool,mdl,'strat_cv',c)[met]
        out.append(f"{c} {x['mean']:.3f}[{x['ci_lo']:.3f},{x['ci_hi']:.3f}] diff {g['mean']-x['mean']:+.3f}")
    print(' | '.join(out))
# selectivity vs permute
print()
for s in S:
  sel=runs[s]['probe']['pools']['mean']['ridge']['selectivity_real_minus_permute']
  print(s[-2:],{t:{m:round(v,3) for m,v in d.items()} for t,d in sel.items()})
sel=lambda s,pool,mdl: runs[s]['probe']['pools'][pool][mdl]['selectivity_real_minus_permute']
print('mlp',{t:{m:round(v,3) for m,v in d.items()} for t,d in sel(S[0],'mean','mlp').items()})
perm=runs[S[0]]['probe']['pools']['mean']['ridge']['label_permute_strat_cv']
for t in perm: print('permute',t,{m:round(perm[t]['all'][m]['mean'],3) for m in ['r2_overall','r2_within_group','acc_cls','slope']}, 'sd acc',perm[t]['all']['acc_cls']['sd'])
