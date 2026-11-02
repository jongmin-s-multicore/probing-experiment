import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
for typ in ['gt','tf1','ol']:
  for sub in ['role_extrap_small','role_trained','role_extrap_large']:
    row=[]
    for s in S:
        d=runs[s]['probe']['pools']['mean']['ridge']['heldout_size'][typ][sub]
        row.append((d['r2_overall']['mean'],d['slope_group_means']['mean'], d['slope']['mean'], d['n_episodes']))
    print(typ,sub,[tuple((round(x,3) if x is not None else None) for x in y) for y in row])
# bias by group for heldout_size and gt_transfer
for proto in ['heldout_size','gt_transfer','strat_cv']:
  for typ in ['gt','tf1','ol']:
    try:
        b=[runs[s]['probe']['pools']['mean']['ridge'][proto][typ]['all']['bias_log_area_by_group'] for s in S]
    except KeyError: continue
    G=list(b[0]); print(proto,typ,{g:[round(x[g],2) for x in b] for g in G})
