import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json'))
runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
print("== (2) slopes")
for key in ['recon','tf1','ol']:
  for k in ['3','6','12']:
    row=[]
    for s in S:
        d=runs[s]['diagnose']['size_curve']['slope_model_vs_real_group_means'][f'{key}@k{k}']
        row.append(f"{d['slope']:.3f}[{d['lo']:.3f},{d['hi']:.3f}]{'X' if d['hi']<1 or d['lo']>1 else 'I'}")
    print(key,k,' | '.join(row))
# across seeds
m=r['across_seeds']['baseline_single']['metrics']
ks=[k for k in m if 'slope' in k]
print(len(ks)); print(ks[:30])
# size curve by group
for s in S[:1]:
    for g in G:
        d=runs[s]['diagnose']['size_curve']['by_group'][g]
        print(g,d['nominal_area_ratio'],d['measured_area_ratio_frame3_vs_ref'],{k:{a:round(b,0) for a,b in d[k].items()} for k in ['k3','k6','k12']})
