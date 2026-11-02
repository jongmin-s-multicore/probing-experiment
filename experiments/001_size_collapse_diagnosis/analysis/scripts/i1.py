import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
print("remove: u gain | random: mean, median, max, n>1 | rank of u among 21 (1=largest) | emp p=(#rand>=u +1)/21 | decoded dlogarea u vs random frame0")
rows=[]
for s in S:
  for g in G:
    d=runs[s]['intervene']['modes']['remove'][g]
    u=d['tf_response_gain_along_u']['mean']; rd=np.array(d['tf_response_gain_along_random_r']['per_direction_mean'])
    p=(np.sum(rd>=u)+1)/(len(rd)+1)
    du=[x['mean'] for x in d['tf_decoded_dlogarea_u_by_frame']]; dr=d['tf_decoded_dlogarea_random_mean_by_frame']
    rows.append((s[-2:],g,u,d['tf_response_gain_along_u']['lo'],d['tf_response_gain_along_u']['hi'],rd.mean(),d['tf_response_gain_along_random_r']['mean'],np.median(rd),rd.max(),(rd>1).sum(),p,du,dr, d['n_episodes']))
    print(f"{s[-2:]} {g} n={d['n_episodes']} u {u:.3f}[{d['tf_response_gain_along_u']['lo']:.3f},{d['tf_response_gain_along_u']['hi']:.3f}] rand mean(file) {d['tf_response_gain_along_random_r']['mean']:.3f} recalcmean {rd.mean():.3f} med {np.median(rd):.3f} p5-p95(file) {d['tf_response_gain_along_random_r']['p5']:.3f}-{d['tf_response_gain_along_random_r']['p95']:.3f} max {rd.max():.2f} n>1: {(rd>1).sum()} p={p:.3f} | dlog u {np.round(du,3)} rand {np.round(dr,4)}")
