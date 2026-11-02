import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
for s in S:
  for g in ['a125','a080','ind']:
    rd=np.array(runs[s]['intervene']['modes']['remove'][g]['tf_response_gain_along_random_r']['per_direction_mean'])
    print(s[-2:],g,'argmax',rd.argmax(),round(rd.max(),2),'second', np.sort(rd)[-2].round(3),'top3 idx',np.argsort(rd)[-3:])
# robust summary: u gain vs median random, excluding outlier; paired across 9 groups, per seed
for s in S:
    u=[];m=[];mx=[]
    for g in G:
        d=runs[s]['intervene']['modes']['remove'][g]; rd=np.array(d['tf_response_gain_along_random_r']['per_direction_mean'])
        u.append(d['tf_response_gain_along_u']['mean']); m.append(np.median(rd)); mx.append(rd.mean())
    u=np.array(u);m=np.array(m)
    print(s[-2:],'u-median(rand) per group',np.round(u-m,3),'ratio u/median',np.round(u/m,2))
from scipy import stats
# across 3 seeds x 9 groups: u vs median random (paired); groups are not independent
allu=[];allm=[]
for s in S:
    for g in G:
        d=runs[s]['intervene']['modes']['remove'][g]; rd=np.array(d['tf_response_gain_along_random_r']['per_direction_mean'])
        allu.append(d['tf_response_gain_along_u']['mean']); allm.append(np.median(rd))
allu=np.array(allu);allm=np.array(allm)
print('wilcoxon 27 pairs (non-independent):',stats.wilcoxon(allu,allm))
# aggregated dlogarea frame0
for mode in ['remove']:
    for s in S:
        du=np.array([[x['mean'] for x in runs[s]['intervene']['modes'][mode][g]['tf_decoded_dlogarea_u_by_frame']] for g in G])
        dc=np.array([[x['lo']>0 or x['hi']<0 for x in runs[s]['intervene']['modes'][mode][g]['tf_decoded_dlogarea_u_by_frame']] for g in G])
        print(s[-2:],'dlogarea_u mean by frame over groups',du.mean(0).round(4),'groups with CI excl 0 per frame',dc.sum(0))
        dr=np.array([runs[s]['intervene']['modes'][mode][g]['tf_decoded_dlogarea_random_mean_by_frame'] for g in G]); print('   random',dr.mean(0).round(4))
# open-loop
for s in S[:1]:
    for g in ['a026','ind','a248']:
        d=runs[s]['intervene']['modes']['remove'][g]
        print(g,'ol dlog by k',np.round(d['ol_decoded_dlogarea_by_k'],3))
# set modes
print()
for mode in runs[S[0]]['intervene']['modes']:
    if mode=='remove': continue
    for s in S:
        g=list(runs[s]['intervene']['modes'][mode])[0]
        d=runs[s]['intervene']['modes'][mode][g]
        rd=np.array(d['tf_response_gain_along_random_r']['per_direction_mean'])
        print(mode,s[-2:],f"u gain {d['tf_response_gain_along_u']['mean']:.3f}[{d['tf_response_gain_along_u']['lo']:.3f},{d['tf_response_gain_along_u']['hi']:.3f}] rand med {np.median(rd):.3f} mean {rd.mean():.3f} min {rd.min():.2f} max {rd.max():.2f}  dlog u {np.round([x['mean'] for x in d['tf_decoded_dlogarea_u_by_frame']],3)} rand {np.round(d['tf_decoded_dlogarea_random_mean_by_frame'],3)} OL final dlog {d['ol_decoded_dlogarea_by_k'][-1]:.3f}")
