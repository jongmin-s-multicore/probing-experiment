import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json'))
runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
KS=['1','2','3','4','6','8','12']
print("log-log slope of ARITHMETIC group means (tf1 vs real), by k; ratio_tf1 at a026 / a248 (mean-of-ratio)")
for s in S:
    d=runs[s]['diagnose']['area']['groups']
    for key in ['tf1','ol','recon']:
        out=[]
        for k in KS:
            x=np.log([d[g]['by_k'][k]['real']['mean'] for g in G]); y=np.log([d[g]['by_k'][k][key]['mean'] for g in G])
            out.append(f"k{k}:{np.polyfit(x,y,1)[0]:.3f}")
        print(s[-2:],key,' '.join(out))
    out=[]
    for k in KS:
        out.append(f"k{k}:{d['a026']['by_k'][k]['ratio_tf1_over_real']['mean']:.2f}/{d['a248']['by_k'][k]['ratio_tf1_over_real']['mean']:.2f}")
    print(s[-2:],'tf1 ratio a026/a248',' '.join(out))
# real area by k (ind) and spread
d=runs[S[0]]['diagnose']['area']['groups']
for g in ['a026','ind','a248']:
    print(g,'real mean by k',[ (k,round(d[g]['by_k'][k]['real']['mean']), round(d[g]['by_k'][k]['real']['hi']-d[g]['by_k'][k]['real']['lo'])) for k in KS])
    print(g,'tf1 mean by k',[ (k,round(d[g]['by_k'][k]['tf1']['mean']), round(d[g]['by_k'][k]['tf1']['hi']-d[g]['by_k'][k]['tf1']['lo'])) for k in KS])
# replicate the group-mean-log slope at k12 using arithmetic means
for s in S:
    sc=runs[s]['diagnose']['size_curve']
    for k in ['k3','k6','k12']:
        x=np.log([sc['by_group'][g][k]['real'] for g in G]);
        for key in ['tf1','ol']:
            y=np.log([sc['by_group'][g][k][key] for g in G])
            print(s[-2:],k,key,'arith-mean loglog slope %.3f'%np.polyfit(x,y,1)[0], 'reported', round(sc['slope_model_vs_real_group_means'][f'{key}@{k}']['slope'],3), end=' | ')
    print()
