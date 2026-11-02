import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
r=json.load(open('results.json'))
runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
print("== (1) decoder ratio recon/real")
for s in S:
    d=runs[s]['diagnose']['area']['groups']
    print(s)
    for g in G:
        lf=d[g]['last_frame']['ratio_recon_over_real']
        pe=np.array(d[g]['per_episode_ratio_recon_k12'])
        ks=d[g]['by_k']
        mk=[ks[k]['ratio_recon_over_real']['mean'] for k in ['1','2','3','4','6','8','12']]
        rm=d[g]['last_frame']['recon']['mean']/d[g]['last_frame']['real']['mean']
        print(f" {g} k12 mean {lf['mean']:.4f} [{lf['lo']:.4f},{lf['hi']:.4f}] ratio-of-means {rm:.4f} median_pe {np.median(pe):.3f} min_k_mean {min(mk):.4f} max_k_mean {max(mk):.4f} frac_pe<0.9 {np.mean(pe<0.9):.2f} n={len(pe)} nan={np.isnan(pe).sum()}")
