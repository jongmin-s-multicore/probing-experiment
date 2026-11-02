import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np
from scipy import stats
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
print("k=12 ol/real ratio: mean-of-ratio (file), ratio-of-means, median per-episode; per seed")
for g in G:
    mr=[runs[s]['diagnose']['area']['groups'][g]['last_frame']['ratio_ol_over_real']['mean'] for s in S]
    rm=[runs[s]['diagnose']['area']['groups'][g]['last_frame']['ol']['mean']/runs[s]['diagnose']['area']['groups'][g]['last_frame']['real']['mean'] for s in S]
    md=[np.median(runs[s]['diagnose']['area']['groups'][g]['per_episode_ratio_ol_k12']) for s in S]
    t1=[runs[s]['diagnose']['area']['groups'][g]['last_frame']['ratio_tf1_over_real']['mean'] for s in S]
    print(g,'MoR',np.round(mr,2),'RoM',np.round(rm,2),'median',np.round(md,2),'tf1 MoR',np.round(t1,2),'seed sd RoM %.3f'%np.std(rm,ddof=1))
# slopes across seeds with t-tests
sl=lambda key,k:[runs[s]['diagnose']['size_curve']['slope_model_vs_real_group_means'][f'{key}@k{k}']['slope'] for s in S]
for key in ['tf1','ol','recon']:
    for k in [3,6,12]:
        x=np.array(sl(key,k)); t=stats.ttest_1samp(x,1.0)
        print(key,k,'mean %.3f sd %.3f'%(x.mean(),x.std(ddof=1)),'t=%.1f p=%.4f'%(t.statistic,t.pvalue), 'seed-t 95%% CI [%.3f,%.3f]'%tuple(stats.t.interval(0.95,2,x.mean(),x.std(ddof=1)/np.sqrt(3))))
a=np.array(sl('tf1',3));b=np.array(sl('tf1',12)); print('tf1 k12-k3 paired',b-a,stats.ttest_rel(b,a))
# difference of CI-width implied SE
for k in [3,12]:
    d=[runs[s]['diagnose']['size_curve']['slope_model_vs_real_group_means'][f'tf1@k{k}'] for s in S]
    print(k,'implied SE',[(x['hi']-x['lo'])/3.92 for x in d])
# bootstrap nominal size vs measured area ratios (maskcheck vs nominal)
print('measured/nominal area ratio frame3 (s0):',[ (g, round(runs[S[0]]['diagnose']['size_curve']['by_group'][g]['measured_area_ratio_frame3_vs_ref'],3), runs[S[0]]['diagnose']['size_curve']['by_group'][g]['nominal_area_ratio']) for g in G])
# label check
print(runs[S[0]]['diagnose']['label_check_particles_vs_mask_area']['by_frame'])
