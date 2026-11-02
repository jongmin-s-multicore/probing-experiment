import sys; sys.path.insert(0,"/tmp/claude-1000/-home-jaekyungkim-mylab/f0a84b41-5f2a-4d16-ab5d-48c13f8581a2/scratchpad/pkgs")
import json, numpy as np, csv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=json.load(open('results.json')); runs=r['runs']; S=list(runs)
G=['a026','a040','a060','a080','ind','a125','a160','a200','a248']
out='analysis/'
# 1 decoder ratio
fig,ax=plt.subplots(1,3,figsize=(15,4))
x=np.arange(9)
for i,s in enumerate(S):
    lf=[runs[s]['diagnose']['area']['groups'][g]['last_frame']['ratio_recon_over_real'] for g in G]
    ax[0].errorbar(x+0.12*(i-1),[d['mean'] for d in lf],yerr=[[d['mean']-d['lo'] for d in lf],[d['hi']-d['mean'] for d in lf]],fmt='o',label=f'seed{i}',capsize=2)
ax[0].axhline(0.9,color='r',ls='--');ax[0].axhline(1.1,color='r',ls='--');ax[0].set_xticks(x);ax[0].set_xticklabels(G,rotation=45);ax[0].set_title('recon/real area, k=12 (mean-of-ratio, 95% boot CI)');ax[0].legend()
# 2 slopes vs k
for j,key in enumerate(['tf1','ol']):
    for i,s in enumerate(S):
        d=[runs[s]['diagnose']['size_curve']['slope_model_vs_real_group_means'][f'{key}@k{k}'] for k in (3,6,12)]
        ax[1+j].errorbar(np.arange(3)+0.1*(i-1),[a['slope'] for a in d],yerr=[[a['slope']-a['lo'] for a in d],[a['hi']-a['slope'] for a in d]],fmt='o-',capsize=3,label=f'seed{i}')
    ax[1+j].axhline(1,color='k',ls=':');ax[1+j].set_xticks(range(3));ax[1+j].set_xticklabels(['k3','k6','k12']);ax[1+j].set_title(f'{key} log-log slope vs real (episode-boot 95% CI)')
plt.tight_layout();plt.savefig(out+'fig1_decoder_slopes.png',dpi=110);plt.close()
# 3 probe
fig,ax=plt.subplots(1,3,figsize=(15,4))
labels=['gt','rvit','pix','tf1','ol']
for a,met in zip(ax,['r2_overall','r2_within_group','acc_cls']):
    for i,l in enumerate(labels):
        vals=[runs[s]['probe']['pools']['mean']['ridge']['strat_cv'][l]['all'][met] for s in S]
        a.errorbar([i+0.1*(j-1) for j in range(3)],[v['mean'] for v in vals],yerr=[[v['mean']-v['ci_lo'] for v in vals],[v['ci_hi']-v['mean'] for v in vals]],fmt='o',capsize=3)
    a.set_xticks(range(5));a.set_xticklabels(labels);a.set_title(f'strat_cv ridge/mean {met} (3 train seeds, boot CI)')
    if met=='r2_within_group': a.set_ylim(-0.5,1)
plt.tight_layout();plt.savefig(out+'fig2_probe_controls.png',dpi=110);plt.close()
# 4 intervene
fig,ax=plt.subplots(1,3,figsize=(15,4))
for i,s in enumerate(S):
    u=[runs[s]['intervene']['modes']['remove'][g]['tf_response_gain_along_u']['mean'] for g in G]
    ax[i].plot(x,u,'ro-',label='u')
    for g_i,g in enumerate(G):
        rd=runs[s]['intervene']['modes']['remove'][g]['tf_response_gain_along_random_r']['per_direction_mean']
        ax[i].scatter([g_i]*len(rd),rd,s=8,c='gray')
    ax[i].set_ylim(0,0.5);ax[i].set_xticks(x);ax[i].set_xticklabels(G,rotation=45);ax[i].set_title(f'seed{i} remove: gain u (red) vs 20 random dirs (clipped at 0.5)')
plt.tight_layout();plt.savefig(out+'fig3_intervene_remove.png',dpi=110);plt.close()
# csv tables
with open(out+'decoder_ratio_table.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['seed','group','recon_over_real_k12_MoR','ci_lo','ci_hi','ratio_of_means','min_over_k_MoR','max_over_k_MoR'])
    for s in S:
        for g in G:
            d=runs[s]['diagnose']['area']['groups'][g]; lf=d['last_frame']['ratio_recon_over_real']
            ks=[d['by_k'][k]['ratio_recon_over_real']['mean'] for k in d['by_k']]
            w.writerow([s,g,lf['mean'],lf['lo'],lf['hi'],d['last_frame']['recon']['mean']/d['last_frame']['real']['mean'],min(ks),max(ks)])
print('ok')
