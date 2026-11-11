import json, numpy as np, itertools
R=json.load(open('experiments/012_erasure_residual_info/results.json'))
B=R['blocks']
W='late'
def rec(m,key,s,ts,meas,split,win=W,aux=False):
    k='records_depth_m_int_plus_1' if aux else 'records_depth_m_int'
    r=B[str(m)][k][f'strength_{s}'][str(ts)][meas][win]
    return r[split]['per_seed']
out=[]
# 1 recompute verdict
print('== recompute verdict (split all for b,c; train for a)')
for m in range(1,7):
    row=[]
    for ts in range(3):
        a=np.mean(rec(m,0,1,ts,'a_level_logistic','train'))
        mlp=np.mean(rec(m,0,1,ts,'b_mlp','all')); att=np.mean(rec(m,0,1,ts,'b_attentive','all'))
        c=np.mean(rec(m,0,1,ts,'c_action_only_linear','all'))
        row.append((a,mlp,att,c))
    row=np.array(row)
    na=(row[:,0]>=0.5333333333).sum(); nb=((row[:,1]>=.2)|(row[:,2]>=.2)).sum(); nc=(row[:,3]>=.2).sum()
    v=B[str(m)]['verdict']['items']
    print(m,'a',row[:,0].round(3),na,v['a_class_structure']['n_seeds_met'],'| mlp',row[:,1].round(3),'att',row[:,2].round(3),nb,v['b_nonlinear']['n_seeds_met'],'| c',row[:,3].round(3),nc,v['c_action_path']['n_seeds_met'])

import sys
def ms(x): x=np.array(x); return x.mean(), (x.std(ddof=1) if len(x)>1 else float('nan'))
print('\n== T1: mean over probe seeds, per block/train seed, late, split all (a: train). s0 -> s1')
for meas,sp in [('a_level_logistic','train'),('b_mlp','all'),('b_attentive','all'),('d_linear_pooled','all'),('c_action_only_linear','all')]:
    print('--',meas,sp)
    for m in range(1,7):
        cells=[]
        for ts in range(3):
            x0=rec(m,0,0,ts,meas,sp); x1=rec(m,0,1,ts,meas,sp)
            cells.append(f'{np.mean(x0):.3f}->{np.mean(x1):.3f}')
        s0=[np.mean(rec(m,0,0,ts,meas,sp)) for ts in range(3)]; s1=[np.mean(rec(m,0,1,ts,meas,sp)) for ts in range(3)]
        d=np.array(s1)-np.array(s0)
        print(m,' | '.join(cells),f' delta mean {d.mean():+.3f} sd_ts {d.std(ddof=1):.3f}')
print('\n== T2: probe-seed SD (s1, all, late) and across-train-seed SD')
for meas in ['b_mlp','b_attentive','c_action_only_linear','d_linear_pooled']:
    for m in range(1,7):
        sds=[ms(rec(m,0,1,ts,meas,'all'))[1] for ts in range(3)]
        means=[np.mean(rec(m,0,1,ts,meas,'all')) for ts in range(3)]
        print(meas,m,'probe-seed SD per ts',np.round(sds,4),'train-seed SD of means',round(np.std(means,ddof=1),4),'range',round(min(means),3),round(max(means),3))
print('\n== T3: split values s1 late (mean over ts) per block')
for meas in ['b_mlp','b_attentive','c_action_only_linear']:
    for m in range(1,7):
        row={sp:np.mean([np.mean(rec(m,0,1,ts,meas,sp)) for ts in range(3)]) for sp in ['train','interp','extrap','all']}
        print(meas,m,{k:round(v,3) for k,v in row.items()})
print('\n== T4: split verdict counts for b (mlp or att >=.2) and c per split')
for sp in ['train','interp','extrap','all']:
    for m in range(1,7):
        nb=sum((np.mean(rec(m,0,1,ts,'b_mlp',sp))>=.2) or (np.mean(rec(m,0,1,ts,'b_attentive',sp))>=.2) for ts in range(3))
        nbm=sum((np.mean(rec(m,0,1,ts,'b_mlp',sp))>=.2) for ts in range(3))
        nba=sum((np.mean(rec(m,0,1,ts,'b_attentive',sp))>=.2) for ts in range(3))
        nc=sum(np.mean(rec(m,0,1,ts,'c_action_only_linear',sp))>=.2 for ts in range(3))
        print(sp,m,'b',nb,'(mlp',nbm,'att',nba,') c',nc)
print('\n== T5: c margin vs threshold; c probe-seed values; c s0==s1?')
for m in range(1,7):
    for ts in range(3):
        x1=rec(m,0,1,ts,'c_action_only_linear','all'); x0=rec(m,0,0,ts,'c_action_only_linear','all')
        print(m,ts,'c all s1 mean %.4f sd %.4f min %.4f max %.4f; identical s0=%s'%(np.mean(x1),np.std(x1,ddof=1),min(x1),max(x1),np.allclose(x0,x1)))
print('\n== T6: a margin; a early vs late; per-level recall (s1, late)')
for m in range(1,7):
    for ts in range(3):
        r=B[str(m)]['records_depth_m_int']['strength_1'][str(ts)]['a_level_logistic']
        e=r['early']['train']['mean']; l=r['late']['train']
        ex=r['extras'] if 'extras' in r else None
        r0=B[str(m)]['records_depth_m_int']['strength_0'][str(ts)]['a_level_logistic']
        print(m,ts,'s1 late %.3f early %.3f recall %s C %s | s0 late %.3f early %.3f'%(l['mean'],e,{k:round(v,3) for k,v in l['per_level_recall'].items()},ex['C'] if ex else None,r0['late']['train']['mean'],r0['early']['train']['mean']))
print('\n== T7: aux m_int+1 (train split, late) s1')
for m in range(1,6):
    for meas in ['a_level_logistic','b_mlp','b_attentive','c_action_only_linear','d_linear_pooled']:
        sp='train' if meas=='a_level_logistic' else 'all'
        v=[np.mean(rec(m,0,1,ts,meas,sp,aux=True)) for ts in range(3)]
        print(m,meas,np.round(v,3))
print('\n== T8 d repro')
for m in range(1,7):
    d=B[str(m)]['d_reproduction_vs_009_at_m_int']['by_strength']
    print(m,{s:{ts:{sp:round(x['diff_012_minus_009'],3) for sp,x in d[s][ts].items()} for ts in d[s]} for s in d} if m==1 else '')
