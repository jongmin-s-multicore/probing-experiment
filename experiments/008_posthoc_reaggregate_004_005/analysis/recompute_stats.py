"""008 독립 재계산 (csv+numpy; 008 run.py 와 코드 공유 없음). 실행: lab07 python. 출력: analysis/recompute_stats.txt"""
import csv, json, math, numpy as np, yaml
from collections import defaultdict, Counter
R='/home/jaekyungkim/mylab/experiments/'
rows=list(csv.DictReader(open(R+'004_depth_truncation/results/depth_r2.csv')))
res=json.load(open(R+'008_posthoc_reaggregate_004_005/results.json'))
print(Counter((r['module'],r['control'],r['level']=='ALL') for r in rows))
print('train_seeds',sorted({r['train_seed'] for r in rows}),'probe seeds',sorted({r['seed'] for r in rows}))
print('depths enc',sorted({int(r['depth']) for r in rows if r['module']=='encoder'}),'pred',sorted({int(r['depth']) for r in rows if r['module']=='predictor'}))
print('probes',Counter(r['probe'] for r in rows),'fit',Counter(r['fit'] for r in rows),'tp',Counter(r['timepoint'] for r in rows))
B=[r for r in rows if r['probe']=='linear' and r['fit']=='new' and r['timepoint']=='late']
nl={}
for r in B:
    if r['level']!='ALL': nl[(r['level'],r['split'])]=int(r['n_episodes'])
print('levels',nl)
def fd(ds,vs,thr=.2):
    m=None
    for d,v in zip(ds,vs):
        if m is not None and m-v>=thr-1e-12: return (d,round(v,3),round(m,3))
        m=v if m is None else max(m,v)
def get(mod,ctl,split):
    D=defaultdict(dict)
    for r in B:
        if r['module']==mod and r['control']==ctl and r['level']=='ALL' and r['split']==split:
            D[int(r['depth'])][(r['train_seed'],r['seed'])]=float(r['r2'])
    return D
print("== A")
for mod in('encoder','predictor'):
  for s in('train','interp','extrap'):
    re=get(mod,'real',s); rn=get(mod,'random_init',s); ds=sorted(re)
    mean=[];lo=[];sdl=[];rm=[];xm=[]
    for d in ds:
        dl={k:re[d][k]-rn[d][k] for k in re[d]}
        if mod=='predictor':
            g=defaultdict(list)
            for (ts,ps),v in dl.items(): g[ts].append(v)
            u=[np.mean(v) for v in g.values()]
        else: u=list(dl.values())
        sd=np.std(u,ddof=1); mean.append(np.mean(list(dl.values()))); sdl.append(sd); lo.append(np.mean(u)-2*sd)
        rm.append(np.mean(list(re[d].values()))); xm.append(np.mean(list(rn[d].values())))
    jr=res['items']['A_control_relative_curve'][mod]['by_split'][s]
    md=max(abs(jr['delta_curve'][i]['delta_mean']-mean[i]) for i in range(len(ds)))
    ml=max(abs(jr['delta_curve'][i]['mean_minus_2sd']-lo[i]) for i in range(len(ds)))
    print(f'\n{mod} {s}: maxdiff vs json delta {md:.1e} lo {ml:.1e}; firstdrop delta={fd(ds,mean)} real={fd(ds,rm)}; n_gt0 recomputed={sum(l>0 for l in lo)} json={jr["n_depths_delta_gt0"]}')
    print(' depth  real  rand  delta  sd   lo   delta/sd')
    for i,d in enumerate(ds): print(f' {d:2d} {rm[i]:6.3f} {xm[i]:6.3f} {mean[i]:6.3f} {sdl[i]:6.3f} {lo[i]:6.3f} {mean[i]/sdl[i]:7.1f}')
print("\n== B")
c002=yaml.safe_load(open(R+'002_single_size_sweep/config.yaml'))
def find(o):
    if isinstance(o,dict):
        if 'a026' in o and isinstance(o['a026'],(int,float)): return o
        for v in o.values():
            f=find(v)
            if f: return f
area=find(c002); print('area',area)
lab={k:math.log(v) for k,v in area.items()}
lvs=sorted({l for l,_ in nl}); sp={l:s for (l,s) in nl}
n={l:nl[(l,sp[l])] for l in lvs}
N=sum(n.values()); mu=sum(n[l]*lab[l] for l in lvs)/N; Va=sum(n[l]*(lab[l]-mu)**2 for l in lvs)/N
print('Var_all',Va,'json',res['items']['B_unified_scale']['detail']['common_label_variance_all_eval_episodes'])
for s in('train','interp','extrap'):
    ls=[l for l in lvs if sp[l]==s]; ns=sum(n[l] for l in ls); m=sum(n[l]*lab[l] for l in ls)/ns
    v=sum(n[l]*(lab[l]-m)**2 for l in ls)/ns
    print(s,ls,'n',ns,'split var',round(v,4),'split label range',[round(lab[l],3) for l in ls],'ratio var_split/var_all',round(v/Va,3))
# independent check of reconstruction: per row, MSE_split from level rows
mod='predictor'
cur=defaultdict(lambda: defaultdict(list))
for ctl in('real',):
  for mod in('encoder','predictor'):
    keyed=defaultdict(dict)
    for r in B:
        if r['module']==mod and r['control']==ctl: keyed[(int(r['depth']),r['seed'],r['train_seed'])][(r['level'],r['split'])]=r
    for k,dct in keyed.items():
        for s in('train','interp','extrap'):
            ls=[l for l in lvs if sp[l]==s]; ns=sum(n[l] for l in ls)
            mse=sum(n[l]*(1-float(dct[(l,s)]['r2']))*Va for l in ls)/ns
            cur[(mod,s)][k[0]].append((mse,float(dct[('ALL',s)]['mae']),float(dct[('ALL',s)]['r2'])))
    for s in('train','interp','extrap'):
        ds=sorted(cur[(mod,s)]); 
        mse=[np.mean([x[0] for x in cur[(mod,s)][d]]) for d in ds]
        mae=[np.mean([x[1] for x in cur[(mod,s)][d]]) for d in ds]
        r2s=[np.mean([x[2] for x in cur[(mod,s)][d]]) for d in ds]
        r2c=[1-m/Va for m in mse]
        jr=res['items']['B_unified_scale']['modules'][mod]['curves_by_control_split_depth']['real'][s]
        md=max(abs(jr[str(d)]['r2_common']-r2c[i]) for i,d in enumerate(ds))
        print(f'\n{mod} {s} maxdiff r2_common vs json {md:.1e}; firstdrop common={fd(ds,r2c)} split={fd(ds,r2s)}')
        print(' depth  mae   mse   r2split r2common')
        for i,d in enumerate(ds): print(f' {d:2d} {mae[i]:6.3f} {mse[i]:6.3f} {r2s[i]:7.3f} {r2c[i]:7.3f}')
print("\n== C")
for s in('interp','extrap'):
  for mod in('encoder','predictor'):
    g=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in B:
        if r['module']==mod and r['control']=='real' and r['level']=='ALL': g[r['split']][int(r['depth'])][r['train_seed']].append(float(r['r2']))
    ds=sorted(g['train']); ins=0; out=[]
    for d in ds:
        if mod=='predictor':
            tr=[np.mean(v) for v in g['train'][d].values()]; oo=[np.mean(v) for v in g[s][d].values()]
        else:
            tr=[x for v in g['train'][d].values() for x in v]; oo=[x for v in g[s][d].values() for x in v]
        tol=2*max(np.std(tr,ddof=1),np.std(oo,ddof=1)); gap=np.mean(tr)-np.mean(oo)
        ins+=abs(gap)<=tol; out.append((d,round(gap,3),round(tol,3)))
    print(mod,s,'inside',ins,'/',len(ds),out)
print('json C predictor', {s:res['items']['C_overlap_recompute_train_seed_units']['predictor']['by_split'][s]['fraction_depths_inside'] for s in('interp','extrap')}, 'ref004', res['reference_004_summary_ind_ood_gap_fractions'])
