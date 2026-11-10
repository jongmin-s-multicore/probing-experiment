"""추가 점검: (1) 규칙 도달 가능성(최대 관측 낙폭), (2) 학습시드 n=3 t-구간, (3) MSE 단위 임계 환산, (4) C sd 추정 불확실성, (5) 2sd 규칙의 등가 t. """
import csv, numpy as np
from collections import defaultdict
from scipy import stats
R='/home/jaekyungkim/mylab/experiments/'
rows=[r for r in csv.DictReader(open(R+'004_depth_truncation/results/depth_r2.csv')) if r['probe']=='linear' and r['fit']=='new' and r['timepoint']=='late' and r['level']=='ALL']
def cur(mod,ctl,s):
    g=defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['module']==mod and r['control']==ctl and r['split']==s: g[int(r['depth'])][r['train_seed']].append(float(r['r2']))
    return g
print('== 최대 관측 낙폭(러닝맥스 - 이후 최솟값), Δ 곡선 평균 기준')
for mod in('encoder','predictor'):
  for s in('train','interp','extrap'):
    re,rn=cur(mod,'real',s),cur(mod,'random_init',s)
    d=[np.mean([x for v in re[k].values() for x in v])-np.mean([x for v in rn[k].values() for x in v]) for k in sorted(re)]
    md=max(max(d[:i+1])-d[i] for i in range(len(d)))
    print(f'{mod} {s}: max Δ={max(d):.3f}, max drawdown={md:.3f} (임계 0.2)')
print('\n== predictor Δ: 학습시드 3개 단위 t 구간 (df=2)')
for s in('train','interp','extrap'):
    re,rn=cur('predictor','real',s),cur('predictor','random_init',s)
    for k in (3,6):
        u=np.array([np.mean(re[k][t])-np.mean(rn[k][t]) for t in sorted(re[k])])
        m,sd=u.mean(),u.std(ddof=1); se=sd/np.sqrt(3); tc=stats.t.ppf(.975,2)
        t=m/se; p=2*stats.t.sf(abs(t),2)
        print(f'{s} d={k}: units={np.round(u,4)} mean={m:.4f} sd={sd:.4f} 95%CI=[{m-tc*se:.4f},{m+tc*se:.4f}] t={t:.2f} p2={p:.3f} dz={m/sd:.2f}')
print('\n2sd 규칙 등가: mean-2sd>0 <=> t>2*sqrt3 = %.2f ; df=2 단측 p=%.3f ; 양측 p=%.3f'%(2*np.sqrt(3),stats.t.sf(2*np.sqrt(3),2),2*stats.t.sf(2*np.sqrt(3),2)))
print('Bonferroni 21 비교(7깊이x3 split) 단측 95pct 요구 t df=2: %.2f'%stats.t.ppf(1-.05/21,2))
print('\n== 임계 0.2 의 MSE 환산(log 면적비^2): split R² 기준 0.2*Var_split, 공통 R² 기준 0.2*Var_all')
Va=0.49999293541242407
for s,v in(('train',0.161),('interp',0.0493),('extrap',0.9655)): print(s,'split R² 0.2 ->ΔMSE',round(.2*v,4),'| 공통 R² 0.2 -> ΔMSE',round(.2*Va,4))
print('\n== C: n=3 sd 추정의 95pct 신뢰구간 배율 (chi2 df=2): sd 참값/표본sd ∈ [%.2f, %.2f]'%(np.sqrt(2/stats.chi2.ppf(.975,2)),np.sqrt(2/stats.chi2.ppf(.025,2))))
print('\n== C 대안: 고정 동등 여유 ±0.05 (R² 단위) 안에 드는 깊이 수, predictor')
g=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    if r['module']=='predictor' and r['control']=='real': g[r['split']][int(r['depth'])][r['train_seed']].append(float(r['r2']))
for s in('interp','extrap'):
    gaps=[np.mean([np.mean(v) for v in g['train'][d].values()])-np.mean([np.mean(v) for v in g[s][d].values()]) for d in range(7)]
    print(s,np.round(gaps,3),'inside ±0.05:',sum(abs(x)<=.05 for x in gaps),'/7')
