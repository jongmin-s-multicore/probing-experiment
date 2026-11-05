import json,re,html,sys,hashlib
R='/home/jaekyungkim/mylab/'
raw=open(R+'experiments/002_single_size_sweep/results.json','rb').read()
d=json.loads(raw); sha=hashlib.sha256(raw).hexdigest()
rows={r['id']:r for r in d['decision_table']}
ROLE={'interp':'Interp','extrap':'Extrap'}; PROBE={'linear_pooled':'linear','attentive':'attentive'}
ha=[]
for role in ('interp','extrap'):
    for pr in ('linear_pooled','attentive'):
        r=rows[f'H_a:{role}:{pr}']; x=r['result']
        ha.append(dict(label=f'{ROLE[role]} {PROBE[pr]}',real=x['r2_real'],rand=x['r2_random_init'],mask=x['r2_mask_baseline'],delta=x['delta_vs_mask_baseline'],ci=x['delta_vs_mask_baseline_ci95'],drand=x['delta_vs_random_init'],cirand=x['delta_vs_random_init_ci95'],status=r['status'],holm=r['holm_adjusted_p'],seeds=x['r2_real_per_probe_seed']))
early=[dict(label=f"{ROLE[e['role']]} {PROBE[e['probe']]}",real=e['r2_real'],rand=e['r2_random_init'],mask=e['r2_mask_baseline'],delta=e['delta_vs_mask_baseline']) for e in d['h_a_early_flat_report']['rows']]
G=d['exploratory']['q2_layer_time_grid_r2_point_estimates']
roll={}
pre='log_nominal_area_ratio|linear_pooled|OUT|'
for c in ('multi_60_ind_160','single_ind'):
    roll[c]={}
    for s in '012':
        g=G[f'{c}/seed{s}']
        roll[c][s]={k:{'enc':[g[pre+f'encoder|step{i}'][k] for i in range(1,13)],'out':[g[pre+f'rollout|step{i}'][k] for i in range(1,13)]} for k in ('unseen','trained')}
layer={}
for c in ('multi_60_ind_160','single_ind'):
    p=d['q2_layerwise']['conditions'][c]['per_seed']
    layer[c]={s:{k:[p[s][str(l)][k]['change_vs_layer0'] for l in range(7)] for k in ('unseen','trained')} for s in '012'}
    if c=='multi_60_ind_160': layer_abs={s:[p[s][str(l)]['unseen']['r2'] for l in range(7)] for s in '012'}
steer=rows['H_c:steering']['result']['cells']
steer=[dict(cell=k,T=v['T_size_direction'],rnd=v['random_percentile_value'],out=v['outside_percentile'],rho=v['spearman'],p=v['p_rank'],n=v['n_random']) for k,v in steer.items()]
diag={}
for c in ('single_ind','multi_60_ind_160'):
    r=rows['diag_4c:'+c]['result']; diag[c]=dict(verdict=r['verdict_worst_over_seeds'],seeds={})
    for s,p in r['per_training_seed'].items():
        diag[c]['seeds'][s]={l:dict(ratio=[x['per_layer'][str(i)]['ratio_to_noise_floor'] for i in range(7)],maxr=x['max_ratio_over_layers'],err=x['next_frame_error_ratio_to_ind'],cls=x['activation_class']) for l,x in p['levels'].items()}
hb={}
for k in ('H_b:last_block','H_b:last_rollout_step'):
    x=rows[k]['result']; hb[k]=dict(status=rows[k]['status'],holm=rows[k]['holm_adjusted_p'],mean=x['drop_mean_over_seeds'],ci=x['drop_ci95_episode_bootstrap'],seeds={s:v for s,v in x['per_training_seed'].items()})
ca=rows['cond_a_vs_b:last_block']; cb=rows['cond_a_vs_b:last_rollout_step']
cond=dict(block=dict(status=ca['status'],holm=ca['holm_adjusted_p'],diff=ca['result']['change_vs_layer0_a_minus_b'],ci=ca['result']['ci95_episode_bootstrap']),roll=dict(status=cb['status'],holm=cb['holm_adjusted_p'],diff=cb['result']['drop_a_minus_drop_b'],ci=cb['result']['ci95_episode_bootstrap'],a=cb['result']['per_training_seed_a'],b=cb['result']['per_training_seed_b']))
rem=rows['H_c:subspace_removal']
# reviewer table: copied verbatim from report section 6
rep=open(R+'reports/002_single_size_sweep.md',encoding='utf8').read()
sec=rep[rep.index('## 6.'):rep.index('## 7.')]
rv=[[c.strip() for c in l.strip().strip('|').split('|')] for l in sec.splitlines() if l.startswith('|')][2:]
TR={'H_a 보간 linear':'H_a interp linear','H_a 보간 attentive':'H_a interp attentive','H_a 외삽 linear':'H_a extrap linear','H_a 외삽 attentive':'H_a extrap attentive','아니오':'No','조건부로 쓸 수 있음. DINOv2 특이적 증거는 약함':'Usable conditionally; DINOv2-specific evidence is weak','강함(이 가설 중 유일하게 "강함")':'Strong (the only "strong" within this hypothesis)','경계(CI 하한 0.015)':'Borderline (CI lower bound 0.015)','약함':'Weak','정보 없음의 뜻이 아님(구조적 불리)':'Does not mean no information (structural disadvantage)','유의하지만 작음. 마진 미달은 신뢰 가능':'Significant but small; missing the margin is reliable','제한적으로 예':'Yes, with limits','probe 외삽 한계가 섞였을 가능성(추정)':'May include probe extrapolation limits (estimate)','baseline 보다 나쁨(-0.125)':'Worse than baseline (-0.125)','약함(probe 일반화 실패가 섞임)':'Weak (mixed with probe generalization failure)','약한 증거(실패하기 어려움)':'Weak evidence (hard to fail)','약하게 지지(정보량 적음)':'Weak support (little information)','하락 방향은 일관, 원인은 분리 못함':'Drop direction is consistent; cause not separated','높음(시드 단위로도 견고)':'High (robust at seed level too)','하락은 예, 소실은 미정':'Drop: yes; information loss: undetermined','반쪽 증거':'Partial evidence','기준 충족이나 변별력 약함':'Meets the criterion; weak discrimination','아니오(무효)':'No (invalid)','결론 없음':'No conclusion','통계적 검출일 뿐 실질 차이 없음':'Statistical detection only; no practical difference','시드 운으로 설명될 가능성 높음':'Likely explained by seed luck','차이는 방어 가능, 인과는 아님':'Difference is defensible; causation is not','낮은 우연 가능성(견고)':'Low chance of coincidence (robust)','차이는 예, 원인은 미정':'Difference: yes; cause: undetermined','(a) 층별 결과 해석 시 주의 문구 필요':'Caution note needed when interpreting (a) layer-wise results','변별력 낮음. 시드 하나의 2% 안에서 갈림':'Low discrimination; split within 2% on one seed','임계 바로 위(0.04-0.30). 규칙 문구와 대상이 어긋남':'Just above the threshold (0.04-0.30); rule wording targets a different condition','변별력 낮음':'Low discrimination'}
rv=[[TR.get(c,c) for c in r] for r in rv]
data=dict(ha=ha,early=early,roll=roll,layer=layer,layer_abs=layer_abs,steer=steer,diag=diag,hb=hb,cond=cond,rv=rv,rem_note=rem['note'],sha=sha[:12],created=d['created'],steer_status=rows['H_c:steering']['status'],steer_holm=rows['H_c:steering']['holm_adjusted_p'],ci_caveat=d['ci_caveat'],lim=d['q2_layerwise']['limitation'])
t=open(sys.argv[1],encoding='utf8').read().replace('/*DATA*/null',json.dumps(data,ensure_ascii=False))
open(sys.argv[2],'w',encoding='utf8').write(t)
# sanity prints
print('ceiling step12 (unseen):',[round(roll['multi_60_ind_160'][s]['unseen']['enc'][11],4) for s in '012'],[round(roll['single_ind'][s]['unseen']['enc'][11],4) for s in '012'])
print('rollout step12:',[round(roll['multi_60_ind_160'][s]['unseen']['out'][11],4) for s in '012'],[round(roll['single_ind'][s]['unseen']['out'][11],4) for s in '012'])
allv=[v for c in roll.values() for s in c.values() for k in s.values() for a in k.values() for v in a]; print('roll range',min(allv),max(allv))
lv=[v for c in layer.values() for s in c.values() for a in s.values() for v in a]; print('layer range',min(lv),max(lv))
print('rv rows',len(rv),rv[0]); print('bytes',len(t))
