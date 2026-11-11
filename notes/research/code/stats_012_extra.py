import json,numpy as np
exec(open('notes/research/code/stats_012.py').read().split("# 1 recompute")[0].split("out=[]")[0])
def r2(m,s,ts,meas,sp,win): return np.mean(rec(m,0,s,ts,meas,sp,win=win))
print('early window s1: a / mlp / att / c(all)')
for m in range(1,7):
    print(m,[round(r2(m,1,ts,'a_level_logistic','train','early'),3) for ts in range(3)],[round(r2(m,1,ts,'b_mlp','all','early'),3) for ts in range(3)],[round(r2(m,1,ts,'b_attentive','all','early'),3) for ts in range(3)],[round(r2(m,1,ts,'c_action_only_linear','all','early'),3) for ts in range(3)])
print('retained fraction s1/s0 (late, all): mlp, att')
for m in range(1,7):
    print(m,round(np.mean([r2(m,1,ts,'b_mlp','all','late')/r2(m,0,ts,'b_mlp','all','late') for ts in range(3)]),3),round(np.mean([r2(m,1,ts,'b_attentive','all','late')/r2(m,0,ts,'b_attentive','all','late') for ts in range(3)]),3))
# a: z vs chance with n=300 episodes (rough, binomial; independence across episodes assumed)
for m in range(1,7):
    v=[r2(m,1,ts,'a_level_logistic','train','late') for ts in range(3)]
    se=np.sqrt(.25/300)
    print(m,'a mean',round(np.mean(v),3),'z vs chance(1/3, se_binom@n=300 %.3f)'%np.sqrt((1/3)*(2/3)/300),round((np.mean(v)-1/3)/np.sqrt((1/3)*(2/3)/300),1),'gap to .5333:',np.round(np.array(v)-0.5333333,3),'SE@p=.5',round(se,3))
# C sensitivity for a: cv_logloss
for m in (6,):
  for ts in range(3):
    ex=B[str(m)]['records_depth_m_int']['strength_1'][str(ts)]['a_level_logistic']['extras']
    print('blk6 ts',ts,'C',ex['C'],{k:round(v) for k,v in ex['cv_logloss_by_C'].items()},'train_acc',round(ex['train_accuracy'],3))
# c threshold sensitivity
for thr in (0.15,0.2,0.25,0.3):
    print('c thr',thr,[int(sum(np.mean(rec(m,0,1,ts,'c_action_only_linear','all'))>=thr for ts in range(3))) for m in range(1,7)])
for thr in (0.5,0.55,0.6):
    print('a thr',thr,[int(sum(np.mean(rec(m,0,1,ts,'a_level_logistic','train'))>=thr for ts in range(3))) for m in range(1,7)])
# rank/trend a with m
import itertools
am=[np.mean([r2(m,1,ts,'a_level_logistic','train','late') for ts in range(3)]) for m in range(1,7)]
print('a block means',np.round(am,3))
# paired: attentive vs mlp at s1
