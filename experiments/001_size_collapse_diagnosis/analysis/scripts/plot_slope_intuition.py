# 기울기 직관 그림: 크기 9종의 (실제 면적, 모델 면적) 산점도. 값은 results.json 에서만 읽는다.
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open("results.json"))["runs"]; seeds=list(R)
G=["a026","a040","a060","a080","ind","a125","a160","a200","a248"]
def pts(s,key,k=12):
    g=R[s]["diagnose"]["area"]["groups"]
    return np.array([g[x]["by_k"][str(k)]["real"]["mean"] for x in G]),np.array([g[x]["by_k"][str(k)][key]["mean"] for x in G])
fig,ax=plt.subplots(1,3,figsize=(16,5.2),sharex=True,sharey=True)
cm=plt.cm.viridis(np.linspace(0.05,0.9,9)); mk=["o","s","^"]
panels=[("recon","decoder only: decode(encode(real))","recon@k12"),("tf1","teacher-forced 1-step","tf1@k12"),("ol","open-loop rollout","ol@k12")]
for a,(key,title,sk) in zip(ax,panels):
    lo=min(pts(seeds[0],"real")[0]); hi=max(pts(seeds[0],"real")[0])
    xs=np.linspace(lo*0.8,hi*1.25,50)
    a.plot(xs,xs,color="#999",ls="--",lw=1.2,label="slope 1 (model = real)")
    sl=[]
    for i,s in enumerate(seeds):
        x,y=pts(s,key); a.scatter(x,y,c=cm,marker=mk[i],s=55,edgecolor="white",lw=.8,zorder=3)
        p=np.polyfit(np.log(x),np.log(y),1); sl.append(p[0])
        if i==0: a.plot(xs,np.exp(p[1])*xs**p[0],color="#d1603d",lw=2.2,label="fitted line (seed 0)")
    a.set_xscale("log");a.set_yscale("log");a.set_title(title,fontsize=11,loc="left")
    a.set_xlabel("real cloth area (px, size mean)")
    a.text(.04,.93,"fitted slope (seeds 0/1/2):\n"+" / ".join(f"{v:.2f}" for v in sl),transform=a.transAxes,va="top",fontsize=10)
    if key=="ol":
        sj=[R[s]["diagnose"]["size_curve"]["slope_model_vs_real_group_means"][sk]["slope"] for s in seeds]
        print("check vs results.json ol@k12:",np.round(sl,3),np.round(sj,3))
ax[0].set_ylabel("model-produced cloth area (px)")
ax[0].legend(loc="lower right",frameon=False,fontsize=9)
# 크기 색 범례
for c,gname in zip(cm,G): ax[2].scatter([],[],color=c,label=gname)
ax[2].legend(loc="lower right",frameon=False,fontsize=8,ncol=3,title="size (dark=small, light=large)")
fig.suptitle("Each dot = one cloth size (9 per seed, k=12). Slope 1 = model keeps size differences; slope→0 = every size drawn the same",fontsize=11,x=0.01,ha="left")
fig.tight_layout(rect=(0,0,1,.94)); fig.savefig("analysis/fig4_slope_intuition.png",dpi=130)
