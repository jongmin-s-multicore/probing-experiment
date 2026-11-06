# results.json / decoder_ratio_table.csv 에서만 값을 읽어 요약 그래프를 그린다.
import json, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
M = json.load(open("results.json"))["across_seeds"]["baseline_single"]["metrics"]
sizes = ["a026","a040","a060","a080","ind","a125","a160","a200","a248"]
area = [0.259,0.399,None,None,None,None,None,None,2.479]  # 라벨용 아님; x는 순번 사용
ks = [1,2,3,4,6,8,12]
def g(s,k): return M[f"diagnose/area/groups/{s}/by_k/{k}/ratio_ol_over_real/mean"]
INK="#222"; MUT="#777"; C1="#2a6fb0"; C2="#d1603d"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
x=np.arange(len(sizes))
# (1) open-loop k=12
a=ax[0]; m=[g(s,12)["mean"] for s in sizes]
a.axhline(1,color=MUT,lw=1,ls="--"); a.bar(x,m,color=C1,width=.6,alpha=.85)
for i,s in enumerate(sizes):
    v=list(g(s,12)["by_seed"].values()); a.scatter([i]*3,v,s=14,color=INK,zorder=3)
a.set_xticks(x,sizes,rotation=45); a.set_yscale("log"); a.set_ylabel("open-loop area ratio (model / real), k=12")
a.set_title("(1) Open-loop area ratio drifts from 1 away from train size (ind)\nbar = seed mean, dots = seeds 0/1/2, dashed = 1",fontsize=10,loc="left")
# (2) by k
a=ax[1]
for s,c in [("a026",C2),("a248",C1)]:
    mu=[g(s,k)["mean"] for k in ks]; lo=[g(s,k)["min"] for k in ks]; hi=[g(s,k)["max"] for k in ks]
    a.plot(ks,mu,color=c,lw=2,marker="o",ms=5,label=s); a.fill_between(ks,lo,hi,color=c,alpha=.18)
a.axhline(1,color=MUT,lw=1,ls="--"); a.set_yscale("log"); a.set_xlabel("rollout step k"); a.set_ylabel("model / real area")
a.legend(frameon=False); a.set_title("(2) Error accumulates with rollout step k (band = seed min-max)",fontsize=10,loc="left")
# (3) decoder recon ratio
a=ax[2]; rows=list(csv.DictReader(open("analysis/decoder_ratio_table.csv")))
a.axhspan(0.9,1.1,color=C1,alpha=.12,label="pre-registered band [0.9, 1.1]"); a.axhline(1,color=MUT,lw=1,ls="--")
for sd,mk in [("s0","o"),("s1","s"),("s2","^")]:
    v=[float([r for r in rows if r["group"]==s and r["seed"].endswith(sd)][0]["recon_over_real_k12_MoR"]) for s in sizes]
    a.scatter(x,v,marker=mk,s=28,color=INK,label=f"seed {sd[1]}",zorder=3)
a.set_xticks(x,sizes,rotation=45); a.set_ylim(0.8,1.2); a.set_ylabel("decode(encode(real)) / real, k=12")
a.legend(frameon=False,fontsize=8,loc="upper right"); a.set_title("(3) Decoder recon ratio: all 27 cells inside band (point est.)",fontsize=10,loc="left")
fig.tight_layout(); fig.savefig("analysis/fig0_summary.png",dpi=140)
