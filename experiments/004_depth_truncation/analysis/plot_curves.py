"""Plot primary curve + controls + mask baseline per split. Run with lab07 python (matplotlib needed)."""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.abspath(__file__)); rows=list(csv.DictReader(open(f"{H}/../results/depth_r2.csv")))
def line(mod,ctrl,s,tp="late",probe="linear",fit="new"):
    g={}
    for r in rows:
        if (r["module"],r["probe"],r["fit"],r["control"],r["timepoint"],r["level"],r["split"])==(mod,probe,fit,ctrl,tp,"ALL",s): g.setdefault(int(r["depth"]),[]).append(float(r["r2"]))
    d=sorted(g); return d,[np.mean(g[x]) for x in d],[np.std(g[x],ddof=1) for x in d]
fig,ax=plt.subplots(2,3,figsize=(14,7))
for j,s in enumerate(("train","interp","extrap")):
    for i,mod in enumerate(("encoder","predictor")):
        a=ax[i,j]
        for ctrl,c in (("real","C0"),("random_init","C1"),("label_shuffle","C2")):
            d,m,sd=line(mod,ctrl,s); a.errorbar(d,m,sd,label=ctrl,color=c,marker="o",ms=3)
        mk=line("baseline","mask_area_bbox",s)[1][0]; px=line("baseline","pixel",s)[1][0]
        a.axhline(mk,ls="--",c="k",label="mask_area_bbox"); a.axhline(px,ls=":",c="gray",label="pixel")
        a.set_title(f"{mod} {s} (linear/new/late)"); a.set_ylim(-0.3,1)
ax[0,0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{H}/fig_primary_vs_controls.png",dpi=110)
