"""Plot supplied saved reports; do not imply a shared-set model ranking."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
verified = json.loads((root/"results/saved_artifact_verification.json").read_text())
historical = json.loads((root/"results/thesis_reported.json").read_text())
reports = verified["reports"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig,axes=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={"width_ratios":[1.7,1]})
fig.patch.set_facecolor("#fafbfd")
names=["Hierarchical\n1,207 test posts","Flat BERT\n604 development posts"]
for offset,metric,color,label in [(-.17,"micro","#238878","Micro-F1"),(.17,"macro","#58718e","Macro-F1")]:
    vals=[reports[k][metric]["f1-score"] for k in ["hierarchical","flat_bert"]]
    ys=[i+offset for i in range(2)]
    axes[0].barh(ys,vals,height=.28,color=color,label=label)
    for y,v in zip(ys,vals):axes[0].text(v+.012,y,f"{v:.4f}",va="center",fontsize=10)
axes[0].set_yticks([0,1],names)
axes[0].set_ylim(1.70,-.45);axes[0].set_xlim(0,1.03)
axes[0].set_xlabel("Saved report F1")
axes[0].set_title("Different evaluation sets",loc="left",weight="bold",pad=16)
axes[0].legend(loc="lower left",frameon=False,ncol=2,fontsize=9)
counts=[historical["default_threshold"]["fp_assignments"],reports["hierarchical"]["inferred_fp"]]
axes[1].bar(["Default 0.5", "Calibrated"],counts,color=["#a9b7c8","#238878"],width=.55)
axes[1].set_ylim(0,1550)
axes[1].set_ylabel("False-positive label assignments")
axes[1].set_title("Reported calibration counts",loc="left",weight="bold",pad=16)
for i,v in enumerate(counts):axes[1].text(i,v+28,f"{v:,}",ha="center",weight="bold")
for ax in axes:
    ax.set_facecolor("#fafbfd")
    for side in ["top","right"]:ax.spines[side].set_visible(False)
    for side in ["left","bottom"]:ax.spines[side].set_color("#d7dee7")
fig.suptitle("Emotion detection | Saved evaluation artifacts",x=.02,ha="left",fontsize=17,weight="bold")
fig.text(.02,.05,"Sources: supplied evaluation CSVs. Neural inference not rerun; models used different evaluation sets.",color="#526174",fontsize=9)
fig.text(.02,.02,"Calibration: 1,313 FP is thesis-reported; 1,202 FP is consistent with the saved calibrated CSV.",color="#526174",fontsize=9)
fig.tight_layout(rect=[0,.13,1,.92]);(root/"assets").mkdir(exist_ok=True)
fig.savefig(root/"assets/reported_results.png",dpi=160)
