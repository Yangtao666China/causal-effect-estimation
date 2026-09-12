"""Publication-friendly figures generated from the same report data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .estimators import LABELS


COLORS = {"naive":"#9ba8b5","linear":"#307ab5","hgb":"#00866b"}


def make_plots(report,output):
    output = Path(output)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"#f7f9fb","axes.facecolor":"#f7f9fb"})
    fig,axes = plt.subplots(1,len(report["datasets"]),figsize=(6.4*len(report["datasets"]),4.8),
                            squeeze=False,layout="constrained")
    for ax,dataset in zip(axes[0],report["datasets"]):
        rows = [r for r in report["results"] if r["dataset"] == dataset["id"]]
        for i,row in enumerate(rows):
            ax.errorbar(row["estimate"],i,xerr=[[row["estimate"]-row["ci_low"]],
                         [row["ci_high"]-row["estimate"]]],fmt="o",capsize=5,color=COLORS[row["method"]],markersize=8)
        if report["benchmark"]:
            reference = report["benchmark"]
            ax.axvspan(reference["ci_low"],reference["ci_high"],color="#d5e6ee",alpha=.4,label="RCT reference 95% interval")
            ax.axvline(reference["estimate"],color="#46687a",linestyle="--",label="RCT point reference")
        ax.axvline(0,color="#7b8692",linewidth=.7)
        ax.set(yticks=range(len(rows)),yticklabels=[r["label"] for r in rows],xlabel="Income difference (source units)",
               title=f"{dataset['label']}\n{dataset['treated']:,} treated / {dataset['controls']:,} controls",ylim=(-.6,len(rows)-.4))
        ax.invert_yaxis()
    if report["benchmark"]:
        axes[0][-1].legend(loc="best",fontsize=8)
    fig.suptitle("Same treated group. Different comparison data.\nApproximate 95% intervals; randomized reference is not ground truth.",fontsize=14)
    fig.savefig(output/"effects.png",dpi=160)
    plt.close(fig)

    target = "observational" if any(d["id"] == "observational" for d in report["datasets"]) else report["datasets"][0]["id"]
    diag = [d for d in report["diagnostics"] if d["dataset"] == target]
    if diag:
        fig,axes = plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
        for item in diag:
            values = item["balance"]
            if item is diag[0]:
                axes[0].scatter([abs(b["before"]) if b["before"] is not None else np.nan for b in values],range(len(values)),color=COLORS["naive"],marker="x",label="Before")
            axes[0].scatter([abs(b["after"]) if b["after"] is not None else np.nan for b in values],range(len(values)),color=COLORS[item["method"]],label=LABELS[item["method"]])
        axes[0].set(yticks=range(len(values)),yticklabels=[b["feature"] for b in values],xlabel="Absolute standardized mean difference",title="Balance after ATT odds weighting")
        axes[0].axvline(.1,color="#b1bac2",linestyle="--",linewidth=.8)
        axes[0].legend(fontsize=8)
        for method in ("linear","hgb"):
            rows = [r for r in report["sensitivity"] if r["dataset"] == target and r["method"] == method]
            axes[1].plot([r["clip"] for r in rows],[r["estimate"] for r in rows],"o-",color=COLORS[method],label=LABELS[method])
        axes[1].set(xscale="log",xlabel="Propensity clipping threshold",ylabel="ATT estimate (source units)",title="Clipping sensitivity | primary fold seed")
        axes[1].legend(fontsize=8)
        fig.suptitle("Diagnostics before conclusions",fontsize=15)
        fig.savefig(output/"diagnostics.png",dpi=160)
        plt.close(fig)

    if report["simulation"]:
        fig,axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
        scenarios = list(dict.fromkeys(r["scenario"] for r in report["simulation"]))
        for ax,scenario in zip(axes,scenarios):
            rows = [r for r in report["simulation"] if r["scenario"] == scenario]
            ax.bar(range(len(rows)),[r["rmse"] for r in rows],color=[COLORS[r["method"]] for r in rows])
            ax.set(xticks=range(len(rows)),xticklabels=[r["method"] for r in rows],ylabel="RMSE of ATT estimate",title=scenario.replace("_"," ").title())
        fig.suptitle(f"Known-effect simulation | {report['simulation'][0]['repetitions']} repetitions per scenario",fontsize=14)
        fig.savefig(output/"simulation.png",dpi=160)
        plt.close(fig)
