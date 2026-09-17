"""Regenerate Figure 10 (correctness vs consistency) from the RECOVERED REAL
per-condition data (paper/figures/correctness_analysis.json), replacing the
earlier version of this figure which was illustratively synthesized. Every
point below is a real (task, model) condition."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("paper/figures/correctness_analysis.json"))
rows = d["per_task_model"]
tss = np.array([r["tss"] for r in rows])
ac = np.array([r["ac"] for r in rows])
cor = np.array([r["correctness_rate"] for r in rows])
n = len(rows)
total_traces = sum(r["n_traces"] for r in rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, x, name, stats in [
    (axes[0], tss, "Tool Sequence Similarity (TSS)", d["correlations"]["tss_pearson"]),
    (axes[1], ac, "Argument Consistency (AC)", d["correlations"]["ac_pearson"]),
]:
    ax.scatter(x, cor, s=28, alpha=0.65, edgecolors="k", linewidths=0.4)
    m, b = np.polyfit(x, cor, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, "r--", lw=1.5)
    ax.set_xlabel(name)
    ax.set_title(f"r = {stats['r']:.2f}, p = {stats['p']:.3f}  (n = {n} conditions)")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Correctness rate")
fig.suptitle(f"Real per-condition data: {n} (task, model) conditions, {total_traces} traces", fontsize=9)
fig.tight_layout()
fig.savefig("paper/figures/fig10_correctness_vs_consistency.pdf", bbox_inches="tight")
fig.savefig("paper/figures/fig10_correctness_vs_consistency.png", dpi=200, bbox_inches="tight")
print(f"fig10 regenerated from REAL data: {n} conditions, {total_traces} traces")
# verify correlations reproduce from the raw points
from math import sqrt
r_tss = np.corrcoef(tss, cor)[0, 1]
r_ac = np.corrcoef(ac, cor)[0, 1]
print(f"verify: tss r={r_tss:.4f} (paper 0.323)  ac r={r_ac:.4f} (paper 0.119)")
