"""Additional figures for the agent consistency paper."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis.generate_figures import load_results, results_to_dataframe

plt.rcParams.update({
    "font.size": 11, "font.family": "serif", "axes.labelsize": 12,
    "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 300, "savefig.dpi": 300,
})

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

results = load_results()
df = results_to_dataframe(results)

# Filter out claude-haiku (only 4 valid tasks)
df_main = df[df["model"] != "claude-haiku"].copy()

# ============================================================
# Figure 7: Model x Category Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df_main.pivot_table(values="seq_similarity", index="model", columns="category", aggfunc="mean")
pivot = pivot.reindex(columns=["retrieval", "scheduling", "computation", "composition", "ambiguous"])
pivot = pivot.sort_index()
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.4, vmax=1.0,
            ax=ax, linewidths=0.5, cbar_kws={"label": "Sequence Similarity"})
ax.set_title("Tool Sequence Similarity: Model × Category", fontweight="bold")
ax.set_ylabel(""); ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/fig7_model_category_heatmap.pdf")
plt.savefig(f"{FIGURE_DIR}/fig7_model_category_heatmap.png")
plt.close()
print("  -> fig7_model_category_heatmap")

# ============================================================
# Figure 8: Unique sequences distribution by model
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
models_ordered = df_main.groupby("model")["seq_ratio"].mean().sort_values().index.tolist()
box_data = [df_main[df_main["model"] == m]["unique_sequences"].values for m in models_ordered]
bp = ax.boxplot(box_data, labels=models_ordered, patch_artist=True, vert=True)
colors = sns.color_palette("coolwarm", len(models_ordered))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Unique Tool Sequences (out of 10)")
ax.set_title("Behavioral Diversity by Model", fontweight="bold")
ax.set_xticklabels(models_ordered, rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/fig8_unique_seqs_by_model.pdf")
plt.savefig(f"{FIGURE_DIR}/fig8_unique_seqs_by_model.png")
plt.close()
print("  -> fig8_unique_seqs_by_model")

# ============================================================
# Figure 9: Arg consistency vs Seq similarity colored by category (multi-model)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
markers = {"gpt-4o-mini": "o", "gpt-4o": "s", "gpt-4.1-mini": "^", "gpt-4.1": "D",
           "claude-sonnet-4": "P", "llama-3.3-70b": "*", "o1": "X"}
categories = df_main["category"].unique()
palette = dict(zip(categories, sns.color_palette("Set2", len(categories))))

for model in df_main["model"].unique():
    sub = df_main[df_main["model"] == model]
    for cat in categories:
        ss = sub[sub["category"] == cat]
        if ss.empty:
            continue
        ax.scatter(ss["seq_similarity"], ss["arg_consistency"],
                   marker=markers.get(model, "o"), color=palette[cat],
                   s=70, edgecolors="black", linewidth=0.3, alpha=0.7)

# Legend for categories
from matplotlib.lines import Line2D
cat_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[c],
               markersize=8, label=c.capitalize()) for c in categories]
model_handles = [Line2D([0], [0], marker=markers.get(m, "o"), color="w",
                markerfacecolor="gray", markersize=8, label=m) for m in markers if m in df_main["model"].unique()]

l1 = ax.legend(handles=cat_handles, title="Category", loc="lower left", fontsize=8)
ax.add_artist(l1)
ax.legend(handles=model_handles, title="Model", loc="lower right", fontsize=7)

ax.set_xlabel("Tool Sequence Similarity")
ax.set_ylabel("Argument Consistency")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
ax.set_title("Sequence vs. Argument Consistency (All Models)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/fig9_scatter_all_models.pdf")
plt.savefig(f"{FIGURE_DIR}/fig9_scatter_all_models.png")
plt.close()
print("  -> fig9_scatter_all_models")

# ============================================================
# LaTeX: Multi-model summary table
# ============================================================
summary = df_main.groupby("model").agg(
    seq_sim_mean=("seq_similarity", "mean"),
    seq_sim_std=("seq_similarity", "std"),
    arg_con_mean=("arg_consistency", "mean"),
    arg_con_std=("arg_consistency", "std"),
    output_match=("output_match_rate", "mean"),
    unique_seqs=("unique_sequences", "mean"),
    n_tasks=("task_id", "count"),
).round(3)

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Cross-model consistency comparison (19 tasks, 10 runs each). TSS = Tool Sequence Similarity, AC = Argument Consistency.}")
lines.append(r"\label{tab:model_comparison}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lcccc}")
lines.append(r"\toprule")
lines.append(r"Model & TSS ($\mu \pm \sigma$) & AC ($\mu \pm \sigma$) & Output Match & Uniq.\ Seq. \\")
lines.append(r"\midrule")
for model, row in summary.sort_values("seq_sim_mean", ascending=False).iterrows():
    lines.append(f"{model} & ${row['seq_sim_mean']:.2f} \\pm {row['seq_sim_std']:.2f}$ & "
                 f"${row['arg_con_mean']:.2f} \\pm {row['arg_con_std']:.2f}$ & "
                 f"{row['output_match']:.1%} & {row['unique_seqs']:.1f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(f"{FIGURE_DIR}/table_model_comparison.tex", "w") as f:
    f.write("\n".join(lines))
print("  -> table_model_comparison.tex")

# Category breakdown table
cat_summary = df_main.groupby("category").agg(
    seq_sim=("seq_similarity", "mean"),
    arg_con=("arg_consistency", "mean"),
    unique_seqs=("unique_sequences", "mean"),
).round(3)
lines2 = []
lines2.append(r"\begin{table}[t]")
lines2.append(r"\centering")
lines2.append(r"\caption{Consistency metrics by task category (averaged across all models).}")
lines2.append(r"\label{tab:category}")
lines2.append(r"\small")
lines2.append(r"\begin{tabular}{lccc}")
lines2.append(r"\toprule")
lines2.append(r"Category & TSS & AC & Uniq.\ Seq. \\")
lines2.append(r"\midrule")
for cat in ["retrieval", "scheduling", "computation", "composition", "ambiguous"]:
    if cat in cat_summary.index:
        row = cat_summary.loc[cat]
        lines2.append(f"{cat.capitalize()} & {row['seq_sim']:.2f} & {row['arg_con']:.2f} & {row['unique_seqs']:.1f} \\\\")
lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table}")

with open(f"{FIGURE_DIR}/table_category.tex", "w") as f:
    f.write("\n".join(lines2))
print("  -> table_category.tex")

print("\nDone!")
