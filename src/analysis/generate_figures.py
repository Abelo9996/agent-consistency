"""
Analysis and figure generation for the agent consistency paper.

Generates publication-quality figures from experiment results.
"""

import json
import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats

# Paper-quality settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,

})

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_results(results_dir: str = "results") -> list[dict]:
    """Load all experiment result files and merge."""
    files = sorted(glob(os.path.join(results_dir, "experiment_*.json")))
    if not files:
        print(f"No result files found in {results_dir}/")
        sys.exit(1)

    all_results = []
    for f in files:
        with open(f) as fh:
            all_results.extend(json.load(fh))
    print(f"Loaded {len(all_results)} task-model results from {len(files)} files")
    return all_results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Flatten results into a DataFrame for analysis."""
    rows = []
    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        rows.append({
            "task_id": r["task_id"],
            "category": r["category"],
            "difficulty": r["difficulty"],
            "model": r["model"],
            "seq_similarity": m["tool_sequence_similarity"],
            "arg_consistency": m["argument_consistency"],
            "unique_sequences": m["unique_sequences"],
            "n_runs": m["n_runs"],
            "seq_ratio": m["unique_sequences"] / m["n_runs"],
            "divergence_point": m.get("divergence_point"),
            "output_match_rate": m["output_agreement"]["exact_match_rate"],
            "unique_outputs": m["output_agreement"]["unique_responses"],
            "mean_tool_calls": m["tool_call_stats"]["mean"],
            "std_tool_calls": m["tool_call_stats"]["std"],
            "cv_tool_calls": m["tool_call_stats"]["cv"],
        })
    return pd.DataFrame(rows)


# ============================================================
# Figure 1: Consistency by Category (grouped bar chart)
# ============================================================
def fig_consistency_by_category(df: pd.DataFrame):
    """Bar chart of sequence similarity and argument consistency by task category."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, title in [
        (axes[0], "seq_similarity", "Tool Sequence Similarity"),
        (axes[1], "arg_consistency", "Argument Consistency"),
    ]:
        cat_data = df.groupby("category")[metric].agg(["mean", "std"]).reindex(
            ["retrieval", "scheduling", "computation", "composition", "ambiguous"]
        )
        colors = sns.color_palette("Set2", len(cat_data))
        bars = ax.bar(range(len(cat_data)), cat_data["mean"], yerr=cat_data["std"],
                       capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(cat_data)))
        ax.set_xticklabels([c.capitalize() for c in cat_data.index], rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Behavioral Consistency by Task Category", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig1_consistency_by_category.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig1_consistency_by_category.png")
    plt.close()
    print("  -> fig1_consistency_by_category")


# ============================================================
# Figure 2: Consistency by Difficulty
# ============================================================
def fig_consistency_by_difficulty(df: pd.DataFrame):
    """Show how consistency degrades with task difficulty."""
    fig, ax = plt.subplots(figsize=(6, 4))

    metrics = ["seq_similarity", "arg_consistency"]
    labels = ["Tool Sequence", "Arguments"]
    colors = ["#2196F3", "#FF9800"]
    difficulty_order = ["easy", "medium", "hard"]

    x = np.arange(len(difficulty_order))
    width = 0.35

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        means = [df[df["difficulty"] == d][metric].mean() for d in difficulty_order]
        stds = [df[df["difficulty"] == d][metric].std() for d in difficulty_order]
        ax.bar(x + i * width - width/2, means, width, label=label, color=color,
               yerr=stds, capsize=4, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulty_order])
    ax.set_ylabel("Consistency Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Consistency Degrades with Task Difficulty", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig2_consistency_by_difficulty.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig2_consistency_by_difficulty.png")
    plt.close()
    print("  -> fig2_consistency_by_difficulty")


# ============================================================
# Figure 3: Divergence Point Distribution
# ============================================================
def fig_divergence_points(df: pd.DataFrame):
    """Where does divergence first happen?"""
    fig, ax = plt.subplots(figsize=(6, 4))

    dp_data = df[df["divergence_point"].notna()]["divergence_point"]
    if dp_data.empty:
        print("  -> fig3 skipped (no divergence data)")
        return

    ax.hist(dp_data, bins=range(1, int(dp_data.max()) + 2), color="#4CAF50",
            edgecolor="black", linewidth=0.5, alpha=0.8, align="left")
    ax.set_xlabel("Step at Which Divergence First Occurs")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("Divergence Occurs Early in the Pipeline", fontweight="bold")
    ax.axvline(x=dp_data.mean(), color="red", linestyle="--", label=f"Mean = {dp_data.mean():.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig3_divergence_points.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig3_divergence_points.png")
    plt.close()
    print("  -> fig3_divergence_points")


# ============================================================
# Figure 4: Model Comparison (if multiple models)
# ============================================================
def fig_model_comparison(df: pd.DataFrame):
    """Compare consistency across models."""
    models = df["model"].unique()
    if len(models) < 2:
        print("  -> fig4 skipped (single model)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, title in [
        (axes[0], "seq_similarity", "Tool Sequence Similarity"),
        (axes[1], "arg_consistency", "Argument Consistency"),
    ]:
        model_data = df.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean", ascending=False)
        colors = sns.color_palette("coolwarm", len(model_data))
        ax.barh(range(len(model_data)), model_data["mean"], xerr=model_data["std"],
                capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(model_data)))
        ax.set_yticklabels(model_data.index)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1.05)

    fig.suptitle("Model Comparison: Behavioral Consistency", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig4_model_comparison.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig4_model_comparison.png")
    plt.close()
    print("  -> fig4_model_comparison")


# ============================================================
# Figure 5: Heatmap — Task × Metric
# ============================================================
def fig_heatmap(df: pd.DataFrame):
    """Heatmap showing consistency metrics per task."""
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.35)))

    # If multiple models, pick the first one for the heatmap
    if df["model"].nunique() > 1:
        model = df["model"].value_counts().index[0]
        hm_df = df[df["model"] == model].copy()
        title_suffix = f" ({model})"
    else:
        hm_df = df.copy()
        title_suffix = f" ({df['model'].iloc[0]})"

    hm_df = hm_df.set_index("task_id")[["seq_similarity", "arg_consistency", "output_match_rate"]].rename(
        columns={
            "seq_similarity": "Tool Seq.",
            "arg_consistency": "Arguments",
            "output_match_rate": "Output Match",
        }
    )

    sns.heatmap(hm_df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                ax=ax, linewidths=0.5, cbar_kws={"label": "Consistency Score"})
    ax.set_title(f"Per-Task Consistency Heatmap{title_suffix}", fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig5_heatmap.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig5_heatmap.png")
    plt.close()
    print("  -> fig5_heatmap")


# ============================================================
# Figure 6: Sequence vs Argument Consistency Scatter
# ============================================================
def fig_seq_vs_arg(df: pd.DataFrame):
    """Scatter plot: do tasks with consistent tool sequences also have consistent arguments?"""
    fig, ax = plt.subplots(figsize=(6, 5))

    categories = df["category"].unique()
    palette = dict(zip(categories, sns.color_palette("Set2", len(categories))))

    for cat in categories:
        subset = df[df["category"] == cat]
        ax.scatter(subset["seq_similarity"], subset["arg_consistency"],
                   label=cat.capitalize(), color=palette[cat], s=60, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Tool Sequence Similarity")
    ax.set_ylabel("Argument Consistency")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)  # Diagonal
    ax.legend()
    ax.set_title("Sequence vs. Argument Consistency", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig6_seq_vs_arg.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig6_seq_vs_arg.png")
    plt.close()
    print("  -> fig6_seq_vs_arg")


# ============================================================
# Table: Summary Statistics
# ============================================================
def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics for the paper."""
    print("\n=== Summary Statistics ===")
    print(f"Models tested: {', '.join(df['model'].unique())}")
    print(f"Tasks: {df['task_id'].nunique()}")
    print(f"Total task-model combos: {len(df)}")
    print(f"\nOverall Means:")
    print(f"  Tool Sequence Similarity: {df['seq_similarity'].mean():.3f} (std {df['seq_similarity'].std():.3f})")
    print(f"  Argument Consistency:     {df['arg_consistency'].mean():.3f} (std {df['arg_consistency'].std():.3f})")
    print(f"  Output Exact Match Rate:  {df['output_match_rate'].mean():.3f} (std {df['output_match_rate'].std():.3f})")

    dp = df["divergence_point"].dropna()
    if not dp.empty:
        print(f"  Mean Divergence Point:    Step {dp.mean():.1f}")

    print(f"\nBy Category:")
    for cat in ["retrieval", "scheduling", "computation", "composition", "ambiguous"]:
        subset = df[df["category"] == cat]
        if subset.empty:
            continue
        print(f"  {cat:15s}: seq_sim={subset['seq_similarity'].mean():.2f}  arg_con={subset['arg_consistency'].mean():.2f}  n={len(subset)}")

    print(f"\nBy Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        subset = df[df["difficulty"] == diff]
        if subset.empty:
            continue
        print(f"  {diff:8s}: seq_sim={subset['seq_similarity'].mean():.2f}  arg_con={subset['arg_consistency'].mean():.2f}  n={len(subset)}")

    # Correlation: difficulty → consistency
    diff_map = {"easy": 1, "medium": 2, "hard": 3}
    df_corr = df.copy()
    df_corr["diff_num"] = df_corr["difficulty"].map(diff_map)
    r_seq, p_seq = stats.pearsonr(df_corr["diff_num"], df_corr["seq_similarity"])
    r_arg, p_arg = stats.pearsonr(df_corr["diff_num"], df_corr["arg_consistency"])
    print(f"\nCorrelation (difficulty vs consistency):")
    print(f"  Seq Sim:  r={r_seq:.3f}, p={p_seq:.4f}")
    print(f"  Arg Con:  r={r_arg:.3f}, p={p_arg:.4f}")


# ============================================================
# LaTeX Table
# ============================================================
def generate_latex_table(df: pd.DataFrame):
    """Generate a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-task consistency metrics (10 runs per task, gpt-4o-mini). Seq.\ Sim.\ = tool sequence similarity, Arg.\ Con.\ = argument consistency.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Category & Diff. & Seq.\ Sim. & Arg.\ Con. & Uniq.\ Seq. & Div.\ Pt. \\")
    lines.append(r"\midrule")

    for _, row in df.sort_values(["category", "difficulty"]).iterrows():
        dp = f"{row['divergence_point']:.1f}" if pd.notna(row["divergence_point"]) else "--"
        lines.append(
            f"{row['task_id']} & {row['category']} & {row['difficulty']} & "
            f"{row['seq_similarity']:.2f} & {row['arg_consistency']:.2f} & "
            f"{row['unique_sequences']}/{row['n_runs']} & {dp} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)
    with open(f"{FIGURE_DIR}/table_results.tex", "w") as f:
        f.write(table_str)
    print(f"  -> table_results.tex")
    return table_str


# ============================================================
# Main
# ============================================================
def main():
    results = load_results()
    df = results_to_dataframe(results)

    if df.empty:
        print("No valid results to analyze!")
        sys.exit(1)

    print(f"\nGenerating figures...")
    fig_consistency_by_category(df)
    fig_consistency_by_difficulty(df)
    fig_divergence_points(df)
    fig_model_comparison(df)
    fig_heatmap(df)
    fig_seq_vs_arg(df)

    print(f"\nGenerating tables...")
    generate_latex_table(df)

    print_summary_stats(df)


if __name__ == "__main__":
    main()
