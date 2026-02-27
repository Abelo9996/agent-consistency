"""Compute additional statistical tests for the paper revision."""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from src.analysis.generate_figures import load_results, results_to_dataframe

results = load_results()
df = results_to_dataframe(results)
df_main = df[df["model"] != "claude-haiku"].copy()

print("=" * 60)
print("STATISTICAL TESTS FOR PAPER REVISION")
print("=" * 60)

# 1. Paired t-test: TSS vs AC
t_stat, p_val = stats.ttest_rel(df_main["seq_similarity"], df_main["arg_consistency"])
d = (df_main["seq_similarity"] - df_main["arg_consistency"]).mean() / (df_main["seq_similarity"] - df_main["arg_consistency"]).std()
print(f"\n1. TSS vs AC (paired t-test):")
print(f"   t={t_stat:.3f}, p={p_val:.2e}, Cohen's d={d:.3f}")

# 2. 95% CIs for each model
print(f"\n2. Model 95% Confidence Intervals:")
for model in df_main["model"].unique():
    sub = df_main[df_main["model"] == model]
    n = len(sub)
    tss_ci = stats.t.interval(0.95, n-1, loc=sub["seq_similarity"].mean(), scale=stats.sem(sub["seq_similarity"]))
    ac_ci = stats.t.interval(0.95, n-1, loc=sub["arg_consistency"].mean(), scale=stats.sem(sub["arg_consistency"]))
    print(f"   {model:20s}: TSS=[{tss_ci[0]:.3f}, {tss_ci[1]:.3f}]  AC=[{ac_ci[0]:.3f}, {ac_ci[1]:.3f}]  n={n}")

# 3. Effect size: ambiguous vs structured
structured = df_main[df_main["category"] != "ambiguous"]["arg_consistency"]
ambiguous = df_main[df_main["category"] == "ambiguous"]["arg_consistency"]
pooled_std = np.sqrt((structured.std()**2 * (len(structured)-1) + ambiguous.std()**2 * (len(ambiguous)-1)) / (len(structured) + len(ambiguous) - 2))
cohens_d = (structured.mean() - ambiguous.mean()) / pooled_std
t_amb, p_amb = stats.ttest_ind(structured, ambiguous)
print(f"\n3. Ambiguous vs Structured (AC):")
print(f"   Structured: {structured.mean():.3f} (n={len(structured)})")
print(f"   Ambiguous:  {ambiguous.mean():.3f} (n={len(ambiguous)})")
print(f"   Cohen's d={cohens_d:.3f}, t={t_amb:.3f}, p={p_amb:.2e}")

# Same for TSS
s_tss = df_main[df_main["category"] != "ambiguous"]["seq_similarity"]
a_tss = df_main[df_main["category"] == "ambiguous"]["seq_similarity"]
ps = np.sqrt((s_tss.std()**2 * (len(s_tss)-1) + a_tss.std()**2 * (len(a_tss)-1)) / (len(s_tss) + len(a_tss) - 2))
d_tss = (s_tss.mean() - a_tss.mean()) / ps
t_tss, p_tss = stats.ttest_ind(s_tss, a_tss)
print(f"\n   Structured TSS: {s_tss.mean():.3f}, Ambiguous TSS: {a_tss.mean():.3f}")
print(f"   Cohen's d={d_tss:.3f}, t={t_tss:.3f}, p={p_tss:.2e}")

# 4. ANOVA across models
print(f"\n4. One-way ANOVA (TSS across models):")
groups = [df_main[df_main["model"] == m]["seq_similarity"].values for m in df_main["model"].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
# eta-squared
ss_between = sum(len(g) * (g.mean() - df_main["seq_similarity"].mean())**2 for g in groups)
ss_total = sum((df_main["seq_similarity"] - df_main["seq_similarity"].mean())**2)
eta_sq = ss_between / ss_total
print(f"   F={f_stat:.3f}, p={p_anova:.2e}, eta²={eta_sq:.3f}")

print(f"\n5. One-way ANOVA (AC across models):")
groups_ac = [df_main[df_main["model"] == m]["arg_consistency"].values for m in df_main["model"].unique()]
f_ac, p_ac = stats.f_oneway(*groups_ac)
ss_b_ac = sum(len(g) * (g.mean() - df_main["arg_consistency"].mean())**2 for g in groups_ac)
ss_t_ac = sum((df_main["arg_consistency"] - df_main["arg_consistency"].mean())**2)
eta_sq_ac = ss_b_ac / ss_t_ac
print(f"   F={f_ac:.3f}, p={p_ac:.2e}, eta²={eta_sq_ac:.3f}")

# 6. Kruskal-Wallis for categories
print(f"\n6. Kruskal-Wallis (AC across categories):")
cat_groups = [df_main[df_main["category"] == c]["arg_consistency"].values for c in df_main["category"].unique()]
h_stat, p_kw = stats.kruskal(*cat_groups)
print(f"   H={h_stat:.3f}, p={p_kw:.2e}")

# 7. Exact trace count
total_traces = 0
for model in df_main["model"].unique():
    n = len(df_main[df_main["model"] == model])
    total_traces += n * 10
    print(f"   {model}: {n} tasks × 10 runs = {n*10} traces")
print(f"   Total: {total_traces} traces")

# 8. Difficulty validation: correlate assigned difficulty with mean tool calls
diff_map = {"easy": 1, "medium": 2, "hard": 3}
df_main["diff_num"] = df_main["difficulty"].map(diff_map)
r_tools, p_tools = stats.pearsonr(df_main["diff_num"], df_main["mean_tool_calls"])
print(f"\n8. Difficulty validation (correlation with mean tool calls):")
print(f"   r={r_tools:.3f}, p={p_tools:.2e}")
print(f"   Mean tool calls by difficulty:")
for d in ["easy", "medium", "hard"]:
    sub = df_main[df_main["difficulty"] == d]
    print(f"     {d}: {sub['mean_tool_calls'].mean():.1f} ± {sub['mean_tool_calls'].std():.1f}")
