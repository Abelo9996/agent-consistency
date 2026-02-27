import json, os
from src.analysis.generate_figures import load_results, results_to_dataframe

results = load_results()
df = results_to_dataframe(results)
print("Valid results:", len(df))
print("Models:", df["model"].unique().tolist())
print("Tasks:", df["task_id"].nunique())
print()
print(df.groupby("model")[["seq_similarity", "arg_consistency", "output_match_rate"]].mean().round(3).to_string())
print()
print("By category:")
print(df.groupby("category")[["seq_similarity", "arg_consistency"]].mean().round(3).to_string())
print()
print("By difficulty:")
print(df.groupby("difficulty")[["seq_similarity", "arg_consistency"]].mean().round(3).to_string())
print()
# Divergence points
dp = df["divergence_point"].dropna()
print(f"Mean divergence point: {dp.mean():.2f}")
print(f"Median divergence point: {dp.median():.2f}")
print(f"Pct at step 1-2: {(dp <= 2).mean():.1%}")
