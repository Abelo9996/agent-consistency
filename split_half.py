import json, numpy as np
from glob import glob
from scipy import stats

results = []
for f in sorted(glob("results/experiment_*.json")):
    results.extend(json.load(open(f)))

def seq_sim(seqs):
    """Token-level TSS over tool-name sequences, matching the paper's
    Definition (src/metrics/consistency.py operates on trace objects; this
    applies the same pairwise normalized-edit-distance to raw sequences).
    An earlier version used character-level Levenshtein, a different metric."""
    from itertools import combinations
    seqs = [tuple(x) for x in seqs]
    if len(seqs) < 2:
        return 1.0
    sims = []
    for s1, s2 in combinations(seqs, 2):
        m = max(len(s1), len(s2))
        if m == 0:
            sims.append(1.0); continue
        # token-level edit distance (DP)
        dp = list(range(len(s2) + 1))
        for i, a in enumerate(s1, 1):
            prev, dp[0] = dp[0], i
            for j, b in enumerate(s2, 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (a != b))
        sims.append(1 - dp[-1] / m)
    return float(np.mean(sims))

odd_vals, even_vals = [], []
for r in results:
    if "error" in r.get("metrics", {}):
        continue
    traces = r.get("traces", [])
    if len(traces) < 10:
        continue
    odd = [t["tool_calls"] for i, t in enumerate(traces) if i % 2 == 0]
    even = [t["tool_calls"] for i, t in enumerate(traces) if i % 2 == 1]
    odd_seqs = [[c["tool_name"] for c in tc] for tc in odd]
    even_seqs = [[c["tool_name"] for c in tc] for tc in even]
    odd_vals.append(seq_sim(odd_seqs))
    even_vals.append(seq_sim(even_seqs))

r_val, p_val = stats.pearsonr(odd_vals, even_vals)
print(f"Split-half TSS: r={r_val:.3f}, p={p_val:.2e}, n={len(odd_vals)}")
rs, ps = stats.spearmanr(odd_vals, even_vals)
print(f"Split-half TSS (Spearman): rho={rs:.3f}, p={ps:.2e}")

# Completion by model
by_model = {}
for r in results:
    m = r.get("model", "?")
    if m not in by_model:
        by_model[m] = {"completed": 0, "total": 0}
    for t in r.get("traces", []):
        by_model[m]["total"] += 1
        if t.get("final_response") and not t.get("error"):
            by_model[m]["completed"] += 1

print("\nCompletion rates by model:")
for m, v in sorted(by_model.items()):
    rate = v["completed"] / v["total"] if v["total"] else 0
    print(f"  {m}: {v['completed']}/{v['total']} = {rate:.1%}")
