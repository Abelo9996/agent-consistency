import json, numpy as np
from glob import glob
from scipy import stats

results = []
for f in sorted(glob("results/experiment_*.json")):
    results.extend(json.load(open(f)))

from Levenshtein import distance as lev_dist

def seq_sim(seqs):
    if len(seqs) < 2:
        return 1.0
    sims = []
    for i in range(len(seqs)):
        for j in range(i+1, len(seqs)):
            s1, s2 = " ".join(seqs[i]), " ".join(seqs[j])
            maxlen = max(len(s1), len(s2))
            if maxlen == 0:
                sims.append(1.0)
            else:
                sims.append(1 - lev_dist(s1, s2) / maxlen)
    return np.mean(sims)

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
