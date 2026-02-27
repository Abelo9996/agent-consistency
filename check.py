import json, glob
for f in sorted(glob.glob("results/experiment_*.json")):
    data = json.load(open(f))
    print(f"{f}: {len(data)} tasks")
    for r in data:
        m = r["metrics"]
        if "error" in m:
            continue
        tid = r["task_id"]
        ss = m["tool_sequence_similarity"]
        ac = m["argument_consistency"]
        us = m["unique_sequences"]
        nr = m["n_runs"]
        print(f"  {tid:15s} seq={ss:.2f} arg={ac:.2f} uniq={us}/{nr}")
