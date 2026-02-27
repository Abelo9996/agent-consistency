import json, os

path = "results"
for f in sorted(os.listdir(path)):
    d = json.load(open(os.path.join(path, f)))
    by_model = {}
    for r in d:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"done": 0, "failed": 0}
        has_err = "error" in r.get("metrics", {})
        if has_err:
            by_model[m]["failed"] += 1
        else:
            by_model[m]["done"] += 1
    parts = []
    for m, v in by_model.items():
        parts.append(f"{m}={v['done']}ok/{v['failed']}fail")
    print(f"{f}: {', '.join(parts)}")
