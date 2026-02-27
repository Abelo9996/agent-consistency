import json

d = json.load(open("results/experiment_20260226_124510_partial.json"))
by_model = {}
for r in d:
    m = r["model"]
    if m not in by_model:
        by_model[m] = []
    by_model[m].append(r["task_id"])

for m, tasks in by_model.items():
    print(f"{m}: {len(tasks)} tasks")

last = d[-1]
print(f"\nLast completed: {last['model']} / {last['task_id']}")

errs = [t for t in last.get("traces", []) if t.get("error")]
print(f"Errors in last task: {len(errs)}")
if errs:
    print(errs[0].get("error", "")[:300])
