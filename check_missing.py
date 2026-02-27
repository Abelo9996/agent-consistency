import json

d = json.load(open("results/experiment_20260226_124510_partial.json"))
from src.tasks.definitions import TASKS

all_task_ids = [t["id"] for t in TASKS]
done = {}
for r in d:
    m = r["model"]
    if m not in done:
        done[m] = set()
    done[m].add(r["task_id"])

for m in ["claude-sonnet-4"]:
    missing = [t for t in all_task_ids if t not in done.get(m, set())]
    print(f"{m} missing: {missing}")
