"""Debug: check what went wrong with o1 on failed tasks."""
import json

with open('results/experiment_20260225_223352.json') as f:
    results = json.load(f)

for r in results:
    m = r['metrics']
    if 'error' in m:
        tid = r['task_id']
        traces = r['traces']
        valid = sum(1 for t in traces if not t.get('error'))
        errored = sum(1 for t in traces if t.get('error'))
        print(f'\n{tid}: {valid} valid, {errored} errored')
        for t in traces[:3]:
            if t.get('error'):
                print(f'  ERR: {t["error"][:120]}')
            else:
                print(f'  OK: {len(t["tool_calls"])} tool calls')
