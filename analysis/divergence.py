"""Re-derive the early-divergence statistic from the released raw traces.

Definition (mirrors src/metrics/consistency.py:divergence_point):
  For every unordered pair of the 10 runs within a condition, walk the two tool
  sequences position by position. The first position where the tool NAMES differ
  (a missing position on the shorter sequence counts as a difference) is the
  first-divergence step, 1-indexed. If the tool names agree at every position
  (including equal lengths), we then walk the arguments over the shared prefix
  and take the first position whose argument dicts differ. Pairs that are
  identical in both tool names and arguments contribute NO event.
"""
import json, sys
from itertools import combinations
from collections import Counter
import numpy as np

path = 'results/experiment_20260705_205523.json'
data = json.load(open(path))

events = []            # (step, condition_idx)
per_cond = []
n_pairs_total = 0
n_pairs_identical = 0
n_pairs_len_differ = 0
n_pairs_prefix_extension = 0   # names agree on shared prefix, lengths differ
lengths = []           # (model, task, category, n_tool_calls)

for ci, cond in enumerate(data):
    traces = cond['traces']
    seqs = []
    for t in traces:
        tcs = t.get('tool_calls') or []
        seq = [c['tool_name'] for c in tcs]
        args = [c.get('arguments') for c in tcs]
        seqs.append((seq, args))
        lengths.append((cond['model'], cond['task_id'], cond['category'], len(seq)))
    cond_events = []
    for (s1, a1), (s2, a2) in combinations(seqs, 2):
        n_pairs_total += 1
        if len(s1) != len(s2):
            n_pairs_len_differ += 1
        step = None
        maxlen = max(len(s1), len(s2))
        for i in range(maxlen):
            t1 = s1[i] if i < len(s1) else None
            t2 = s2[i] if i < len(s2) else None
            if t1 != t2:
                step = i + 1
                if t1 is None or t2 is None:
                    n_pairs_prefix_extension += 1
                break
        if step is None:
            for i in range(min(len(s1), len(s2))):
                if a1[i] != a2[i]:
                    step = i + 1
                    break
        if step is None:
            n_pairs_identical += 1
        else:
            cond_events.append(step)
            events.append((step, ci))
    per_cond.append(cond_events)

steps = np.array([e[0] for e in events])
n = len(steps)
print('conditions:', len(data))
print('traces:', sum(len(c['traces']) for c in data))
print('total run pairs:', n_pairs_total, '(= C(10,2)*114 =', 45*114, ')')
print('pairs identical in tools AND args (excluded):', n_pairs_identical)
print('pairs of unequal length:', n_pairs_len_differ)
print('pairs where divergence is a length extension (shared prefix identical):', n_pairs_prefix_extension)
print()
print('=== POOLED OVER ALL PAIRS (event-level denominator) ===')
print('N first-divergence events:', n)
print('mean step: %.3f  median: %.1f' % (steps.mean(), np.median(steps)))
for k in (1, 2, 3):
    print('  step <= %d: %d / %d = %.4f  (%.1f%%)' % (k, (steps <= k).sum(), n, (steps <= k).mean(), 100*(steps <= k).mean()))
print('  histogram:', sorted(Counter(steps.tolist()).items()))
print()
print('=== CONDITION-LEVEL (each condition weighted equally) ===')
fr = [np.mean(np.array(c) <= 2) for c in per_cond if c]
print('conditions with >=1 divergence event:', len(fr), 'of', len(per_cond))
print('mean of per-condition fraction(step<=2): %.4f (%.1f%%)' % (np.mean(fr), 100*np.mean(fr)))
print()
print('=== mean divergence_point stored in the artifact ===')
dps = [c['metrics']['divergence_point'] for c in data if c['metrics'].get('divergence_point') is not None]
print('n conditions with a value:', len(dps), ' mean of condition means: %.3f' % np.mean(dps))
