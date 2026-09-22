"""Reproduce Table 6 (tab:lengths) from the released traces.

Recomputes every cell of the trace-length distribution and length-normalized
divergence table, grouped by model, by category, and overall ("All").

Conventions reused verbatim from scripts/verify.py (trace length = number of
tool calls) and scripts/divergence.py (first-divergence step over within-
condition run pairs, mirroring src/metrics/consistency.py:divergence_point):

  For every unordered pair of the 10 runs within a condition, walk the two
  tool-NAME sequences position by position; the first position where the names
  differ (a missing position on the shorter sequence counts as a difference) is
  the first-divergence step, 1-indexed. If the names agree everywhere (including
  equal lengths), walk the arguments over the shared prefix and take the first
  step whose argument dicts differ. Pairs identical in both names and arguments
  contribute NO divergence event (excluded from divergence stats).

Per-group columns:
  Len.Mean / Len.Med = mean / median number of tool calls over the group's
    individual traces (runs).
  Div.step = mean first-divergence step over the group's diverging run pairs.
  Norm = mean over diverging pairs of (first-divergence step / longer trace
    length), longer trace length = max(len(seq_i), len(seq_j)).
  <=2(%) = share of the group's diverging pairs with first-divergence step in {1,2}.

Grouping: a run pair belongs to the group(s) of ITS condition. "by model" pools
all conditions for that model; "by category" pools all conditions in that
category; "All" pools everything.
"""
import json
from itertools import combinations
from statistics import mean, median

PATH = 'results/experiment_20260705_205523.json'

MODELS = ['claude-sonnet-4', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-4o-mini', 'llama-3.3-70b']
CATEGORIES = ['retrieval', 'scheduling', 'computation', 'composition', 'ambiguous']

MODEL_LABEL = {
    'claude-sonnet-4': 'Claude Sonnet 4',
    'gpt-4.1': 'GPT-4.1',
    'gpt-4.1-mini': 'GPT-4.1-mini',
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o-mini',
    'llama-3.3-70b': 'Llama 3.3 70B',
}
CAT_LABEL = {c: c.capitalize() for c in CATEGORIES}

# Paper Table 6 targets: label -> (Len.Mean, Len.Med, Div.step, Norm, <=2%)
PAPER = {
    'Claude Sonnet 4': (3.21, 3, 3.02, 0.71, 47),
    'GPT-4.1':         (2.20, 2, 2.41, 0.83, 65),
    'GPT-4.1-mini':    (2.12, 2, 2.22, 0.76, 66),
    'GPT-4o':          (2.53, 2, 2.01, 0.73, 74),
    'GPT-4o-mini':     (2.83, 2, 2.17, 0.73, 63),
    'Llama 3.3 70B':   (2.53, 2, 2.17, 0.72, 68),
    'Retrieval':       (2.25, 2, 1.82, 0.66, 94),
    'Scheduling':      (1.74, 1, 1.72, 0.76, 90),
    'Computation':     (3.39, 2, 2.41, 0.73, 61),
    'Composition':     (4.19, 3, 3.41, 0.83, 16),
    'Ambiguous':       (1.48, 1, 1.69, 0.71, 81),
    'All':             (2.57, 2, 2.31, 0.74, 64),
}


def first_divergence(seq1, args1, seq2, args2):
    """Return the 1-indexed first-divergence step, or None if identical."""
    maxlen = max(len(seq1), len(seq2))
    for i in range(maxlen):
        t1 = seq1[i] if i < len(seq1) else None
        t2 = seq2[i] if i < len(seq2) else None
        if t1 != t2:
            return i + 1
    for i in range(min(len(seq1), len(seq2))):
        if args1[i] != args2[i]:
            return i + 1
    return None


def main():
    data = json.load(open(PATH))

    # Per-group accumulators.
    # trace lengths (one entry per run) and divergence events (one per diverging pair).
    lengths = {}   # group -> list of trace lengths
    steps = {}     # group -> list of first-divergence steps
    norms = {}     # group -> list of (step / longer length)

    def groups_for(cond):
        return [('model', cond['model']), ('category', cond['category']), ('all', 'All')]

    for g in [('model', m) for m in MODELS] + [('category', c) for c in CATEGORIES] + [('all', 'All')]:
        lengths[g] = []
        steps[g] = []
        norms[g] = []

    for cond in data:
        gs = groups_for(cond)
        seqs = []
        for t in cond['traces']:
            tcs = t.get('tool_calls') or []
            seq = [c['tool_name'] for c in tcs]
            args = [c.get('arguments') for c in tcs]
            seqs.append((seq, args))
            for g in gs:
                lengths[g].append(len(seq))
        for (s1, a1), (s2, a2) in combinations(seqs, 2):
            step = first_divergence(s1, a1, s2, a2)
            if step is None:
                continue
            longer = max(len(s1), len(s2))
            for g in gs:
                steps[g].append(step)
                norms[g].append(step / longer)

    def row(g):
        L = lengths[g]
        S = steps[g]
        N = norms[g]
        len_mean = mean(L)
        len_med = median(L)
        div_step = mean(S) if S else float('nan')
        norm = mean(N) if N else float('nan')
        leq2 = 100.0 * (sum(1 for s in S if s <= 2) / len(S)) if S else float('nan')
        return len_mean, len_med, div_step, norm, leq2

    # Ordered rows: 6 models, 5 categories, All.
    ordered = ([('model', m) for m in MODELS]
               + [('category', c) for c in CATEGORIES]
               + [('all', 'All')])

    def label(g):
        kind, key = g
        if kind == 'model':
            return MODEL_LABEL[key]
        if kind == 'category':
            return CAT_LABEL[key]
        return 'All'

    header = f"{'Group':<18} {'Len.Mean':>9} {'Len.Med':>8} {'Div.step':>9} {'Norm':>6} {'<=2(%)':>7}"
    print('=' * len(header))
    print('Table 6 (tab:lengths) reproduced from ' + PATH)
    print('=' * len(header))
    print(header)
    print('-' * len(header))

    computed = {}
    for g in ordered:
        lm, lmed, ds, nm, l2 = row(g)
        lab = label(g)
        computed[lab] = (lm, lmed, ds, nm, l2)
        print(f"{lab:<18} {lm:>9.2f} {lmed:>8.1f} {ds:>9.2f} {nm:>6.2f} {l2:>6.0f}%")

    # Verification block.
    print()
    print('=' * 72)
    print('VERIFICATION vs paper Table 6 (tol: +/-0.01 on decimals, +/-1 pp on <=2%)')
    print('=' * 72)
    print(f"{'Group':<18} {'col':<9} {'computed':>10} {'paper':>8} {'match':>7}")
    print('-' * 60)
    colnames = ['Len.Mean', 'Len.Med', 'Div.step', 'Norm', '<=2(%)']
    all_match = True
    mismatches = []
    for g in ordered:
        lab = label(g)
        comp = computed[lab]
        pap = PAPER[lab]
        for i, cn in enumerate(colnames):
            cv = comp[i]
            pv = pap[i]
            if cn == '<=2(%)':
                ok = abs(cv - pv) <= 1.0 + 1e-9
                cvs, pvs = f"{cv:.0f}", f"{pv:.0f}"
            elif cn == 'Len.Med':
                ok = abs(cv - pv) <= 0.01 + 1e-9
                cvs, pvs = f"{cv:.1f}", f"{pv:.1f}"
            else:
                ok = abs(cv - pv) <= 0.01 + 1e-9
                cvs, pvs = f"{cv:.2f}", f"{pv:.2f}"
            if not ok:
                all_match = False
                mismatches.append((lab, cn, cvs, pvs))
            print(f"{lab:<18} {cn:<9} {cvs:>10} {pvs:>8} {'OK' if ok else 'MISMATCH':>8}")

    print()
    if all_match:
        print('RESULT: ALL CELLS MATCH the paper Table 6 within tolerance.')
    else:
        print(f'RESULT: {len(mismatches)} CELL(S) DO NOT MATCH:')
        for lab, cn, cvs, pvs in mismatches:
            print(f'  {lab} / {cn}: computed {cvs} vs paper {pvs}')


if __name__ == '__main__':
    main()
