import json

with open('results/experiment_20260225_223352.json') as f:
    results = json.load(f)

print(f'o1 results: {len(results)} task-model combos')
errs = sum(1 for r in results if 'error' in r['metrics'])
ok = len(results) - errs
print(f'OK: {ok}, ERR: {errs}')
print()
for r in results:
    m = r['metrics']
    tid = r['task_id']
    mod = r['model']
    if 'error' in m:
        err = m.get('error', '')[:60]
        print(f'  {tid:<16} ERR: {err}')
    else:
        ss = m['tool_sequence_similarity']
        ac = m['argument_consistency']
        uq = m['unique_sequences']
        nr = m['n_runs']
        dp = m.get('divergence_point')
        dp_s = f'{dp:.1f}' if dp else '-'
        om = m['output_agreement']['exact_match_rate']
        print(f'  {tid:<16} SeqSim={ss:.2f} ArgCon={ac:.2f} Uniq={uq}/{nr} DivPt={dp_s} OutMatch={om:.0%}')
