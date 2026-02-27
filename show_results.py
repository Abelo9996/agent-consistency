import json

with open('results/experiment_20260225_194752.json') as f:
    results = json.load(f)

print(f'Total task-model combos: {len(results)}')
print()
print(f'{"Task":<16} {"Model":<13} {"SeqSim":>6} {"ArgCon":>6} {"Uniq":>6} {"DivPt":>6} {"OutMatch":>8}')
print('-' * 70)

for r in results:
    m = r['metrics']
    tid = r['task_id']
    mod = r['model']
    if 'error' in m:
        print(f'{tid:<16} {mod:<13} ERR')
    else:
        ss = m['tool_sequence_similarity']
        ac = m['argument_consistency']
        uq = f"{m['unique_sequences']}/{m['n_runs']}"
        dp = f"{m['divergence_point']:.1f}" if m['divergence_point'] else "-"
        om = f"{m['output_agreement']['exact_match_rate']:.0%}"
        print(f'{tid:<16} {mod:<13} {ss:>6.2f} {ac:>6.2f} {uq:>6} {dp:>6} {om:>8}')
