"""Regenerate paper tables from the fresh (July 2026) trace collection."""
import sys; sys.path.insert(0, '.')
from src.analysis.generate_figures import load_results, results_to_dataframe
df = results_to_dataframe(load_results())

# Table: model comparison
rows = []
for m, g in df.groupby('model'):
    rows.append((m, g['seq_similarity'].mean(), g['arg_consistency'].mean(),
                 g['unique_sequences'].mean(), g['output_match_rate'].mean()))
rows.sort(key=lambda r: -r[1])
with open('paper/figures/table_model_comparison.tex', 'w') as f:
    f.write("""\\begin{table}[t]
\\centering
\\caption{Per-model consistency (fresh July-2026 collection; 19 tasks $\\times$ 10 runs
per model, version-pinned via a provider-normalized API). Models ordered by \\tss.}
\\label{tab:models}
\\small
\\begin{tabular}{lcccc}
\\toprule
Model & $\\tss$ & $\\ac$ & Uniq.\\ Seq. & Output Match \\\\
\\midrule
""")
    for m, tss, ac, u, om in rows:
        f.write(f"{m} & {tss:.2f} & {ac:.2f} & {u:.1f}/10 & {om:.0%} \\\\\n".replace('%','\\%'))
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# Table: category
with open('paper/figures/table_category.tex', 'w') as f:
    f.write("""\\begin{table}[t]
\\centering
\\caption{Consistency by task category (fresh collection, all models pooled).}
\\label{tab:category}
\\small
\\begin{tabular}{lcccc}
\\toprule
Category & $\\tss$ & $\\ac$ & Div.\\ Pt. & Tasks \\\\
\\midrule
""")
    for c, g in df.groupby('category'):
        dp = g['divergence_point'].dropna()
        f.write(f"{c.capitalize()} & {g['seq_similarity'].mean():.2f} & {g['arg_consistency'].mean():.2f} & "
                f"{dp.mean():.1f} & {g['task_id'].nunique()} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# Table: per-task (gpt-4o-mini rows as before)
sub = df[df['model'] == 'gpt-4o-mini'].sort_values(['category', 'task_id'])
with open('paper/figures/table_results.tex', 'w') as f:
    f.write("""\\begin{table}[t]
\\centering
\\caption{Per-task consistency metrics (fresh collection; 10 runs per task,
gpt-4o-mini). Seq.\\ Sim.\\ = tool sequence similarity, Arg.\\ Con.\\ = argument
consistency.}
\\label{tab:results}
\\small
\\begin{tabular}{llccccc}
\\toprule
Task & Category & Diff. & Seq.\\ Sim. & Arg.\\ Con. & Uniq.\\ Seq. & Div.\\ Pt. \\\\
\\midrule
""")
    for _, r in sub.iterrows():
        dp = f"{r['divergence_point']:.1f}" if r['divergence_point'] == r['divergence_point'] else '-'
        f.write(f"{r['task_id']} & {r['category']} & {r['difficulty']} & {r['seq_similarity']:.2f} & "
                f"{r['arg_consistency']:.2f} & {int(r['unique_sequences'])}/10 & {dp} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("3 tables regenerated from fresh data")
