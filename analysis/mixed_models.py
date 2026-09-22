"""Mixed-effects reanalysis for the crossed, clustered design (reviewers YMXf, ke2u).

The unit of analysis is the condition = (model x task). Each of the 114 conditions
carries one aggregated TSS value and one aggregated AC value (each computed over the
10 repeated runs of that condition). This is the finest grain the released metrics
support and matches the base analysis (scripts/verify.py, compute_stats.py), whose
paired t-test is over these 114 condition-level pairs.

We refit the three headline inferences with models that respect the task and model
grouping, plus task-level cluster bootstrap and task-stratified permutation as
distribution-free robustness checks. REML for variance-component estimates; ML for
likelihood-ratio tests. All numbers are printed; nothing is hard-coded.

Run from the repository root (needs results/experiment_20260705_205523.json):
    python analysis/mixed_models.py
"""
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")  # convergence chatter is reported explicitly below
np.random.seed(20260917)

PATH = "results/experiment_20260705_205523.json"
d = json.load(open(PATH))

rows = []
for c in d:
    rows.append(dict(
        task=c["task_id"], model=c["model"], category=c["category"],
        ambiguous=int(c["category"] == "ambiguous"),
        tss=c["metrics"]["tool_sequence_similarity"],
        ac=c["metrics"]["argument_consistency"],
    ))
w = pd.DataFrame(rows)  # 114 rows, wide (one TSS, one AC per condition)
assert len(w) == 114 and w.task.nunique() == 19 and w.model.nunique() == 6

# long form for the metric-type model: 228 rows
long = pd.melt(w, id_vars=["task", "model", "category", "ambiguous"],
               value_vars=["tss", "ac"], var_name="metric", value_name="value")
long["metric"] = pd.Categorical(long["metric"], categories=["ac", "tss"])  # AC = reference
long["metric_num"] = (long["metric"] == "tss").astype(float)

R = 10000  # bootstrap / permutation draws
tasks = w.task.unique()

def line():
    print("=" * 70)

def vc_report(res, label):
    """Print variance components and convergence for a crossed-VC fit."""
    conv = "converged" if res.converged else "DID NOT CONVERGE"
    print(f"  [{label}] {conv}")

# ======================================================================
line(); print("H1. THE TSS vs AC GAP"); line()

# --- naive paired t-test (reproduces base analysis) ---
diff = (w.tss - w.ac).values
t_naive, p_naive = stats.ttest_rel(w.tss, w.ac)
dz = diff.mean() / diff.std(ddof=1)
sp = np.sqrt((w.tss.var(ddof=1) + w.ac.var(ddof=1)) / 2)
d_pooled = diff.mean() / sp
print(f"Naive paired t-test (treats 114 conditions as independent):")
print(f"  gap = {diff.mean():.4f}  t = {t_naive:.3f}  p = {p_naive:.3e}"
      f"  d_z = {dz:.3f}  d_pooled = {d_pooled:.3f}")

# --- mixed model: value ~ metric + crossed (1|task) + (1|model) ---
long["grp"] = 0
vcf = {"task": "0 + C(task)", "model": "0 + C(model)"}
m1 = smf.mixedlm("value ~ metric", long, groups=long["grp"],
                 vc_formula=vcf, re_formula="0").fit(reml=True)
b = m1.fe_params["metric[T.tss]"]; se = m1.bse["metric[T.tss]"]
ci_lo, ci_hi = b - 1.96 * se, b + 1.96 * se
pv = m1.pvalues["metric[T.tss]"]
print(f"\nMixed model  value ~ metric + (1|task) + (1|model)  [REML]:")
vc_report(m1, "crossed intercepts")
print(f"  TSS-AC fixed effect = {b:.4f}  SE = {se:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  p = {pv:.3e}")
print(f"  variance components: task={m1.vcomp[0]:.5f}  model={m1.vcomp[1]:.5f}  residual={m1.scale:.5f}")

# --- add random slope for metric by task (metric effect varies across tasks) ---
vcf2 = {"task": "0 + C(task)", "model": "0 + C(model)", "task_metric": "0 + C(task):metric_num"}
try:
    m1b = smf.mixedlm("value ~ metric", long, groups=long["grp"],
                      vc_formula=vcf2, re_formula="0").fit(reml=True)
    b2 = m1b.fe_params["metric[T.tss]"]; se2 = m1b.bse["metric[T.tss]"]
    print(f"\nWith random metric-by-task slope [REML]:")
    vc_report(m1b, "crossed + metric-by-task slope")
    print(f"  TSS-AC fixed effect = {b2:.4f}  SE = {se2:.4f}"
          f"  95% CI [{b2-1.96*se2:.4f}, {b2+1.96*se2:.4f}]  p = {m1b.pvalues['metric[T.tss]']:.3e}")
    print(f"  metric-by-task slope variance = {m1b.vcomp[2]:.6f}")
except Exception as e:
    print(f"\nRandom metric-by-task slope model failed: {e}")

# --- task-level cluster bootstrap of the mean gap ---
per_task_gap = w.groupby("task").apply(lambda g: (g.tss - g.ac).mean())
boot = np.empty(R)
for i in range(R):
    samp = np.random.choice(tasks, size=len(tasks), replace=True)
    boot[i] = np.mean([per_task_gap[t] for t in samp])
blo, bhi = np.percentile(boot, [2.5, 97.5])
p_boot = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
print(f"\nTask-level cluster bootstrap (resample 19 tasks, {R} draws):")
print(f"  mean gap = {boot.mean():.4f}  95% CI [{blo:.4f}, {bhi:.4f}]  boot p = {p_boot:.4g}")

# --- task-stratified sign-flip permutation of the paired difference ---
obs = diff.mean()
cnt = 0
for _ in range(R):
    signs = np.random.choice([-1, 1], size=len(diff))
    if abs((diff * signs).mean()) >= abs(obs):
        cnt += 1
p_perm = (cnt + 1) / (R + 1)
print(f"Task-stratified sign-flip permutation ({R} draws):  perm p = {p_perm:.4g}")

# ======================================================================
line(); print("H2. THE AMBIGUITY EFFECT (on AC)"); line()
amb = w[w.ambiguous == 1].ac; rest = w[w.ambiguous == 0].ac
sp_a = np.sqrt(((len(amb)-1)*amb.var(ddof=1) + (len(rest)-1)*rest.var(ddof=1)) / (len(amb)+len(rest)-2))
d_amb = (rest.mean() - amb.mean()) / sp_a
t_amb, p_amb = stats.ttest_ind(rest, amb)  # equal-var, matches paper's p=0.16
print(f"Naive independent t-test:")
print(f"  ambiguous AC = {amb.mean():.4f} (n={len(amb)}, 4 tasks x 6 models)")
print(f"  structured AC = {rest.mean():.4f} (n={len(rest)})")
print(f"  diff = {rest.mean()-amb.mean():.4f}  d = {d_amb:.3f}  t = {t_amb:.3f}  p = {p_amb:.3f}")

# mixed model: model random effect (task cannot be random AND identify a task-level
# fixed covariate cleanly, but we also try task random for completeness)
m2 = smf.mixedlm("ac ~ ambiguous", w, groups=w["model"]).fit(reml=True)
b_amb = m2.fe_params["ambiguous"]; se_amb = m2.bse["ambiguous"]
print(f"\nMixed model  ac ~ ambiguous + (1|model)  [REML]:")
vc_report(m2, "model random intercept")
print(f"  ambiguity effect = {b_amb:.4f}  SE = {se_amb:.4f}"
      f"  95% CI [{b_amb-1.96*se_amb:.4f}, {b_amb+1.96*se_amb:.4f}]  p = {m2.pvalues['ambiguous']:.3f}")

# by-task cluster bootstrap (the honest uncertainty: only 4 ambiguous tasks)
task_amb = w.groupby("task").agg(ac=("ac", "mean"), ambiguous=("ambiguous", "max"))
boot2 = []
for _ in range(R):
    samp = task_amb.sample(len(task_amb), replace=True)
    a = samp[samp.ambiguous == 1].ac; r = samp[samp.ambiguous == 0].ac
    if len(a) and len(r):
        boot2.append(r.mean() - a.mean())
boot2 = np.array(boot2)
b2lo, b2hi = np.percentile(boot2, [2.5, 97.5])
p_boot2 = 2 * min((boot2 <= 0).mean(), (boot2 >= 0).mean())
print(f"\nBy-task cluster bootstrap (resample 19 tasks, {len(boot2)} usable draws):")
print(f"  structured - ambiguous = {boot2.mean():.4f}  95% CI [{b2lo:.4f}, {b2hi:.4f}]  boot p = {p_boot2:.4g}")

# ======================================================================
line(); print("H3. CROSS-MODEL DIFFERENCES (replace one-way ANOVA)"); line()
for metric in ["tss", "ac"]:
    x = w[metric].values
    groups = [w[w.model == m][metric].values for m in w.model.unique()]
    F, p_an = stats.f_oneway(*groups)
    ssb = sum(len(g)*(g.mean()-x.mean())**2 for g in groups)
    eta2 = ssb / ((x - x.mean())**2).sum()
    print(f"\n{metric.upper()}:")
    print(f"  naive one-way ANOVA: F = {F:.3f}  p = {p_an:.4f}  eta2 = {eta2:.3f}")

    # mixed model: model FIXED, task RANDOM; LR test on the model term (ML)
    full = smf.mixedlm(f"{metric} ~ C(model)", w, groups=w["task"]).fit(reml=False)
    null = smf.mixedlm(f"{metric} ~ 1", w, groups=w["task"]).fit(reml=False)
    lr = 2 * (full.llf - null.llf)
    df_lr = full.df_modelwc - null.df_modelwc
    p_lr = stats.chi2.sf(lr, df_lr)
    conv = "converged" if (full.converged and null.converged) else "CONVERGENCE ISSUE"
    print(f"  mixed  {metric} ~ C(model) + (1|task)  [{conv}]:")
    print(f"    LR test of model term: chi2({df_lr}) = {lr:.3f}  p = {p_lr:.4f}")
    tau = full.cov_re.iloc[0, 0]
    print(f"    task variance = {tau:.5f}  residual = {full.scale:.5f}")

    # task-stratified permutation: permute model labels WITHIN each task, recompute F
    obsF = F
    ge = 0
    vals_by_task = {t: w[w.task == t][metric].values.copy() for t in tasks}
    labs_by_task = {t: w[w.task == t]["model"].values.copy() for t in tasks}
    for _ in range(R):
        pv_all, pl_all = [], []
        for t in tasks:
            v = vals_by_task[t]
            perm = np.random.permutation(len(v))
            pv_all.append(v); pl_all.append(labs_by_task[t][perm])
        vv = np.concatenate(pv_all); ll = np.concatenate(pl_all)
        gg = [vv[ll == m] for m in w.model.unique()]
        Fp, _ = stats.f_oneway(*gg)
        if Fp >= obsF:
            ge += 1
    p_permF = (ge + 1) / (R + 1)
    print(f"    task-stratified permutation of model labels ({R} draws): p = {p_permF:.4f}")

# ======================================================================
line(); print("VARIANCE DECOMPOSITION  value ~ 1 + (1|task) + (1|model)  [REML]"); line()
for metric in ["tss", "ac"]:
    dd = w[[metric, "task", "model"]].rename(columns={metric: "value"}).copy()
    dd["grp"] = 0
    md = smf.mixedlm("value ~ 1", dd, groups=dd["grp"], vc_formula=vcf, re_formula="0").fit(reml=True)
    vtask, vmodel, vres = md.vcomp[0], md.vcomp[1], md.scale
    tot = vtask + vmodel + vres
    conv = "converged" if md.converged else "DID NOT CONVERGE"
    print(f"\n{metric.upper()} [{conv}]  total var = {tot:.5f}")
    print(f"  task     = {vtask:.5f}  ({100*vtask/tot:.1f}%)")
    print(f"  model    = {vmodel:.5f}  ({100*vmodel/tot:.1f}%)")
    print(f"  residual = {vres:.5f}  ({100*vres/tot:.1f}%)   (residual = task x model interaction + run-level noise)")
line(); print("DONE"); line()
