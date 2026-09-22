# 🔄 Agent Consistency

**How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2605.28840-b31b1b.svg)](https://arxiv.org/abs/2605.28840)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> When you give the same task to a tool-calling LLM agent twice, does it behave the same way? We measure where in the pipeline the instability lives. Agents reliably select the same tools in the same order, but vary more in the arguments they pass to those tools. We call this "structural consistency, parametric variance." All results below are from the released traces; every headline number is reproducible with the scripts in `analysis/`.

---

## 📊 Key results

<p align="center">
  <img src="paper/figures/fig4_model_comparison.png" width="80%" alt="Model comparison across TSS and AC">
</p>

| Finding | Result |
|---------|--------|
| Tool Sequence Similarity (TSS), mean | **0.88** (95% CI [0.85, 0.91]) |
| Argument Consistency (AC), mean | **0.70** (95% CI [0.65, 0.75]) |
| Structure vs arguments gap | paired d = 0.76, p = 2e-12; survives a crossed random-effects model (TSS - AC = 0.177, 95% CI [0.124, 0.231]) and a task-level cluster bootstrap ([0.11, 0.25]) |
| First-divergence events in steps 1 to 2 | **64%** of diverging run pairs (short-horizon tasks; see the caveat below) |
| Output exact-string match | **4.1%** of run pairs, even when tool sequences are identical |
| Ambiguity effect on AC | directional 13% drop, but **does not survive clustering** (d = 0.33, p = 0.16; by-task bootstrap CI [-0.10, 0.28]) |
| Cross-model differences | modest (eta^2 approx 0.10) but **survive** clustering (mixed-model LR p = 0.02 for TSS, p = 0.006 for AC) |

**Metric-comparability caveat.** TSS (normalized edit similarity over tool-name sequences) and AC (step-aligned Jaccard over argument key-value sets) are different similarity functions on different objects. Their numerical values are **not** directly comparable, so a TSS of 0.88 and an AC of 0.70 do not mean structure is 0.18 "more consistent." We report the direction and size of the gap within a common similarity family and calibrate both metrics against chance and known perturbations in `analysis/metric_ablation.py`; the structure-over-arguments ordering holds across similarity families.

### The "structural consistency, parametric variance" pattern

Repeated runs of one fixed model on one fixed input tend to follow the same procedure (which tools, in what order) more than they instantiate the same arguments (search strings, date formats, message text). We label the training explanation (procedural schemas from RLHF/SFT) a **hypothesis the paper does not test**, and name alternatives it cannot rule out (tool schemas constraining procedure more than arguments; sharper decoding over a short tool-name vocabulary).

---

## ⚠️ Scope and honest limits

- **Short-horizon tasks.** Trace lengths are short (mean 2.57 tool calls, median 2, range 0 to 19). When first-divergence steps are normalized by trajectory length, the median first divergence sits at 0.78 of the trajectory, so the "divergence is early" reading is weak; we scope the early-divergence observation to short-horizon pipelines rather than to agents that span tens or hundreds of calls.
- **Consistency is not correctness.** On the fresh mid-2026 models (markedly more consistent, median condition-level TSS = 1.0) the TSS-correctness association is a weak rank correlation (Spearman rho = 0.22, p = 0.017; Pearson n.s.) and the AC-correctness correlation is not distinguishable from zero. We present TSS as a variance signal whose predictive value declines as models grow more consistent, **not** as a correctness proxy.
- **Two non-equivalent snapshots.** The comparison to a late-2025 collection is a descriptive comparison of two non-equivalent instruments (unpinned 2025 model aliases, a different API path, and 2025 raw traces not preserved), not a controlled replication or a claim of model progress.
- **All inference is clustered.** The design is crossed (19 tasks x 6 models) and clustered; primary inference is a crossed random-effects mixed model plus a task-level cluster bootstrap and a task-stratified permutation test (`analysis/mixed_models.py`). Naive t-tests and ANOVAs are reported only for comparison.

---

## 🏗️ Benchmark design

**19 tasks** across 5 categories of increasing ambiguity, evaluated with **deterministic simulated tools**:

| Category | Tasks | Description |
|----------|-------|-------------|
| Data retrieval | 4 | Contact lookups, email search, aggregation |
| Scheduling | 4 | Calendar events, conflict detection, free slots |
| Computation | 3 | Inventory calculations, revenue projections |
| Multi-tool composition | 4 | 3 to 5 tools in sequence (find email, look up sender, schedule meeting) |
| Ambiguous | 4 | Intentionally underspecified ("prepare for my meetings tomorrow") |

All tools are **fully deterministic**: identical inputs always produce identical outputs, which isolates model variance from environment variance.

---

## 🤖 Models evaluated

Per-model means over 19 tasks (recompute with `analysis/verify.py`):

| Model | Provider | TSS | AC | Unique seqs / 10 |
|-------|----------|-----|-----|-----------------|
| GPT-4.1 | OpenAI | 0.949 | 0.789 | 1.32 |
| GPT-4.1-mini | OpenAI | 0.895 | 0.783 | 1.63 |
| GPT-4o-mini | OpenAI | 0.879 | 0.676 | 1.89 |
| GPT-4o | OpenAI | 0.883 | 0.635 | 1.74 |
| Claude Sonnet 4 | Anthropic | 0.880 | 0.770 | 2.26 |
| Llama 3.3 70B | Meta | 0.773 | 0.542 | 3.11 |

**6 models x 19 tasks x 10 runs = 1,140 agent traces**, collected July 2026 through a single provider-normalized (OpenAI-style) gateway with version-pinned model identifiers. All 1,140 raw traces are released in `results/`. Temperature is 1.0 (the gateway default); we report a single setting and do not characterize other temperatures.

---

## 📏 Metrics

- **Tool Sequence Similarity (TSS):** mean pairwise normalized Levenshtein similarity over tool-name sequences ("do runs follow the same procedure").
- **Argument Consistency (AC):** mean pairwise Jaccard overlap of key-value argument sets at step-aligned calls ("do runs parameterize it the same way").
- **First-divergence step:** the step at which a run pair first differs; identical pairs contribute no event, so the denominator is the set of diverging pairs.
- **Output agreement:** exact-string match rate of the full final natural-language response (no normalization or semantic matching; the low rate shows exact-string assertions on free text will fail, not that output carries no signal).

See the metric-comparability caveat above; `analysis/metric_ablation.py` reports AC restricted to aligned same-tool calls and both metrics under a common similarity family and against chance/perturbation baselines.

---

## 🔬 Reproducing the paper's numbers

Every number the paper reports is re-derived from `results/experiment_20260705_205523.json` by the scripts in `analysis/`:

```bash
pip install -r requirements.txt   # numpy, scipy, statsmodels, patsy

python analysis/verify.py          # TSS, AC, effect size, output match, per-model, length distribution
python analysis/divergence.py      # first-divergence statistics and denominators
python analysis/mixed_models.py    # crossed random-effects model + cluster bootstrap + permutation
python analysis/metric_ablation.py # common-family ablation, AC decomposition, chance/perturbation baselines
```

All four read `results/...` relative to the repository root; run them from there.

### Re-running the experiment from scratch (optional)

```bash
cp .env.example .env               # add your OpenAI / Anthropic / Together keys
python -m src.runners.run_experiment --experiment exp1
python -m src.analysis.generate_figures
```

---

## 📁 Repository structure

```
agent-consistency/
├── src/
│   ├── tasks/          # 19 task definitions with tool schemas
│   ├── agents/         # Agent runner (provider-normalized tool-calling loop)
│   ├── metrics/        # TSS, AC, divergence point, output agreement
│   ├── runners/        # Experiment orchestration
│   ├── analysis/       # Figure generation
│   └── tools.py        # Deterministic simulated tools
├── analysis/           # Reproducibility scripts for every headline number
├── configs/            # Experiment configuration (YAML)
├── figures/            # Generated figures (PDF + PNG)
├── paper/              # LaTeX source + compiled PDF
├── results/            # Raw trace data: all 1,140 traces (released)
├── requirements.txt
└── .env.example
```

---

## 📄 Citation

```bibtex
@article{yagubyan2026agentconsistency,
  title={How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines},
  author={Yagubyan, Abel},
  year={2026},
  eprint={2605.28840},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

---

## 📜 License

Released under the MIT License. See [LICENSE](LICENSE).
