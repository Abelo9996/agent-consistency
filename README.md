# AgentConsistency: Behavioral Consistency of Tool-Calling LLM Agents

**Paper**: "How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines"

## Key Question
When you give the same task to an LLM agent with tool-calling capabilities, does it:
1. Call the same tools?
2. In the same order?
3. With the same arguments?
4. And produce the same final answer?

## What's New vs Prior Work
- [Mehta et al. (2026)](https://arxiv.org/abs/2602.11619) measured consistency in ReAct agents on HotpotQA (search-only actions)
- We extend to **structured tool-calling** (function calling) across diverse real-world task categories
- We measure consistency at multiple granularities: tool selection, argument values, execution order, and final output
- We test across 5+ models and analyze which factors predict inconsistency

## Structure
```
agent-consistency/
├── src/
│   ├── tasks/          # Task definitions (JSON schemas)
│   ├── agents/         # Agent implementations per provider
│   ├── metrics/        # Consistency metrics
│   ├── runners/        # Experiment orchestration
│   └── analysis/       # Results analysis & figures
├── configs/            # Experiment configs
├── results/            # Raw results (gitignored, large)
├── figures/            # Generated figures for paper
├── paper/              # LaTeX source
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python -m src.runners.run_experiment --config configs/main.yaml
python -m src.analysis.generate_figures
```

## Models Tested
- GPT-4o / GPT-4o-mini (OpenAI)
- Claude Sonnet 4.5 / Claude Haiku (Anthropic)
- Llama 3.1 70B / 8B (via Together/Groq)
- Gemini 2.0 Flash (Google)

## Task Categories
1. **Data Retrieval & Aggregation** — Multi-step API lookups, combine results
2. **Planning & Scheduling** — Calendar management, booking, conflict resolution
3. **Data Transformation** — Parse, filter, compute over structured data
4. **Multi-Tool Composition** — Tasks requiring 3+ different tools in sequence
5. **Ambiguous Tasks** — Intentionally underspecified to test divergence under uncertainty

## Metrics
- **Tool Sequence Similarity** (normalized edit distance on tool call sequences)
- **Argument Consistency** (Jaccard similarity on structured args)
- **Order Consistency** (Kendall's tau on tool call ordering)
- **Output Agreement** (exact match + semantic similarity on final answers)
- **Divergence Point** (which step does inconsistency first appear?)

## License
MIT
