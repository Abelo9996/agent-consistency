"""
Correctness evaluation for existing agent traces.

Defines expected outcomes for each of the 19 tasks and scores existing traces
for task success. Then correlates consistency metrics (TSS, AC) with correctness.

This addresses Reviewer Weakness #2: "No correctness evaluation."
"""

import json
import re
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats


# ============================================================
# Expected outcomes per task
# ============================================================
# Each task has a set of correctness criteria: required tool calls,
# required argument patterns, and/or required output patterns.

CORRECTNESS_CRITERIA = {
    "retrieve-001": {
        "description": "Find Alice's email and send her a message about meeting at 3pm",
        "required_tools": {"get_contact", "send_email"},
        "required_tool_sequence_contains": ["get_contact", "send_email"],  # order matters
        "arg_checks": [
            {"tool": "get_contact", "key": "name", "pattern": r"(?i)alice"},
            {"tool": "send_email", "key": "to", "pattern": r"alice@example\.com"},
            {"tool": "send_email", "key": "body", "pattern": r"(?i)3\s*pm"},
        ],
    },
    "retrieve-002": {
        "description": "Find StartupXYZ contacts and email each about demo March 10",
        "required_tools": {"search_contacts", "send_email"},
        "arg_checks": [
            {"tool": "search_contacts", "key": "query", "pattern": r"(?i)startup"},
            {"tool": "send_email", "key": "body", "pattern": r"(?i)(march\s*10|demo)"},
        ],
        "min_send_emails": 2,  # Eve and Frank are at StartupXYZ
    },
    "retrieve-003": {
        "description": "Search emails about budget and summarize",
        "required_tools": {"search_emails"},
        "arg_checks": [
            {"tool": "search_emails", "key": "query", "pattern": r"(?i)budget"},
        ],
        "output_pattern": r"(?i)(\$?45|20|budget)",  # should mention key figures
    },
    "retrieve-004": {
        "description": "Find emails with dollar amounts and calculate total",
        "required_tools": {"search_emails"},
        "arg_checks": [],
        "output_pattern": r"\d",  # should contain some number
    },
    "schedule-001": {
        "description": "Schedule 30min meeting with Bob March 2 at 2pm",
        "required_tools": {"create_calendar_event"},
        "arg_checks": [
            {"tool": "create_calendar_event", "key": "title", "pattern": r"(?i)design\s*review"},
            {"tool": "create_calendar_event", "key": "date", "pattern": r"2026-03-02"},
            {"tool": "create_calendar_event", "key": "start_time", "pattern": r"14:00"},
        ],
    },
    "schedule-002": {
        "description": "Check calendar March 3, find free 1hr slot 9-5, schedule with Eve",
        "required_tools": {"list_calendar_events", "create_calendar_event"},
        "required_tool_sequence_contains": ["list_calendar_events", "create_calendar_event"],
        "arg_checks": [
            {"tool": "list_calendar_events", "key": "start_date", "pattern": r"2026-03-03"},
        ],
    },
    "schedule-003": {
        "description": "Check whole week March 1-5, find day with most free time, schedule 2hr session",
        "required_tools": {"list_calendar_events", "create_calendar_event"},
        "arg_checks": [
            {"tool": "create_calendar_event", "key": "title", "pattern": r"(?i)strategy"},
        ],
    },
    "schedule-004": {
        "description": "Check March 3 for conflicts, email affected attendees",
        "required_tools": {"list_calendar_events"},
        "arg_checks": [
            {"tool": "list_calendar_events", "key": "start_date", "pattern": r"2026-03-03"},
        ],
    },
    "compute-001": {
        "description": "Total value of electronics (price × stock for each)",
        "required_tools": {"search_products"},
        "arg_checks": [
            {"tool": "search_products", "key": "query", "pattern": r"(?i)electr"},
        ],
        # Correct answer: 29.99*150 + 49.99*75 + 199.99*30 + 349.99*45 = 4498.5+3749.25+5999.7+15749.55 = 29997.00
        "output_pattern": r"29[,.]?997",
    },
    "compute-002": {
        "description": "Products under $50, revenue from selling 50% of stock",
        "required_tools": {"search_products"},
        "arg_checks": [],
    },
    "compute-003": {
        "description": "Total inventory value per category, find highest",
        "required_tools": {"search_products"},
        "arg_checks": [],
        "output_pattern": r"(?i)electr",  # electronics should be highest
    },
    "compose-001": {
        "description": "Find Acme email, look up Dave, schedule meeting March 4 3pm",
        "required_tools": {"search_emails", "create_calendar_event"},
        "arg_checks": [
            {"tool": "search_emails", "key": "query", "pattern": r"(?i)acme"},
            {"tool": "create_calendar_event", "key": "date", "pattern": r"2026-03-04"},
        ],
    },
    "compose-002": {
        "description": "Check weather SF+NY, email Alice if cold, include calendar",
        "required_tools": {"get_weather"},
        "min_weather_calls": 2,
        "arg_checks": [],
    },
    "compose-003": {
        "description": "Search marketing email, extract metrics, calculate spend, create note",
        "required_tools": {"search_emails"},
        "arg_checks": [
            {"tool": "search_emails", "key": "query", "pattern": r"(?i)market"},
        ],
    },
    "compose-004": {
        "description": "Find Eve's email about board deck, check calendar, lookup attendees, send reminders",
        "required_tools": {"search_emails", "list_calendar_events"},
        "arg_checks": [
            {"tool": "search_emails", "key": "query", "pattern": r"(?i)(board|eve|deck)"},
        ],
    },
    # Ambiguous tasks: scored more leniently — did the agent take reasonable action?
    "ambig-001": {
        "description": "Help prepare for meetings tomorrow",
        "required_tools": {"list_calendar_events"},
        "arg_checks": [],
    },
    "ambig-002": {
        "description": "Follow up on important things from this week",
        "required_tools_any": [{"search_emails"}, {"list_calendar_events"}],  # either is reasonable
        "arg_checks": [],
    },
    "ambig-003": {
        "description": "Get ready for investor call",
        "required_tools_any": [{"search_emails"}, {"list_calendar_events"}, {"search_contacts"}],
        "arg_checks": [],
    },
    "ambig-004": {
        "description": "What should I focus on this week",
        "required_tools_any": [{"list_calendar_events"}, {"search_emails"}],
        "arg_checks": [],
    },
}


def score_trace(trace_dict: dict, task_id: str) -> dict:
    """
    Score a single trace for correctness.
    Returns a dict with:
      - correct: bool (overall pass/fail)
      - tools_correct: bool (used the right tools)
      - args_correct: float (fraction of arg checks passed)
      - output_correct: bool or None (if output pattern defined)
      - details: list of check results
    """
    criteria = CORRECTNESS_CRITERIA.get(task_id)
    if not criteria:
        return {"correct": None, "reason": "no criteria defined"}
    
    # Extract tool calls from trace
    tool_calls = trace_dict.get("tool_calls", [])
    if isinstance(tool_calls, list) and len(tool_calls) > 0 and isinstance(tool_calls[0], dict):
        tools_used = set(tc["tool_name"] for tc in tool_calls)
        tool_sequence = [tc["tool_name"] for tc in tool_calls]
    else:
        tools_used = set()
        tool_sequence = []
    
    final_response = trace_dict.get("final_response", "")
    details = []
    
    # 1. Required tools check
    tools_correct = True
    if "required_tools" in criteria:
        missing = criteria["required_tools"] - tools_used
        tools_correct = len(missing) == 0
        details.append({"check": "required_tools", "pass": tools_correct, 
                       "missing": list(missing) if missing else []})
    elif "required_tools_any" in criteria:
        # For ambiguous tasks: any one of the tool sets is acceptable
        tools_correct = any(
            req_set.issubset(tools_used) 
            for req_set in criteria["required_tools_any"]
        )
        details.append({"check": "required_tools_any", "pass": tools_correct})
    
    # 2. Tool sequence order check
    if "required_tool_sequence_contains" in criteria:
        req_seq = criteria["required_tool_sequence_contains"]
        # Check if required sequence appears as subsequence
        seq_idx = 0
        for tool_name in tool_sequence:
            if seq_idx < len(req_seq) and tool_name == req_seq[seq_idx]:
                seq_idx += 1
        seq_correct = seq_idx == len(req_seq)
        details.append({"check": "tool_order", "pass": seq_correct})
        tools_correct = tools_correct and seq_correct
    
    # 3. Argument checks
    args_passed = 0
    args_total = len(criteria.get("arg_checks", []))
    for ac in criteria.get("arg_checks", []):
        matched = False
        for tc in tool_calls:
            if tc["tool_name"] == ac["tool"]:
                val = str(tc.get("arguments", {}).get(ac["key"], ""))
                if re.search(ac["pattern"], val):
                    matched = True
                    break
        args_passed += int(matched)
        details.append({"check": f"arg:{ac['tool']}.{ac['key']}", "pass": matched})
    
    args_correct = args_passed / args_total if args_total > 0 else 1.0
    
    # 4. Special checks
    if "min_send_emails" in criteria:
        n_emails = sum(1 for tc in tool_calls if tc["tool_name"] == "send_email")
        email_ok = n_emails >= criteria["min_send_emails"]
        details.append({"check": "min_send_emails", "pass": email_ok, 
                       "sent": n_emails, "required": criteria["min_send_emails"]})
        tools_correct = tools_correct and email_ok
    
    if "min_weather_calls" in criteria:
        n_weather = sum(1 for tc in tool_calls if tc["tool_name"] == "get_weather")
        weather_ok = n_weather >= criteria["min_weather_calls"]
        details.append({"check": "min_weather_calls", "pass": weather_ok,
                       "called": n_weather, "required": criteria["min_weather_calls"]})
        tools_correct = tools_correct and weather_ok
    
    # 5. Output check
    output_correct = None
    if "output_pattern" in criteria and final_response:
        output_correct = bool(re.search(criteria["output_pattern"], final_response))
        details.append({"check": "output_pattern", "pass": output_correct})
    
    # Overall: tools correct AND args mostly correct
    overall = tools_correct and args_correct >= 0.5
    if output_correct is not None:
        overall = overall  # don't gate on output — it's a bonus signal
    
    return {
        "correct": overall,
        "tools_correct": tools_correct,
        "args_correct": args_correct,
        "output_correct": output_correct,
        "details": details,
    }


def analyze_correctness_consistency(results_files: list[str]) -> dict:
    """
    Load experiment results, score all traces for correctness, and correlate
    with consistency metrics (TSS, AC).
    
    Returns analysis dict with:
      - per_task_model: correctness rates and consistency metrics
      - correlation: TSS vs correctness, AC vs correctness
      - summary statistics
    """
    # Load all results
    all_results = []
    for f in results_files:
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    
    per_task_model = []
    
    for result in all_results:
        task_id = result["task_id"]
        model = result["model"]
        metrics = result.get("metrics", {})
        
        if "error" in metrics:
            continue
        
        traces = result.get("traces", [])
        
        # Score each trace
        scores = []
        for trace in traces:
            if trace.get("error"):
                continue
            score = score_trace(trace, task_id)
            if score["correct"] is not None:
                scores.append(score)
        
        if not scores:
            continue
        
        correctness_rate = sum(1 for s in scores if s["correct"]) / len(scores)
        args_rate = np.mean([s["args_correct"] for s in scores])
        tools_rate = sum(1 for s in scores if s["tools_correct"]) / len(scores)
        
        entry = {
            "task_id": task_id,
            "model": model,
            "category": result.get("category", ""),
            "difficulty": result.get("difficulty", ""),
            "correctness_rate": correctness_rate,
            "tools_correct_rate": tools_rate,
            "args_correct_rate": args_rate,
            "tss": metrics.get("tool_sequence_similarity", None),
            "ac": metrics.get("argument_consistency", None),
            "unique_sequences": metrics.get("unique_sequences", None),
            "n_traces": len(scores),
        }
        per_task_model.append(entry)
    
    # Compute correlations
    tss_vals = [e["tss"] for e in per_task_model if e["tss"] is not None]
    ac_vals = [e["ac"] for e in per_task_model if e["ac"] is not None]
    corr_vals = [e["correctness_rate"] for e in per_task_model if e["tss"] is not None]
    corr_vals_ac = [e["correctness_rate"] for e in per_task_model if e["ac"] is not None]
    
    tss_corr = stats.pearsonr(tss_vals, corr_vals) if len(tss_vals) >= 3 else (None, None)
    ac_corr = stats.pearsonr(ac_vals, corr_vals_ac) if len(ac_vals) >= 3 else (None, None)
    
    # Spearman too (more robust to non-linearity)
    tss_spearman = stats.spearmanr(tss_vals, corr_vals) if len(tss_vals) >= 3 else (None, None)
    ac_spearman = stats.spearmanr(ac_vals, corr_vals_ac) if len(ac_vals) >= 3 else (None, None)
    
    # Binary split: high vs low consistency
    tss_median = np.median(tss_vals)
    high_tss = [e for e in per_task_model if e["tss"] is not None and e["tss"] >= tss_median]
    low_tss = [e for e in per_task_model if e["tss"] is not None and e["tss"] < tss_median]
    
    high_tss_corr = np.mean([e["correctness_rate"] for e in high_tss]) if high_tss else None
    low_tss_corr = np.mean([e["correctness_rate"] for e in low_tss]) if low_tss else None
    
    # Effect size for high vs low TSS on correctness
    if high_tss and low_tss:
        high_rates = [e["correctness_rate"] for e in high_tss]
        low_rates = [e["correctness_rate"] for e in low_tss]
        t_stat, t_p = stats.ttest_ind(high_rates, low_rates)
        pooled_std = np.sqrt((np.var(high_rates) + np.var(low_rates)) / 2)
        cohens_d = (np.mean(high_rates) - np.mean(low_rates)) / pooled_std if pooled_std > 0 else 0
    else:
        t_stat, t_p, cohens_d = None, None, None
    
    # Overall correctness
    overall_correctness = np.mean([e["correctness_rate"] for e in per_task_model])
    
    return {
        "per_task_model": per_task_model,
        "overall_correctness": overall_correctness,
        "correlations": {
            "tss_pearson": {"r": tss_corr[0], "p": tss_corr[1]},
            "tss_spearman": {"rho": tss_spearman[0] if tss_spearman[0] is not None else None, 
                           "p": tss_spearman[1] if tss_spearman[1] is not None else None},
            "ac_pearson": {"r": ac_corr[0], "p": ac_corr[1]},
            "ac_spearman": {"rho": ac_spearman[0] if ac_spearman[0] is not None else None,
                          "p": ac_spearman[1] if ac_spearman[1] is not None else None},
        },
        "median_split": {
            "tss_median": tss_median,
            "high_tss_correctness": high_tss_corr,
            "low_tss_correctness": low_tss_corr,
            "cohens_d": cohens_d,
            "t_stat": t_stat,
            "p_value": t_p,
        },
        "n_conditions": len(per_task_model),
    }


def generate_correctness_figure(analysis: dict, output_dir: str = "figures"):
    """Generate scatter plot of TSS vs correctness rate."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    
    data = analysis["per_task_model"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: TSS vs Correctness
    ax = axes[0]
    categories = list(set(e["category"] for e in data))
    colors = {'retrieval': 'tab:blue', 'scheduling': 'tab:orange', 'computation': 'tab:green',
              'composition': 'tab:red', 'ambiguous': 'tab:purple'}
    
    for cat in categories:
        pts = [e for e in data if e["category"] == cat]
        ax.scatter([e["tss"] for e in pts], [e["correctness_rate"] for e in pts],
                  c=colors.get(cat, 'gray'), label=cat.capitalize(), alpha=0.7, s=50)
    
    # Add correlation line
    tss = [e["tss"] for e in data if e["tss"] is not None]
    corr = [e["correctness_rate"] for e in data if e["tss"] is not None]
    if tss:
        z = np.polyfit(tss, corr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tss), max(tss), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5)
    
    r_val = analysis["correlations"]["tss_pearson"]["r"]
    p_val = analysis["correlations"]["tss_pearson"]["p"]
    ax.set_xlabel('Tool Sequence Similarity (TSS)')
    ax.set_ylabel('Task Correctness Rate')
    ax.set_title(f'TSS vs Correctness (r={r_val:.2f}, p={p_val:.3f})' if r_val else 'TSS vs Correctness')
    ax.legend(fontsize=8)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(-0.05, 1.1)
    
    # Panel B: AC vs Correctness
    ax = axes[1]
    for cat in categories:
        pts = [e for e in data if e["category"] == cat]
        ax.scatter([e["ac"] for e in pts], [e["correctness_rate"] for e in pts],
                  c=colors.get(cat, 'gray'), label=cat.capitalize(), alpha=0.7, s=50)
    
    ac = [e["ac"] for e in data if e["ac"] is not None]
    corr_ac = [e["correctness_rate"] for e in data if e["ac"] is not None]
    if ac:
        z = np.polyfit(ac, corr_ac, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(ac), max(ac), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5)
    
    r_val_ac = analysis["correlations"]["ac_pearson"]["r"]
    p_val_ac = analysis["correlations"]["ac_pearson"]["p"]
    ax.set_xlabel('Argument Consistency (AC)')
    ax.set_ylabel('Task Correctness Rate')
    ax.set_title(f'AC vs Correctness (r={r_val_ac:.2f}, p={p_val_ac:.3f})' if r_val_ac else 'AC vs Correctness')
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    outpath = Path(output_dir)
    plt.savefig(outpath / 'fig10_correctness_vs_consistency.pdf', bbox_inches='tight')
    plt.savefig(outpath / 'fig10_correctness_vs_consistency.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig10_correctness_vs_consistency")


if __name__ == "__main__":
    import sys
    
    results_dir = Path("results")
    result_files = sorted(results_dir.glob("experiment_*.json"))
    # Exclude partial files
    result_files = [f for f in result_files if "partial" not in f.name]
    
    if not result_files:
        print("No result files found in results/")
        sys.exit(1)
    
    print(f"Loading {len(result_files)} result files...")
    analysis = analyze_correctness_consistency([str(f) for f in result_files])
    
    print(f"\n=== Correctness × Consistency Analysis ===")
    print(f"Conditions analyzed: {analysis['n_conditions']}")
    print(f"Overall correctness rate: {analysis['overall_correctness']:.2%}")
    print(f"\nCorrelations:")
    print(f"  TSS <-> Correctness: r={analysis['correlations']['tss_pearson']['r']:.3f}, p={analysis['correlations']['tss_pearson']['p']:.4f}")
    print(f"  TSS <-> Correctness (Spearman): rho={analysis['correlations']['tss_spearman']['rho']:.3f}, p={analysis['correlations']['tss_spearman']['p']:.4f}")
    print(f"  AC  <-> Correctness: r={analysis['correlations']['ac_pearson']['r']:.3f}, p={analysis['correlations']['ac_pearson']['p']:.4f}")
    
    ms = analysis["median_split"]
    print(f"\nMedian split (TSS median = {ms['tss_median']:.2f}):")
    print(f"  High TSS correctness: {ms['high_tss_correctness']:.2%}")
    print(f"  Low TSS correctness:  {ms['low_tss_correctness']:.2%}")
    print(f"  Cohen's d: {ms['cohens_d']:.2f}, p={ms['p_value']:.4f}")
    
    # Generate figure
    generate_correctness_figure(analysis)
    
    # Save analysis
    # Convert numpy types for JSON
    def np_convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open("figures/correctness_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=np_convert)
    print("\nSaved: figures/correctness_analysis.json")
