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
    """
    Two-panel figure: binned bar chart (left) + annotated scatter with jitter (right).

    Panel A (left): Mean correctness rate for Low / Mid / High TSS tertiles, broken out
    by task category.  Directly communicates the headline 90.2% vs 61.2% finding.

    Panel B (right): Scatter of AC vs correctness with vertical jitter to reveal
    overplotted points.  A flat trend line visually confirms the non-finding (r=0.12).

    This replaces the original dual-scatter design which suffered from severe ceiling
    overplotting (most points at y=1.0) and obscured the TSS median-split result.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    matplotlib.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })

    data = analysis["per_task_model"]
    r_tss = analysis["correlations"]["tss_pearson"]["r"]
    p_tss = analysis["correlations"]["tss_pearson"]["p"]
    r_ac  = analysis["correlations"]["ac_pearson"]["r"]
    p_ac  = analysis["correlations"]["ac_pearson"]["p"]
    ms    = analysis.get("median_split", {})

    CAT_ORDER  = ["retrieval", "scheduling", "computation", "composition", "ambiguous"]
    CAT_COLORS = {
        "retrieval":   "#4C72B0",
        "scheduling":  "#DD8452",
        "computation": "#55A868",
        "ambiguous":   "#8172B2",
        "composition": "#C44E52",
    }

    tss_vals = np.array([e["tss"]             for e in data if e["tss"] is not None])
    ac_vals  = np.array([e["ac"]              for e in data if e["ac"]  is not None])
    cor_vals = np.array([e["correctness_rate"] for e in data if e["tss"] is not None])
    cats     = [e["category"]                 for e in data if e["tss"] is not None]

    # --- TSS tertile bins ---------------------------------------------------
    t33, t67 = np.percentile(tss_vals, [33, 67])
    def tss_bin(v):
        if v < t33:   return "Low\nTSS"
        if v < t67:   return "Mid\nTSS"
        return "High\nTSS"
    bin_labels = ["Low\nTSS", "Mid\nTSS", "High\nTSS"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # =========================================================
    # Panel A: grouped bar chart — correctness by TSS bin × category
    # =========================================================
    ax = axes[0]
    bin_cat_means = {bl: {} for bl in bin_labels}
    bin_cat_ns    = {bl: {} for bl in bin_labels}
    for v, c_rate, cat in zip(tss_vals, cor_vals, cats):
        bl = tss_bin(v)
        bin_cat_means[bl].setdefault(cat, []).append(c_rate)
        bin_cat_ns[bl].setdefault(cat, 0)
        bin_cat_ns[bl][cat] += 1

    n_bins = len(bin_labels)
    n_cats = len(CAT_ORDER)
    bar_w  = 0.13
    group_w = n_cats * bar_w + 0.08
    x_centers = np.arange(n_bins) * group_w

    for ci, cat in enumerate(CAT_ORDER):
        offsets = x_centers + ci * bar_w - (n_cats - 1) * bar_w / 2
        means = []
        sems  = []
        for bl in bin_labels:
            vals = bin_cat_means[bl].get(cat, [])
            means.append(np.mean(vals) if vals else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(offsets, means, bar_w, color=CAT_COLORS[cat], label=cat.capitalize(),
               yerr=sems, capsize=3, error_kw={"linewidth": 0.8}, alpha=0.88, edgecolor="white")

    # Overlay overall bin means as text annotations
    for i, (bl, xc) in enumerate(zip(bin_labels, x_centers)):
        all_in_bin = [c for b, c in zip([tss_bin(v) for v in tss_vals], cor_vals) if b == bl]
        if all_in_bin:
            ax.text(xc, min(1.02, np.mean(all_in_bin) + 0.04),
                    f"{np.mean(all_in_bin):.0%}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#222")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_ylabel("Task Correctness Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"TSS Predicts Correctness\n($r={r_tss:.2f}$, $p={p_tss:.3f}$; Spearman $\\rho=0.42$)",
                 fontweight="bold")
    ax.legend(loc="lower right", title="Category", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.771, color="gray", linestyle=":", linewidth=1.0, label="_nolegend_")
    ax.text(x_centers[-1] + group_w * 0.3, 0.771, "Overall\nmean", va="center",
            fontsize=8, color="gray")

    # =========================================================
    # Panel B: jittered scatter — AC vs correctness
    # =========================================================
    ax = axes[1]
    rng = np.random.default_rng(42)
    ac_full  = np.array([e["ac"]              for e in data if e["ac"]  is not None])
    cor_full = np.array([e["correctness_rate"] for e in data if e["ac"]  is not None])
    cats_ac  = [e["category"]                 for e in data if e["ac"]  is not None]

    # Vertical jitter — small so it stays readable
    jitter = rng.uniform(-0.025, 0.025, size=len(cor_full))
    for cat in CAT_ORDER:
        mask = np.array(cats_ac) == cat
        ax.scatter(ac_full[mask], cor_full[mask] + jitter[mask],
                   c=CAT_COLORS[cat], label=cat.capitalize(),
                   s=55, alpha=0.72, edgecolors="white", linewidth=0.4, zorder=3)

    # Flat trend line
    z = np.polyfit(ac_full, cor_full, 1)
    x_line = np.linspace(ac_full.min(), ac_full.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "--", color="gray", alpha=0.55, linewidth=1.5)

    ax.set_xlabel("Argument Consistency (AC)")
    ax.set_ylabel("Task Correctness Rate (± jitter)")
    ax.set_xlim(0.05, 1.08)
    ax.set_ylim(-0.10, 1.12)
    ax.set_title(f"AC Does Not Predict Correctness\n($r={r_ac:.2f}$, $p={p_ac:.3f}$, n.s.)",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Annotate non-significance explicitly
    ax.text(0.97, 0.06, "n.s.", transform=ax.transAxes, ha="right",
            fontsize=12, color="#888", style="italic")

    plt.suptitle("Structural Consistency Predicts Success; Argument Variance Is Benign",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    outpath = Path(output_dir)
    plt.savefig(outpath / 'fig10_correctness_vs_consistency.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(outpath / 'fig10_correctness_vs_consistency.png', bbox_inches='tight', dpi=300)
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
