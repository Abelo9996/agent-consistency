"""
Main experiment runner.

Runs each task N times per model, collects traces, computes metrics.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

from src.tools import get_tool_schemas, execute_tool
from src.tasks.definitions import get_tasks, TASKS
from src.agents.runner import run_agent, PROVIDERS
from src.metrics.consistency import compute_all_metrics

console = Console()

DEFAULT_RUNS_PER_TASK = 10
DEFAULT_MODELS = ["gpt-4o-mini"]  # Start cheap, scale up


def run_experiment(
    models: list[str] = None,
    runs_per_task: int = DEFAULT_RUNS_PER_TASK,
    task_ids: list[str] = None,
    category: str = None,
    output_dir: str = "results",
):
    """Run the full experiment."""
    models = models or DEFAULT_MODELS
    tasks = TASKS
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if category:
        tasks = [t for t in tasks if t["category"] == category]

    tool_schemas = get_tool_schemas()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    partial_file = Path(output_dir) / f"experiment_{timestamp}_partial.json"

    def _save_partial():
        with open(partial_file, "w") as pf:
            json.dump(all_results, pf, indent=2, default=str)

    for model_key in models:
        console.rule(f"[bold blue]Model: {model_key}")

        for task_def in tasks:
            task_id = task_def["id"]
            task_text = task_def["task"]
            console.print(f"\n[yellow]Task {task_id}[/yellow]: {task_text[:80]}...")

            traces = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                ptask = progress.add_task(f"Running {runs_per_task} trials...", total=runs_per_task)

                for run_idx in range(runs_per_task):
                    try:
                        trace = run_agent(model_key, task_text, tool_schemas, execute_tool)
                        trace.task_id = task_id
                        trace.run_index = run_idx
                        traces.append(trace)
                    except Exception as e:
                        console.print(f"  [red]Run {run_idx} failed: {e}[/red]")
                        from src.agents.runner import AgentTrace
                        traces.append(AgentTrace(
                            task_id=task_id, model=model_key,
                            run_index=run_idx, error=str(e),
                        ))

                    progress.advance(ptask)
                    # Delay to avoid rate limits
                    time.sleep(0.5)

            # Compute metrics
            valid_traces = [t for t in traces if not t.error]
            if len(valid_traces) >= 2:
                metrics = compute_all_metrics(valid_traces)
            else:
                metrics = {"error": "insufficient valid traces", "valid_runs": len(valid_traces)}

            result = {
                "task_id": task_id,
                "task": task_text,
                "category": task_def["category"],
                "difficulty": task_def["difficulty"],
                "model": model_key,
                "metrics": metrics,
                "traces": [t.to_dict() for t in traces],
            }
            all_results.append(result)
            _save_partial()

            # Print summary
            if "error" not in metrics:
                _print_task_summary(task_id, model_key, metrics)

    # Save results
    output_file = Path(output_dir) / f"experiment_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    if partial_file.exists():
        partial_file.unlink()

    # Print overall summary table
    _print_overall_summary(all_results)

    return all_results


def _print_task_summary(task_id: str, model: str, metrics: dict):
    """Print a compact summary for a single task."""
    seq_sim = metrics["tool_sequence_similarity"]
    arg_con = metrics["argument_consistency"]
    uniq_seq = metrics["unique_sequences"]
    n_runs = metrics["n_runs"]
    div_pt = metrics["divergence_point"]
    out_agree = metrics["output_agreement"]

    color = "green" if seq_sim > 0.8 else "yellow" if seq_sim > 0.5 else "red"
    console.print(
        f"  [{color}]Seq Sim: {seq_sim:.2f}[/{color}] | "
        f"Arg Con: {arg_con:.2f} | "
        f"Unique Seqs: {uniq_seq}/{n_runs} | "
        f"Div Point: {div_pt or 'none'} | "
        f"Output Match: {out_agree['exact_match_rate']:.1%}"
    )


def _print_overall_summary(results: list):
    """Print a summary table across all tasks and models."""
    console.print("\n")
    console.rule("[bold]Overall Summary")

    table = Table(title="Consistency Results")
    table.add_column("Task", style="cyan")
    table.add_column("Category")
    table.add_column("Difficulty")
    table.add_column("Model", style="blue")
    table.add_column("Seq Sim", justify="right")
    table.add_column("Arg Con", justify="right")
    table.add_column("Unique Seqs", justify="right")
    table.add_column("Div Point", justify="right")
    table.add_column("Output Match", justify="right")

    for r in results:
        m = r["metrics"]
        if "error" in m:
            table.add_row(r["task_id"], r["category"], r["difficulty"], r["model"], "ERR", "", "", "", "")
            continue

        seq_sim = m["tool_sequence_similarity"]
        color = "green" if seq_sim > 0.8 else "yellow" if seq_sim > 0.5 else "red"
        table.add_row(
            r["task_id"],
            r["category"],
            r["difficulty"],
            r["model"],
            f"[{color}]{seq_sim:.2f}[/{color}]",
            f"{m['argument_consistency']:.2f}",
            f"{m['unique_sequences']}/{m['n_runs']}",
            f"{m['divergence_point']:.1f}" if m["divergence_point"] else "-",
            f"{m['output_agreement']['exact_match_rate']:.0%}",
        )

    console.print(table)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run agent consistency experiment")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_TASK)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    run_experiment(
        models=args.models,
        runs_per_task=args.runs,
        task_ids=args.tasks,
        category=args.category,
        output_dir=args.output,
    )
