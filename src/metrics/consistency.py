"""
Consistency metrics for comparing agent traces across repeated runs.

Core insight: we compare traces at multiple granularities:
1. Tool sequence (which tools, in what order)
2. Argument consistency (same args to same tools)
3. Output agreement (same final answer)
4. Divergence point (where does inconsistency first appear)
"""

import json
from collections import Counter
from itertools import combinations
from typing import Optional
import numpy as np


def _seq_edit_distance(s1: tuple, s2: tuple) -> int:
    """Levenshtein edit distance on sequences of tokens."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def tool_sequence_similarity(traces: list) -> float:
    """
    Normalized edit distance between tool call sequences across all pairs of traces.
    Returns 0.0 (completely different) to 1.0 (identical sequences).
    """
    sequences = [tuple(t.tool_sequence) for t in traces]
    if len(sequences) < 2:
        return 1.0

    similarities = []
    for s1, s2 in combinations(sequences, 2):
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            similarities.append(1.0)
            continue
        # Compute edit distance on sequences using LCS-based approach
        dist = _seq_edit_distance(s1, s2)
        similarities.append(1.0 - dist / max_len)

    return float(np.mean(similarities))


def unique_tool_sequences(traces: list) -> int:
    """Count distinct tool call sequences."""
    sequences = [tuple(t.tool_sequence) for t in traces]
    return len(set(sequences))


def argument_consistency(traces: list) -> float:
    """
    Measure how consistent tool arguments are across runs.
    For each step position, compare the arguments used.
    Uses Jaccard similarity on flattened key-value pairs.
    """
    if len(traces) < 2:
        return 1.0

    # Align by step position
    max_steps = max(len(t.tool_calls) for t in traces)
    if max_steps == 0:
        return 1.0

    step_similarities = []
    for step_idx in range(max_steps):
        step_args = []
        for trace in traces:
            if step_idx < len(trace.tool_calls):
                tc = trace.tool_calls[step_idx]
                # Flatten args to comparable set
                flat = _flatten_args(tc.tool_name, tc.arguments)
                step_args.append(flat)

        if len(step_args) < 2:
            continue

        # Pairwise Jaccard similarity
        for a1, a2 in combinations(step_args, 2):
            if not a1 and not a2:
                step_similarities.append(1.0)
            elif not a1 or not a2:
                step_similarities.append(0.0)
            else:
                intersection = a1 & a2
                union = a1 | a2
                step_similarities.append(len(intersection) / len(union) if union else 1.0)

    return float(np.mean(step_similarities)) if step_similarities else 1.0


def _flatten_args(tool_name: str, args: dict) -> frozenset:
    """Flatten arguments dict to a set of (key, value) tuples for comparison."""
    items = set()
    for k, v in args.items():
        if isinstance(v, list):
            v = tuple(sorted(str(x) for x in v))
        items.add((k, str(v)))
    return frozenset(items)


def output_agreement(traces: list) -> dict:
    """
    Measure agreement of final responses.
    Returns exact match rate and semantic similarity (if available).
    """
    responses = [t.final_response.strip() for t in traces if t.final_response]
    if len(responses) < 2:
        return {"exact_match_rate": 1.0, "unique_responses": 1}

    # Exact match: what fraction of pairs are identical
    total_pairs = 0
    matching_pairs = 0
    for r1, r2 in combinations(responses, 2):
        total_pairs += 1
        if r1 == r2:
            matching_pairs += 1

    unique = len(set(responses))

    return {
        "exact_match_rate": matching_pairs / total_pairs if total_pairs > 0 else 1.0,
        "unique_responses": unique,
        "total_runs": len(responses),
    }


def divergence_point(traces: list) -> Optional[float]:
    """
    Find the average step at which traces first diverge.
    Returns the mean step index (1-indexed) where the first disagreement occurs.
    None if all traces are identical.
    """
    if len(traces) < 2:
        return None

    divergence_steps = []
    for t1, t2 in combinations(traces, 2):
        seq1 = t1.tool_sequence
        seq2 = t2.tool_sequence
        max_len = max(len(seq1), len(seq2))

        found = False
        for i in range(max_len):
            tool1 = seq1[i] if i < len(seq1) else None
            tool2 = seq2[i] if i < len(seq2) else None
            if tool1 != tool2:
                divergence_steps.append(i + 1)  # 1-indexed
                found = True
                break

        if not found:
            # Sequences are identical in tool names; check args
            for i in range(min(len(t1.tool_calls), len(t2.tool_calls))):
                if t1.tool_calls[i].arguments != t2.tool_calls[i].arguments:
                    divergence_steps.append(i + 1)
                    found = True
                    break

    if not divergence_steps:
        return None  # All traces identical

    return float(np.mean(divergence_steps))


def tool_call_count_stats(traces: list) -> dict:
    """Stats on number of tool calls per run."""
    counts = [len(t.tool_calls) for t in traces]
    return {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "cv": float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0,
    }


def compute_all_metrics(traces: list) -> dict:
    """Compute all consistency metrics for a set of traces."""
    return {
        "n_runs": len(traces),
        "tool_sequence_similarity": tool_sequence_similarity(traces),
        "unique_sequences": unique_tool_sequences(traces),
        "argument_consistency": argument_consistency(traces),
        "output_agreement": output_agreement(traces),
        "divergence_point": divergence_point(traces),
        "tool_call_stats": tool_call_count_stats(traces),
    }
