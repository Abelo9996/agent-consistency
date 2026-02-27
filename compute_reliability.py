"""Compute split-half reliability and basic correctness proxy."""
import json, os
import numpy as np
from scipy import stats
from src.analysis.generate_figures import load_results

results = load_results()

print("=== SPLIT-HALF RELIABILITY ===")
# For each task-model, split 10 runs into odd/even, compute TSS and AC for each half
# Then correlate the halves

from src.metrics.consistency import tool_sequence_similarity, argument_consistency

odd_tss, even_tss = [], []
odd_ac, even_ac = [], []

for r in results:
    m = r.get("metrics", {})
    if "error" in m:
        continue
    traces = r.get("traces", [])
    if len(traces) < 10:
        continue
    
    odd_traces = [traces[i] for i in range(0, len(traces), 2)]
    even_traces = [traces[i] for i in range(1, len(traces), 2)]
    
    # Compute TSS for each half
    odd_seqs = [t.get("tool_sequence", []) for t in odd_traces]
    even_seqs = [t.get("tool_sequence", []) for t in even_traces]
    
    if odd_seqs and even_seqs:
        # Use existing metric on each half
        odd_tss_val = tool_sequence_similarity(odd_seqs)
        even_tss_val = tool_sequence_similarity(even_seqs)
        odd_tss.append(odd_tss_val)
        even_tss.append(even_tss_val)

if odd_tss:
    r_tss, p_tss = stats.pearsonr(odd_tss, even_tss)
    print(f"TSS split-half: r={r_tss:.3f}, p={p_tss:.2e}, n={len(odd_tss)}")
else:
    print("Could not compute split-half (missing trace data)")

# Basic correctness proxy: did the agent produce a final response (not error/timeout)?
print("\n=== CORRECTNESS PROXY ===")
for r in results:
    m = r.get("metrics", {})
    if "error" not in m:
        traces = r.get("traces", [])
        completed = sum(1 for t in traces if t.get("final_response") and not t.get("error"))
        total = len(traces)
        # Just print a few
