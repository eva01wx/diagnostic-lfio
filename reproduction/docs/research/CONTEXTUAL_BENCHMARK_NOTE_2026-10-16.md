# Separate CPU measurement from allocation tracing

After the frozen execution and audit, the original representation benchmark
contained CPU measurements made while tracemalloc was active. Keep those raw
records. Allocation instrumentation can distort runtime comparisons, so add a
separate, untraced measurement on exactly the same 32 cases and 64-row batches.
Do not change cases, query limits, algorithms, cost policies or endpoints.

`benchmark_contextual_forgetting.py` freezes its own source hash and the input
benchmark hash before measurement. Use three fresh repetitions of setup, first
batch and warm batch for each engine, alternating engine order by case. Record
per-case medians. Report these CPU values separately from the original traced
Python-allocation peaks. The deliberate stress subset remains unsuitable for
claiming general deployment speed or total resident-memory improvement.
