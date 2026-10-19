# Execution and reporting notes

The independent protocol was frozen at 2026-10-15 15:52:12 UTC, before its first model fit. Work continued after midnight on 16 October China Standard Time. The directory date identifies the protocol cohort, not a claim that every artifact existed on that date.

All 72 settings completed all eight model families under the original source hashes. No new-data test outcome changed a switching threshold, vocabulary rule, hyperparameter, dataset choice, or model-selection decision.

The separate real-task certificate runner initially failed to serialize `numpy.bool_` in `gap_certified`. Its original manifest and four successful records were retained. `execution_amendment.json` records old/new source hashes and the explicit cast to built-in `bool`; mathematical constraints, objective, fitting rows, endpoints and solver limits were unchanged. Failed serialization attempts are not counted as successful certificates.

Replay identified one single-class test teacher: Climate, seed 6001, clean RF predicts only class one. The frozen `score` function averages agreement over classes present in its target partition and falls back to ordinary agreement for a single class. The final paper calls this observed-class **macro fidelity**, distinguishing it from two-class balanced fidelity. Its frozen primary contrast remains unchanged. An additional two-class-only sensitivity excludes this one fold, preserving all 72 settings in ordinary-fidelity and feasibility reporting. No model is refitted or selected by this reporting clarification.

An absent feasible native baseline candidate remains an explicit infeasibility. A solver timeout remains an unresolved optimum with any valid dual bound retained. Neither is replaced by a hidden fallback, a false zero gap, or an unreported discarded setting.

Final relocation checks harmonized the machine-readable `numbers.json` method means with the tables' dataset-equal aggregation. The earlier summary used a feasible-split mean, which differed for NeuRules because its feasible split counts vary across datasets. Tables, paired contrasts, frozen primary results, model fitting and selection are unchanged.

The frozen `DiagnosticLFIO` class docstring describes the probe share as one third. Its executable allocations and the prespecified protocol are `W/24 + 3W/24 + W/24 + W/24 = W/4`, one quarter. The paper reports the executed quarter allocation. The historical docstring is retained with its original source hash; this note corrects that prose without rewriting a frozen experimental source.
