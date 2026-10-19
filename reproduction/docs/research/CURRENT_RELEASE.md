# Current code and experiment release

Algorithm: Diagnostic LFIO.
Release: `contextual-forgetting-2026-10-16`.

| Component | Location |
| --- | --- |
| Diagnostic allocation | `src/lfio_diagnostic.py` |
| Explicit generalization | `src/rule_generalization.py`, `src/lfio_generalized.py` |
| Global feature forgetting | `src/feature_forgetting.py` |
| Contextual forgetting | `src/contextual_forgetting.py` |
| Cost-aware feature selection | `src/forgetting_selection.py` |
| Latest metrics and audits | `docs/research/contextual_forgetting_2026-10-16/` |
| Frozen records | `artifacts/experiment_records_2026-10-16.zip` at repository root |

Use the [reproduction guide](REPRODUCE.md) to restore records and replay each stage.
Experimental sources, fixed protocols, data, splits and raw records retain their
original bytes. Later extensions reuse the same six datasets; they are not new
independent dataset cohorts. Record replay and fresh training are separate workflows.
