# Diagnostic LFIO

Code and experiments for the paper **Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting**.

Public source code and experiment records: [https://github.com/eva01wx/diagnostic-lfio/](https://github.com/eva01wx/diagnostic-lfio/).

## Get the code and verify the files

```bash
git clone https://github.com/eva01wx/diagnostic-lfio.git
cd diagnostic-lfio
python tools/verify_release.py
```

For SSH, use `git@github.com:eva01wx/diagnostic-lfio.git`.

| Directory | Contents |
| --- | --- |
| `reproduction/src/` | LFIO algorithms, numerical rule generalization and feature forgetting |
| `reproduction/research_baselines/` | Baseline adapters and independent logical oracles |
| `reproduction/tests/` | Focused algorithm and baseline tests |
| `reproduction/data/` | Six datasets, source archives and provenance |
| `reproduction/docs/research/` | Frozen protocols, usage guides, metrics and experiment reports |
| `artifacts/` | Frozen experiment records with member checksums |
| `provenance/` | Repository metadata, record archive metadata and checksums |
| `tools/` | File verification and safe record restoration |

## Install and run the focused tests

Use Python 3.12 and the pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements-major-revision.txt
cd reproduction
python -m pytest -q tests/test_contextual_forgetting.py tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
```

These four modules contain 47 tests. To run all 54 bundled tests, restore the
pinned NeuRules files first:

```bash
python fetch_prospective_dependencies.py
python fetch_prospective_dependencies.py --verify-only
python -m pytest -q tests
```

FastSRS and BRCG sources retain their third-party licenses. NeuRules is fetched
from a fixed commit and verified by SHA-256; fetching requires network access.

## Restore records and replay the experiments

From the repository root, with the environment activated:

```bash
python tools/restore_records.py --check
python tools/restore_records.py
cd reproduction
python fetch_prospective_dependencies.py
python generate_contextual_forgetting.py --workers 12
```

The record archive is approximately 62 MB compressed and 1.02 GB expanded.
Restoration verifies member hashes, skips identical files and refuses conflicting
existing files. Restored records are ignored by Git. The generator replays frozen
choices and predictions; this is distinct from fresh model training.
See the [reproduction guide](reproduction/docs/research/REPRODUCE.md) for all
stages, source locks and fresh execution commands. Use a separate copy for new runs.

## Scope and limits

Diagnostic allocation is implemented in `reproduction/src/lfio_diagnostic.py`;
explicit numerical generalization in `rule_generalization.py` and
`lfio_generalized.py`; global forgetting in `feature_forgetting.py`; constrained
residual queries in `contextual_forgetting.py`; and cost-aware selection in
`forgetting_selection.py`.

Later extensions reuse the same six-dataset cohort. Logical certificates concern
the source surrogate and declared completion domain. Controlled feature costs,
synthetic constraints and structural compression do not establish real acquisition
costs or human comprehension. See the [implementation and limits](reproduction/docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md).
