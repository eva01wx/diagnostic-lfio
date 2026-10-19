# Contextual and cost-aware feature forgetting

Current release: contextual-forgetting-2026-10-16.
Title: Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting.
Package index: docs/research/CURRENT_RELEASE.md.
Release directory: docs/research/contextual_forgetting_2026-10-16/.

Extract this code ZIP and KBS_Contextual_Forgetting_Experiment_Records_2026-10-16.zip
together. Python 3.12 and requirements-major-revision.txt define the environment.
Pinned baseline sources/data and all earlier extension provenance are retained.
NeuRules is not bundled; use fetch_prospective_dependencies.py if needed.

python generate_contextual_forgetting.py --workers 12
python -m pytest -q tests/test_contextual_forgetting.py tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
python build_paper.py --all

The four core extension modules contain 47 tests; all five bundled modules
contain 54 tests and require pinned NeuRules sources for the diagnostic module.
The historical full workspace report covers 167 tests;
historical application tests/datasets outside the research release are omitted.
The submission entry is elsarticle-template-num.tex (elsarticle review,12pt)
with numeric references. Build output elsarticle-template-num.pdf is preferred;
the development supplement retains its separate historical CAS layout.
See docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md for guarantees, limits,
fresh execution and the separation of untraced CPU from Python allocation peaks.
This is reuse of known datasets, not additional independent validation.
