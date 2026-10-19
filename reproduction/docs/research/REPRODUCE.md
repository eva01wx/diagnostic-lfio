# Reproduce Diagnostic LFIO experiments

Run commands from `reproduction/` with the Python 3.12 environment activated,
unless a different location is explicitly stated. The [root README](../../../README.md)
contains installation and verification commands. Versions are pinned in
`requirements-major-revision.txt`.

## Restore the frozen records

From the repository root:

```bash
python tools/verify_release.py
python tools/restore_records.py --check
python tools/restore_records.py
cd reproduction
python fetch_prospective_dependencies.py
python fetch_prospective_dependencies.py --verify-only
python -m pytest -q tests
```

The archive restores five `results/` trees: `prospective_20261015`,
`rule_generalization_20261016`, `feature_forgetting_20261016`,
`feature_forgetting_20261016_v2`, and `contextual_forgetting_20261016`.
All original result payloads, including failed-run provenance and amendments,
are retained byte for byte. Restoration checks hashes and refuses conflicting
files. FastSRS is bundled with its license; pinned NeuRules sources are fetched
and hash-checked. Downloads need network access.

## Replay and audit recorded experiments

```bash
python generate_major_revision.py --replay-workers 16
python generate_rule_generalization.py
python generate_feature_forgetting.py --workers 16
python generate_contextual_forgetting.py --workers 12
```

These commands verify source/data locks, saved models, selection decisions and
prediction vectors, then regenerate numerical reports under `docs/research/`.
They do not refit the saved teachers or students. Generated report tables and
figures are experimental outputs. The contextual generator also supports
`--report-only` to regenerate summaries from existing audited CSVs and benchmarks;
that option does not rerun the full audit.

The allocation stage covers six datasets, two native numerical teachers, three
seeds and clean/20% noisy supervision: 72 settings and 1,728 budget selections.
The generalization stage uses new splits of the same datasets. The forgetting
stages reuse the resulting source programs, including 648 Original programs for
the contextual/cost study. These stages do not add independent datasets.

## Fresh execution

Use a separate working copy without restored `results/` records. Fetch and verify
the pinned dependencies there. Run the stages in this order:

```bash
python run_prospective_revision.py --prepare-only
python run_prospective_revision.py --workers 16
python certify_real_revision.py --workers 6
python generate_major_revision.py --replay-workers 16

python run_rule_generalization.py --prepare-only
python run_rule_generalization.py --workers 16
python generate_rule_generalization.py

python run_feature_forgetting.py --prepare-only
python run_feature_forgetting.py --workers 16
python generate_feature_forgetting.py --workers 16

python run_contextual_forgetting.py --prepare-only
python run_contextual_forgetting.py --workers 12
python benchmark_contextual_forgetting.py
python generate_contextual_forgetting.py --workers 12
```

The `--prepare-only` commands freeze source, data and configuration hashes before
execution. Existing manifests reject incompatible sources; retain the original
locks when replaying frozen records. Fresh fits can differ across environments;
compare regenerated results separately rather than replacing the frozen archive.

See the fixed protocols for [allocation](MAJOR_REVISION_PROTOCOL_2026-10-15.md),
[generalization](RULE_GENERALIZATION_PROTOCOL_2026-10-16.md),
[feature forgetting](FEATURE_FORGETTING_PROTOCOL_2026-10-16.md), its
[identity-case amendment](FEATURE_FORGETTING_AMENDMENT_2026-10-16.md), and
[contextual forgetting](CONTEXTUAL_FORGETTING_PROTOCOL_2026-10-16.md).
The [execution notes](major_revision_2026-10-15/EXECUTION_NOTES.md) explain
certificate serialization and the observed-class macro-fidelity convention.
The [benchmark note](CONTEXTUAL_BENCHMARK_NOTE_2026-10-16.md) separates CPU
repetitions from traced Python allocation measurements.

## Interpretation

Source-surrogate certificates are conditional on the declared completion domain.
They do not certify teacher correctness. Empty feasible completion sets and
resource-limited proof failures remain explicit states. Controlled costs,
synthetic constraints and structural compression retain their stated limits.
