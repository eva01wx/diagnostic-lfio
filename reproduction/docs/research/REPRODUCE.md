## Selected submission format (19 October 2026)

Use `elsarticle-template-num.tex` / `elsarticle-template-num.pdf` as the submission
entry. The manuscript uses `\documentclass[review,12pt]{elsarticle}` and
`\bibliographystyle{elsarticle-num}`: 62 single-column review pages.
`LFIO_Negation_Paper.tex` remains the shared scientific source; the review
filename remains a compatibility wrapper for the same layout. The historical
development supplement remains a separate 16-page CAS document. Build all
entries with `python build_paper.py --all`. The current source ZIP contains a
self-contained flat `elsarticle-template-num.tex` for LaTeX submission.
See [format notes](KBS_FORMAT.md). Scientific code, input records and results
are unchanged by this format migration.

## Current contextual and cost-aware forgetting extension (2026-10-16)

The current manuscript is **Compact Rule Distillation with Diagnostic Allocation
and Certified Feature Forgetting**. The extension adds exact difference-bound
background constraints, infeasibility handling, cached residual queries and
coverage-constrained input-cost selection. See
[the implementation and results guide](CONTEXTUAL_FORGETTING_2026-10-16.md).

```bash
venv/bin/python generate_contextual_forgetting.py --workers 12
venv/bin/python -m pytest -q tests/test_contextual_forgetting.py tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
venv/bin/python build_paper.py --all
venv/bin/python package_contextual_forgetting.py
```

The current release is `docs/research/contextual_forgetting_2026-10-16/`.
[CURRENT_RELEASE.md](CURRENT_RELEASE.md) lists the title, PDFs and three packages.
Use `package_contextual_forgetting.py` to regenerate the current packages.
Commands in historical sections below describe their own archived snapshots;
run historical packaging scripts only in a copy of the corresponding snapshot.
It reuses 648 known Original programs and does not add independent datasets.
The preceding stages and releases below preserve their historical identities.

## Preceding numerical feature-forgetting extension (2026-10-16)

The preceding extension added global raw-coordinate forgetting and necessary/sufficient
conditions for source-surrogate decisions under partial inputs. See
[FEATURE_FORGETTING_2026-10-16.md](FEATURE_FORGETTING_2026-10-16.md) for the API,
assumptions, fixed-cohort results and narrow identity-case amendment.

```bash
venv/bin/python -m pytest -q tests/test_feature_forgetting.py
venv/bin/python generate_feature_forgetting.py --workers 16
venv/bin/python build_paper.py --all
venv/bin/python package_feature_forgetting.py
```

This extension reuses 1,296 source programs (648 existing fits, two versions each),
not newly independent data or training. That stage's records are under
`results/feature_forgetting_20261016_v2`; the initial run is retained separately.
A new run in an isolated copy starts with `run_feature_forgetting.py --prepare-only`
then `run_feature_forgetting.py --workers 16`. Frozen source locks intentionally
reject a modified implementation in an already populated output directory.
That stage's historical release folder is `docs/research/feature_forgetting_2026-10-16/`.
Earlier dated ZIPs remain historical snapshots of their own source versions.

# Reproduce the revised research study

## Explicit rule generalization: 16 October 2026

`GeneralizedLFIO` in `src/lfio_generalized.py` adds explicit condition deletion,
rule merging, and numerical equivalence compression. Its exact default retains
global predictions; calibrated deployment requires a separate calibration set.
See [implementation, usage and results](RULE_GENERALIZATION_2026-10-16.md) and
the [fixed extension protocol](RULE_GENERALIZATION_PROTOCOL_2026-10-16.md).

```bash
venv/bin/python -m pytest -q tests/test_rule_generalization.py tests/test_lfio_generalized.py
venv/bin/python generate_rule_generalization.py
venv/bin/python build_paper.py --all
```

The extension contains 648 original fits and six variants (3,888 result rows),
with new seeds on the same six datasets. It keeps the October 15 source locks
intact. Refitting uses `run_rule_generalization.py --prepare-only` followed by
`run_rule_generalization.py --workers 16` in a separate directory. The public
wrapper and reusable generalizer export both programs, proof traces, grow-only
paths and calibration records. Group-disjoint calibration does not by itself
establish the iid assumptions for population binomial coverage. Structural
savings are not evidence of human comprehension.

## Earlier allocation revision: protocol frozen 15 October, completed 16 October 2026

The earlier allocation-stage manuscript was **Diagnosing and Allocating Effort in Compact Rule
Distillation**. New algorithm code is `src/lfio_diagnostic.py`; the independent
extension and audited report are in `major_revision_2026-10-15/`. The earlier
experimental sources and dated records remain unchanged. The development
material is rendered separately in `LFIO_Development_Supplement.pdf`.

The [frozen protocol](MAJOR_REVISION_PROTOCOL_2026-10-15.md) fixes six previously
unused UCI datasets, two native numerical teachers, three seeds and clean/20%
noisy supervision: 72 settings, eight methods and 1,728 budget selections.
FastSRS uses its official pinned implementation; NeuRules retains continuous
predicates. The [execution notes](major_revision_2026-10-15/EXECUTION_NOTES.md)
record a certificate-serialization amendment and the single-class-test-teacher
sensitivity. Neither changes model fitting or selection.

Use Python 3.12 and the package versions in `requirements-major-revision.txt`.
The allocation-stage code and experiment-record ZIPs are designed to extract into
the same directory. The code package includes MIT-licensed FastSRS; pinned
NeuRules files are fetched rather than redistributed.

```bash
venv/bin/python fetch_prospective_dependencies.py --verify-only
venv/bin/python generate_major_revision.py --replay-workers 16
venv/bin/python -m pytest -q tests/test_diagnostic_revision.py
venv/bin/python build_paper.py --all
```

If upstream files are absent, run `fetch_prospective_dependencies.py` without
`--verify-only`; it checks downloaded SHA-256 hashes. The generator verifies
sources/data/splits, fitting-only vocabularies, model exports, selection,
resource caps, 216 cached training-only decision replays and all real-task
certificate records. Remove only a copied `controller_replay.json` in a separate
verification directory to force every internal probe to rerun; the published
cache itself is source-hash checked.

For a fresh training reproduction, extract into a **separate directory without
recorded `results/prospective_20261015`**; never overwrite the frozen records.

```bash
venv/bin/python run_prospective_revision.py --prepare-only
venv/bin/python run_prospective_revision.py --workers 16
venv/bin/python certify_real_revision.py --workers 6
venv/bin/python generate_major_revision.py --replay-workers 16
```

The 15-second oracle allowance applies to optimization, not matrix construction
or all solver setup; the largest real-task diagnostic took several minutes.
Timeouts retain bounds and are not claimed optimal. The frozen scoring routine
averages over observed teacher classes: one clean-RF Climate test fold is
single-class, so the paper labels this macro fidelity and also reports a
17-fold two-class-only sensitivity. Feasibility and ordinary fidelity retain
all settings. Only the clean-RF eight-condition median-restarts contrast is
primary. Fixed MDL outperforms the conservative controller; the primary gain's
interval includes zero. The procedure is not presented as a generally superior
optimizer.

The submission-source ZIP is flattened for a LaTeX editor and includes main,
review and development-supplement inputs. `build_paper.py --all` builds all three
in the workspace. `package_major_revision.py` produces that stage's source,
code and experiment-record ZIPs. September ZIPs remain historical snapshots.
Relocation verification reuses the installed environment and pinned NeuRules
files; it is not a clean dependency-installation test.

## Earlier literature and diagnostic revision: 15 October 2026

That earlier manuscript emphasized search, vocabulary, and capacity diagnosis.
The existing training records and September evidence remain its numerical base.
Five additional primary-source references and their relevance are documented in
[the literature review](REFERENCES_2026-10-15.md); see also
[the revision record](REVISION_2026-10-15.md).

The new analysis is retrospective and does not retrain or select models. It
reconstructs fitting-only predicates, verifies frozen source/data/split hashes,
replays the eight-condition students, and computes empirical encoding ceilings
separately on each partition of both native-teacher cohorts:

```bash
MPLCONFIGDIR=/tmp/xai_vocabulary_mpl venv/bin/python generate_vocabulary_diagnostic.py
venv/bin/python -m pytest -q tests/test_vocabulary_diagnostic.py
venv/bin/python build_paper.py --all
venv/bin/python audit_recent_manuscript.py --output docs/research/paper_2026-10-15/manuscript_audit.json --render-dir results/tex_build/cas_20261015
venv/bin/python audit_recent_manuscript.py --paper LFIO_Negation_Paper_review.pdf --output docs/research/paper_2026-10-15/review_manuscript_audit.json --render-dir results/tex_build/cas_review_20261015
```

New outputs are in `paper_2026-10-15/`: 900 vocabulary/partition records from
120 settings, dataset and cohort summaries, a 960-prediction replay audit,
and the new manuscript table. The upper bound is exact for the observed sample,
uses that partition's teacher outputs, and is not a deployable model or a
population guarantee. The later cohort's median is diagnostic only. The dated
September submission ZIPs represent their original snapshots; the revised PDF
and current sources are the October manuscript.

## Archived KBS manuscript: 24 September 2026

That archived version's inputs and reports are in `paper_2026-09-24/`; presentation tables,
PDF audits and three deliverable ZIPs are in `kbs_2026-09-24/`. See the
[checklist audit](CHECKLIST_AUDIT_2026-09-24.md), [revision record](REVISION_2026-09-24.md)
and [frozen protocol](SEARCH_VOCABULARY_PROTOCOL_2026-09-24.md).
The prior core and cohorts remain unchanged. New optional implementations are
in `src/lfio_budgeted.py`; do not mix their results with the original defaults.

Extract the code and experiment-record archives into the same empty directory.
The supplied `SUPPLEMENT_README.md` explains Python 3.12, pinned package versions
and restoring the two upstream baseline repositories with
`python fetch_paper_baselines.py --fetch`. The scientific supplement excludes
unlicensed upstream checkouts, Tectonic resources and the separate MovieLens data.

Regenerate recorded evidence in this order, using the same environment:

```bash
venv/bin/python generate_budget_manuscript.py --run-dir results/optimized_full --external-dir results/external_20260920 --output-dir docs/research/paper_2026-09-24
venv/bin/python generate_strengthened_manuscript.py --output-dir docs/research/paper_2026-09-24
venv/bin/python generate_checklist_revision.py
venv/bin/python build_paper.py --all
venv/bin/python audit_recent_manuscript.py
venv/bin/python audit_recent_manuscript.py --paper LFIO_Negation_Paper_review.pdf --output docs/research/kbs_2026-09-24/review_manuscript_audit.json --render-dir results/tex_build/cas_review_20260924
venv/bin/python docs/research/kbs_2026-09-24/audit_citations.py
venv/bin/python prepare_kbs_submission.py
venv/bin/python prepare_reproducibility_bundle.py
```

The last generator invokes the earlier review generator with its output redirected
to that historical evidence directory, then audits its 120 settings and
5,040 candidate exports. JSON mappings compare by values, independently of key
ordering. A relocation check reuses the existing Python environment and pinned
upstream checkouts; it is not a clean dependency-installation test.

Fresh training uses a new directory to preserve the recorded results:

```bash
venv/bin/python run_checklist_experiments.py --output-dir results/checklist_rerun --freeze-only
venv/bin/python run_checklist_experiments.py --output-dir results/checklist_rerun --block work --workers 16
venv/bin/python run_checklist_experiments.py --output-dir results/checklist_rerun --block vocabulary --workers 16
```

The manuscript generator deliberately audits `results/checklist_extension_20260924`;
fresh runs are compared separately. Changing frozen sources or package versions
cannot silently resume the recorded run. Equal scoring counts are checked within
each paired grid cell; costs include preprocessing where appropriate. Four primary
contrasts were fixed before fitting, and all their descriptive intervals cross zero.
The adaptive vocabulary is retained as a cost/size option without a fidelity
non-inferiority claim. The full regression suite passes 109 tests with eight
preexisting upstream warnings.

After visual inspection of the rendered pages, the dated verification scripts
check the flat LaTeX package, extraction/regeneration of the scientific supplement,
and final file hashes. `verify_final_revision.py` requires completed visual audits.
No publishing or submission is performed by these commands.

## Archived KBS revision: 22 September 2026

That revision's evidence is `paper_2026-09-22/`, with CAS presentation tables and
submission/reproducibility archives in `kbs_2026-09-22/`. The 21 September
experiment source and predictions remain unchanged. The new implementation is
`src/lfio_followup.py`; its controls and numerical vocabularies were fixed in
`FOLLOWUP_PROTOCOL_2026-09-22.md` before running new seeds 4001--4005.

For the accompanying archives, extract **both** the code and experiment-record
ZIP files into the same directory. Read `SUPPLEMENT_README.md` there. External
baseline repositories without explicit redistribution terms are not copied
into the code ZIP; exact URLs/revisions and a fetch/check script are supplied.
Original source fingerprints require restoring the pinned source checkouts.
Python 3.12 and the package versions in the run manifests/lock files reproduce
the recorded environment. Tectonic is needed only for rebuilding the PDF.

The following commands describe that archived code snapshot and its September
build and packaging defaults. Use the current October instructions above for
the working tree. With the archived dependencies and pinned sources available:

```bash
# The existing completed outputs resume only with unchanged source/data/package hashes.
venv/bin/python run_review_followup.py --freeze-only
venv/bin/python run_review_followup.py --block mechanism --workers 32
venv/bin/python run_review_followup.py --block native --workers 24
# Audit all 160 settings, recompute predictions/selections, and render evidence.
venv/bin/python generate_review_revision.py
venv/bin/python -m pytest -q
venv/bin/python build_paper.py --all
venv/bin/python audit_recent_manuscript.py
venv/bin/python audit_recent_manuscript.py --paper LFIO_Negation_Paper_review.pdf --output docs/research/kbs_2026-09-22/review_manuscript_audit.json --render-dir results/tex_build/cas_review_20260922
venv/bin/python prepare_kbs_submission.py
venv/bin/python prepare_reproducibility_bundle.py
```

For a fresh rerun, use an empty output directory with `run_review_followup.py
--output-dir ...`. This preserves the original recorded experiment. The paper
generator intentionally audits `results/review_followup_20260922`, the frozen
22 September snapshot; compare a fresh rerun separately. A source or
package change cannot silently resume that snapshot. No default or threshold
recipe was selected after viewing follow-up test results.

The 100 mechanism settings have five methods/controls and two budgets; the
60 native-teacher settings have five method/vocabulary configurations and four
budgets: 800 method grids, 2,200 selections, 5,800 fitted candidate records.
That revision's full regression suite had 105 passing tests. Equal validation-slot checks
are not equal-runtime checks. The numerical comparison fixes depth two in
both vocabulary arms and charges each literal threshold as a condition.
Post-hoc repair examples are selected explicitly for illustration, and their
exported programs are replayed against recorded selected predictions.

The new report includes a copy of earlier numerical evidence needed for the
paper. To regenerate that evidence from raw historical records, first run the
older commands below with `--output-dir docs/research/paper_2026-09-22`; then run
`generate_review_revision.py` **last**, as it adds both tree encodings, the
balanced-fidelity diagnostics, actual-size plots and new follow-up tables.
Original protocol filenames remain dated 21 September by design.

## Initial KBS extension: 21 September 2026

That revision's evidence was generated into `docs/research/paper_2026-09-21/`.
Presentation-only CAS table copies and the flat source archive are under
`docs/research/kbs_2026-09-21/`. The article presents Refined LFIO with component
ablations. The independent MovieLens study is not an input to the new extension.
See [the frozen protocol](STRENGTHENING_PROTOCOL_2026-09-21.md),
[the revision record](REVISION_2026-09-21.md), and [format notes](KBS_FORMAT.md).

Use Python 3.12, `requirements-research.txt`, and the pinned baseline setup from
`./run_all_experiments.sh --setup`. The new Boolean baseline is vendored in
`research_baselines/vendor_brcg/` with its Apache license and exact commit.
Its additional dependencies are CVXPY 1.7.3 and CLARABEL 0.11.1. Neural dependencies
and their checkouts are unchanged. Complete versions are in the run manifest.
The four new datasets, download links, and SHA-256 hashes are distributed under
`data/research/extension_2026-09-21/`; no dataset download is required for this block.

```bash
venv/bin/pip install -r requirements-research.txt
# Freeze sources and execute all classical/ablation/sensitivity settings.
MPLCONFIGDIR=/tmp/xai_matplotlib venv/bin/python run_strengthening_experiments.py --block classical --workers 32
# Recorded adapter for RuleFit's three single-class fitting-target errors.
# Run after the first command exits, even if it reports those expected errors.
venv/bin/python repair_strengthening_constants.py
# Expanded small-architecture neural grids on the same main settings.
MPLCONFIGDIR=/tmp/xai_matplotlib venv/bin/python run_strengthening_experiments.py --block neural --workers 16
# Preserve undefined single-class pruning AUC as null, with identical predictions.
venv/bin/python repair_strengthening_neural_metadata.py
# Regenerate retained earlier evidence (requires the earlier local runs).
venv/bin/python generate_budget_manuscript.py --run-dir results/optimized_full --external-dir results/external_20260920 --output-dir docs/research/paper_2026-09-21
# Audit the new evidence, recompute metrics/selections, and generate new tables.
venv/bin/python generate_strengthened_manuscript.py
venv/bin/python -m pytest -q
venv/bin/python build_paper.py --all
venv/bin/python audit_recent_manuscript.py
venv/bin/python audit_recent_manuscript.py --paper LFIO_Negation_Paper_review.pdf --output docs/research/kbs_2026-09-21/review_audit.json --render-dir results/tex_build/cas_review_20260921
venv/bin/python prepare_kbs_submission.py
```

The constant-target adapter does not inspect test scores or tune parameters.
It checks that the fitting teacher labels contain one class and emits that
constant prediction with zero conditions. Its source hash, affected settings,
and original execution errors are retained in `constant_target_amendment.json`
and `classical_completion_before_amendment.json`. The original frozen source
manifest remains unchanged. Subsequent resumes reuse these exact checkpoints.
The failed RuleFit attempts are not included in corrected constant-fit timing.
All other original successes are retained. This is an explicit implementation
amendment, not an unreported replacement of test results.

The same three single-class validation targets give TT-Sparse undefined AUC
metadata. `repair_strengthening_neural_metadata.py` preserves training/pruning
and maps only these log values to JSON null. It rejects nonfinite losses or
selection scores and verifies every saved prediction against the original
pre-serialization-failure arrays. `neural_metadata_amendment.json` records the
paths and adapter hash; the original neural completion errors are retained.
Run this adapter after the neural block exits, including when that block
reports its three JSON errors. Both amendments are checked by the final audit.

The new runner refuses resume with changed source, data, protocol or package
versions. Choose a fresh `--output-dir` for a new execution; the repair and
report scripts accept the matching `--run-dir`. The two blocks may run
concurrently in the same output directory because their method files differ.
A clean rebuild requires the recorded Python/package versions; bit-for-bit
PDF identity additionally depends on the TeX/font environment.

To recreate the earlier evidence if local outputs are unavailable:

```bash
./run_all_experiments.sh --workers 4 --dataset-root /path/to/MovieLens/datasets --output-dir results/optimized_full
venv/bin/python run_external_validation.py --workers 4 --output-dir results/external_20260920
```

Earlier manifests include the source at their execution time; use the archived
snapshot when reproducing those exact hashes. The current cache/work-counter
change preserves LFIO predictions and selection, as checked by regression
tests. See `OPTIMIZATION_2026-09-16.md` and
`EXTERNAL_VALIDATION_PROTOCOL_2026-09-20.md` for the original protocols.

Tectonic 0.17.0 under `.tools/tectonic/tectonic` (or on PATH) builds both layouts.
The build uses a project-local TeX cache. PDF auditing requires PyMuPDF and
Pillow, renders every page, and resets visual review until the images have
been inspected. Both PDFs use the same manuscript body. The official CAS
files are verified against the author-supplied archive in `docs/vendor/els-cas/`.

Large prediction/checkpoint traces under `results/` remain local generated
artifacts. The committed reporting folder contains aggregated and per-setting
metrics, provenance, source hashes, and audits. Numerical reproduction and
byte-for-byte artifact reproduction are distinct. Historical instructions
below describe earlier snapshots, not the current paper's reporting inputs.

## Historical September 10 protocol

The paper concerns Boolean rule extraction. MovieLens is a separate supplementary application. All results here are exploratory; the new seeds do not constitute independent population replications or preregistration.

## Environment and data

From the repository root, use Python 3.12 and install `requirements-research.txt`. The tested environment is recorded in each manifest; `requirements-research-lock.txt` is the exact installed research environment. Windows commands below use the existing `.venv`; replace that executable with `.venv/bin/python` on Linux.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe -m pytest -q
```

The three UCI text datasets are in `data/research/`, with original source descriptions, attribution, URLs, and archive checksums. Breast cancer comes from scikit-learn. All numerical thresholds are calculated independently inside each fitting split. No external dataset is silently sampled.

MovieLens is in the sibling `../dataset/` directory. The large July 2023 snapshot has 33,832,162 ratings; ZIP SHA-256: `66a9e518c747d76b241d9a859b001a2619d3ed1672ceef599eb50daf73a7b4a3`. The complete download/extraction verification is in `docs/validation/movielens_latest_download_2026-09-10.json`. These are development snapshots, not fixed official publication benchmarks.

## Exact experimental commands

```powershell
.\.venv\Scripts\python.exe -u run_all_experiments.py --output-dir results/full_suite_revised_20260910
.\.venv\Scripts\python.exe summarize_full_experiments.py results/full_suite_revised_20260910

.\.venv\Scripts\python.exe -u run_research_experiments.py --ablations --output-dir results/research_real_final

.\.venv\Scripts\python.exe -u run_research_experiments.py --datasets and or xor majority complex nested sudoku4 nqueens4 --seeds 1001 1002 1003 1004 1005 --noise 0 0.1 --targets labels distillation --ablations --workers 2 --output-dir results/research_synthetic_final

.\.venv\Scripts\python.exe -u -m src.movielens_scalable --data-path ml-latest --output-dir results/research_movielens_latest

.\.venv\Scripts\python.exe summarize_research.py
```

The suite refuses to overwrite an existing output directory; choose a fresh name when rerunning it. The research runner can resume completed settings only with identical source, configuration, and dependency signatures. The last standalone MovieLens command duplicates the suite's large sparse study and was originally run first; either verified result can be passed to `summarize_research.py --movie-latest`. The final audit checks that both runs agree.

Main study: four datasets × ten seeds × three noise levels × two target types × eleven methods/variants = 2,640 selected fits. Generated study: eight tasks × five seeds × two noise levels × two target types × eleven methods/variants = 1,760 selected fits. Hyperparameter search makes additional fits. Both training and observed validation labels are corrupted for direct classification; teacher predictions supply both partitions' targets for distillation. Final test labels remain untouched.

Legacy suite: environment, tests, six neural tasks over five seeds, four legacy benchmark variants, nine exact reference/backend pairs, Sudoku sizes four and nine, old small/full MovieLens settings, and sparse small/large MovieLens studies. Historical diagnostic results are preserved and not substituted for corrected independent-validation scores.

## Outputs and audit

`docs/research/results_2026-09-10/` contains all final selected-fit metrics, per-condition means and standard deviations, paired differences, explanation-stability measurements, generated LaTeX tables, scientific figures, data protocols, and the final audit. Full programs, parameter traces, split indices, and per-sample predictions remain under each `results/research_*_final/` setting. Manifests record source/data hashes and software versions; captured library warnings are saved in model JSON files.

The bootstrap intervals summarize paired seed-wise data-set averages. Repeated splits reuse observations; do not interpret these as independent population trials. Rule Jaccard compares canonical head-labelled bodies and is representation-sensitive. Default-class agreement and prediction agreement are reported separately.

MovieLens ranks every eligible warm user against the full training catalogue in batches. It retains every eligible training interaction but bounds the rule vocabulary to 128 popular anchors. The archive, train-filter counts, cold-user/item exclusions, all per-user scores, and score-contribution examples are auditable. Fallback popularity does not count toward explanation coverage.

## Manuscript

`LFIO_Negation_Paper.tex` is the rewritten manuscript. The previous draft is preserved in `docs/research/archive/`. Numbers and tables are generated by the summary script, so run it before building the PDF. Tectonic 0.17.0 is installed locally in `.tools/tectonic/`; the official binary archive was verified against its GitHub release SHA-256. It downloads standard TeX resources on first use. The build helper supplies a project-local Windows font configuration/cache when no custom one is configured.

```powershell
.\.venv\Scripts\python.exe build_paper.py
```

The compiled PDF is `LFIO_Negation_Paper.pdf`. The final report explains what the experiments support and what would still be needed for a stronger publication claim. No manuscript was submitted or publicly posted.

## Recent-method extension

The additional paired TT-Sparse/NeuRules study and SPINRec explanation checks are documented in [RECENT_BASELINES_PROTOCOL.md](RECENT_BASELINES_PROTOCOL.md). It records exact upstream revisions, input/split adaptations, both NeuRules profiles and the distinction between native-model consistency, teacher fidelity, explanation size, and perturbation effects. Use `requirements-recent-baselines-lock.txt` for the original research package versions plus the pinned TT-Sparse source. Run both summary scripts before building the extended manuscript. Historical paper hashes refer to the pre-extension manuscript preserved under `archive/`; the extension produces a new artifact manifest.
