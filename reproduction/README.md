# LFIO: Inductive Rule Learning and Extraction from Black Box Models

This repository studies Boolean rule learning and extraction from black-box models. The manuscript evaluates a propositional learner with an explicit default class; the repository also contains earlier ILP-inspired demonstrations and a separate recommendation application. It contains three lines of work:

1. **Synthetic XAI pipeline** — train a neural network on data with known Boolean rules, then compare rule-extraction methods (decision tree surrogate, beam-search ILP, LFIO).
2. **Budget-constrained Boolean rule extraction** — validation-selected depth, head polarity and an error/length objective, two initializations, bounded local repair, packed coverage, and classical/recent rule baselines. The revised manuscript is `LFIO_Negation_Paper.tex`.
3. **MovieLens recommender experiments** — sparse item-association scoring with exact contribution accounting, evaluated separately from Boolean distillation.

## Current manuscript (16 October 2026)

[Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting](elsarticle-template-num.pdf)
combines empirical error attribution, Diagnostic LFIO, explicit numerical rule
generalization and certified predictions with hidden inputs. The latest extension
adds exact difference-bound background constraints, an infeasible-input state,
coverage-constrained cost selection and cached residual queries.

The allocation study has 72 six-dataset teacher settings and 1,728 budget
selections. Later extensions reuse the same dataset cohort; the current cost study
uses 648 Original programs and 19,440 policy records. Controlled feature costs and
synthetic background constraints retain their stated limits. Full-workspace
validation records **167 passed**; the latest metadata synchronization changes
only documentation and release information.

- [Current version and package index](docs/research/CURRENT_RELEASE.md),
  [review PDF](LFIO_Negation_Paper_review.pdf), and
  [historical development supplement](LFIO_Development_Supplement.pdf).
- [Implementation and results](docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md),
  [reproduction commands](docs/research/REPRODUCE.md), and
  [submission checklist](docs/research/SUBMISSION_CHECKLIST.md).
- [Current reports and validation](docs/research/contextual_forgetting_2026-10-16/).
- [LaTeX source bundle](docs/research/contextual_forgetting_2026-10-16/KBS_Contextual_Forgetting_LaTeX_sources_2026-10-16.zip),
  [reproducibility code](docs/research/contextual_forgetting_2026-10-16/KBS_Contextual_Forgetting_Reproducibility_Code_2026-10-16.zip),
  and [experiment records](docs/research/contextual_forgetting_2026-10-16/KBS_Contextual_Forgetting_Experiment_Records_2026-10-16.zip).

The selected submission uses Elsevier `elsarticle` with `review,12pt` and
`elsarticle-num` numbered references (62 single-column pages).
The entry is `elsarticle-template-num.tex`; established manuscript filenames
remain compatibility entries. The separate development supplement retains its
historical CAS layout. Build all PDFs with `python build_paper.py --all`; see
[formatting notes](docs/research/KBS_FORMAT.md). Packages are local review
artifacts. Earlier dated records and ZIPs retain their historical identities.

The numerical-input helpers below originated in the September development
studies, which remain documented in the dated protocols and archives.

Numerical inputs can be transformed with `threshold_vocabulary(train, (.25, .5, .75))`
and `apply_vocabulary(inputs, vocabulary)` from `src.lfio_followup`. Fit thresholds
on the fitting partition only. The controlled experiment uses
`RefinedLFIO(depths=(1, 2), candidate_budget=75000, literal_budget=8)` on the resulting
predicates. `OneSlotLFIO` is an experimental validation-policy control, not a renamed
proposed method. The unchanged `RefinedLFIO.prepare_candidates(X)` supports repeated
budget fits through a fingerprint-checked training-only cache.

For the optional supervised vocabulary, call
`mdl_vocabulary(X_fit, teacher_fit)` from `src.lfio_budgeted`; the returned pair
contains predicates and the split audit. Apply predicates with the same
`apply_vocabulary` function. `ScoredWorkLFIO(work_multiplier=32,
allocate_repair=True, random_state=5001, literal_budget=8)` exposes the bounded
search option; set `allocate_repair=False` for the equal-score restart control.
Both submit one training winner per validation cell. Neither changes the frozen
original `RefinedLFIO` implementation.

## Algorithm optimization and one-command experiments (2026-09-16)

The historical optimization study introduced `RefinedLFIO`, adding
pure-rule initialization, bounded add/delete/replace search, and hard total-condition
budgets. Initial and repaired programs are selected on validation only. New studies
include component ablations, fidelity/size frontiers, and small MILP optimality-gap
diagnostics. Full new results must be measured; historical paper numbers are not
replaced by implementation checks.

On Linux, run from this directory (or call the shell script by absolute path):

```bash
./run_all_experiments.sh --setup
./run_all_experiments.sh --check --dataset-root ../dataset
./run_all_experiments.sh --workers 4 --dataset-root ../dataset --output-dir results/optimized_20260916
```

The dataset directory must contain both `ml-latest-small/` and `ml-latest/`, each
with `movies.csv` and `ratings.csv`. `--setup` installs dependencies and pinned
official baseline sources; it does not download MovieLens or start training.
Use `--skip-movielens` explicitly if those data are unavailable. `--skip-recent`
explicitly omits the external neural/explanation methods. A full run includes the
legacy suite, real and generated studies, TT-Sparse, both NeuRules recipes, budget
curves, exact-search diagnostics, native-export/capacity checks and MovieLens
explanation checks. It can take hours; each stage has its own log.

```bash
# Small end-to-end check, without MovieLens or external neural training:
./run_all_experiments.sh --smoke --output-dir results/optimization_smoke
# Resume with exactly the same options, source and environment:
./run_all_experiments.sh --resume --workers 4 --dataset-root ../dataset --output-dir results/optimized_20260916
# Print the full command plan without running anything:
./run_all_experiments.sh --dry-run
```

Read `<run-directory>/summary/SUMMARY.md` after completion. Predictions, rule exports,
validation traces, solver bounds, CSV summaries, frontier figures and stage logs are
kept under that run directory. The shell wrapper accepts `RESEARCH_PYTHON` to select
an interpreter. See [the optimization protocol](docs/research/OPTIMIZATION_2026-09-16.md)
for search details and interpretation limits.

## Revised research study (2026-09-10)

The environment and all experiment families have been checked. The main study compares classification and random-forest rule distillation on four real datasets, ten splits, and 0/10/20% observed-label noise, plus eight generated tasks and four component restrictions. The large MovieLens run uses all eligible training ratings and full-catalogue evaluation for all eligible warm users.

Install `requirements-research.txt` and follow [the reproduction guide](docs/research/REPRODUCE.md). The study preserves negative results and distinguishes improvement over the fixed original learner from comparisons against tuned trees, RIPPER, and RuleFit. Model length is not a human interpretability assessment, and the MovieLens model is a separate supplementary branch.

```powershell
.\.venv\Scripts\python.exe -u run_research_experiments.py --ablations
.\.venv\Scripts\python.exe -u -m src.movielens_scalable --data-path ml-latest
```

Read [the historical Chinese comparison report](docs/research/REPORT_RECENT_BASELINES_2026-09-10.md) or [the initial research report](docs/research/REPORT_2026-09-10.md), [the paper source](LFIO_Negation_Paper.tex), or [the compiled PDF](LFIO_Negation_Paper.pdf). The original manuscript is archived under `docs/research/archive/`. Rebuild with `python build_paper.py` after generating the tables.

## Recent-method and explanation comparison

The extension runs pinned official TT-Sparse (2026) and NeuRules (NeurIPS 2025) implementations on the same Boolean inputs, splits and teachers. It compares teacher fidelity, reference-label prediction quality, executable rule conditions, and clean/noisy feature and prediction stability. NeuRules selection includes both API and experimental CLI training profiles.

The recommendation supplement adds SPINRec (AAAI 2026) path attributions and the SIGIR 2025 fixed-budget faithfulness framework on the frozen full MovieLens scorers. Its additive-model specialization is checked against official code; this is not a reproduction of SPINRec's neural recommender training benchmark.

Read [the recent-method comparison report](docs/research/REPORT_RECENT_BASELINES_2026-09-10.md). Follow [the exact recent-baseline protocol and reproduction commands](docs/research/RECENT_BASELINES_PROTOCOL.md). Detailed traces remain in `results/`; summary tables and figures are generated in `docs/research/recent_baselines_2026-09-10/`.

## Project Structure

```
xai-ilp/
├── src/
│   ├── data_generator.py         # Synthetic Boolean datasets with known rules
│   ├── black_box.py              # PyTorch MLP black box + trainer
│   ├── rule_extractor.py         # Decision tree surrogate rule extraction
│   ├── ilp_learner.py            # Beam-search ILP baseline
│   ├── lfio_learner.py           # Core LFIO (LGG, subsumption, inverse resolution)
│   ├── lfio_extended.py          # LFIO with explicit Boolean negation
│   ├── sudoku.py / nqueens.py / hamiltonian_cycle.py   # Constraint-problem datasets
│   ├── sudoku_experiment.py      # Sudoku validation experiment
│   ├── lfio_recommender.py       # LFIO recommender system (rules with support/confidence/lift)
│   ├── movielens_experiment.py   # MovieLens experiment driver
│   └── datasets.py               # ILP dataset utilities and UCI references
├── tests/                        # pytest test suite
├── main.py                       # Synthetic XAI pipeline entry point
├── run_lfio_negation_benchmarks.py   # Negation paper benchmarks
├── run_movielens_*.sh            # MovieLens experiment wrappers (small / full / large)
├── LFIO_Negation_Paper.tex       # Current manuscript (legacy filename)
└── requirements.txt
```

## Setup

Requires Python 3.10+. On Windows, use the existing `.venv` directly (activation is optional):

```powershell
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe main.py --rule complex --use-lfio --use-negation --no-viz
```

For a new Windows environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The environment checker records dependency versions, imports, CPU computation,
CUDA availability, and optional data/tools in `results/environment.json`. CPU-only
PyTorch is sufficient for the included experiments. No Prolog installation is
required by the current Python implementation.

On Linux/macOS, the shell wrappers accept `.venv/` or `venv/`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All commands below assume you run them **from the project root** with the venv activated.

### MovieLens data (only for the recommender experiments)

The MovieLens datasets are not bundled. Download them from [GroupLens](https://grouplens.org/datasets/movielens/) and unpack into a sibling `dataset/` directory:

```
<parent>/
├── dataset/
│   ├── ml-latest-small/   # movies.csv, ratings.csv
│   └── ml-latest/         # for run_movielens_large.sh
└── xai-ilp/               # this repository
```

## Usage

### Run all locally available experiments

```powershell
.\.venv\Scripts\python.exe -u run_all_experiments.py
```

The runner creates a fresh `results/full_suite_<UTC timestamp>/` directory and
executes dependency checks, tests, all six synthetic tasks at 1,000 samples over
seeds 42–46, four five-seed negation benchmark configurations (clean, black-box,
10% training-label noise, engineered-feature ablation), exact search-backend
comparisons, both Sudoku demos, and the small/full MovieLens wrapper settings.
The large MovieLens run is recorded as unavailable if `dataset/ml-latest` is absent.
Each job has its own log and output directory. `manifest.json` records commands,
source hashes, timing, status, and output checks. Existing results are preserved.
CPU computation threads are fixed to one and plots use the noninteractive Agg backend.

The corrected Sudoku drivers use independent validation for early stopping and
primitive constraint features by default. Historical `sudoku_diagnostic.json` files
are retained as diagnostic records. High scores on supplied constraint summaries
do not establish reasoning from raw boards.
Interactive menus and every possible parameter combination are not separate jobs.

After completion, run `summarize_full_experiments.py <run-directory>` to validate
and summarize the outputs. See [the full-suite record](docs/FULL_EXPERIMENTS_2026-09-10.md)
for the latest completed run, including unsuccessful model outcomes.


### 1. Synthetic XAI pipeline

```bash
python main.py --rule complex --samples 1000 --features 5            # decision tree + ILP
python main.py --rule complex --use-lfio                             # add LFIO
python main.py --rule complex --use-lfio --use-negation              # add LFIO with negation
python main.py --rule all                                            # all rule types
python main.py --interactive                                         # interactive mode
```

Options: `--rule {and,or,xor,majority,complex,nested,all}`, `--samples N`, `--features N`, `--use-lfio`, `--use-negation`, `--sudoku-size {4,9}`, `--no-viz`, `--interactive`.

| Rule Type | Description |
|-----------|-------------|
| `and` | y = 1 if ALL features are 1 |
| `or` | y = 1 if ANY feature is 1 |
| `xor` | y = 1 if ODD number of features are 1 |
| `majority` | y = 1 if MAJORITY of features are 1 |
| `complex` | y = 1 if (x1 AND x2) OR (x3 AND NOT x4) |
| `nested` | y = 1 if (x1 OR x2) AND (x3 OR x4) AND x5 |

### 2. Negation benchmarks (paper experiments)

```bash
python run_lfio_negation_benchmarks.py                    # 5 seeds, all datasets
python run_lfio_negation_benchmarks.py --seed 42 --samples 400 --max-literals 3
python run_lfio_negation_benchmarks.py --no-packaged      # skip the sklearn breast-cancer dataset
python run_lfio_negation_benchmarks.py --target blackbox --output results/benchmark_blackbox.json
python run_lfio_negation_benchmarks.py --label-noise 0.1 --output results/benchmark_noise_10.json
python benchmark_search_backends.py                      # compare bitset / original Python search
```

Compares `LFIO-beam`, `LFIO-full`, `LFIO+Neg.`, `LFIO+Neg. no-canon`, a depth-limited
decision tree, and a majority classifier on generated Boolean tasks (`mixed`,
`dnf`, `hierarchical`), 4×4 Sudoku validity, 4-Queens validity, and Wisconsin
breast cancer. Results are written to `results/lfio_negation_benchmark.json`.

Protocol v2 splits before fitting breast-cancer median thresholds, seeds both
NumPy and Python constraint generators, and records split hashes and thresholds.
The default constraint features are three Sudoku violation predicates and four
N-Queens predicates (row/column/diagonal safety and queen count). Direct validity,
known-solution and completion shortcuts are excluded. `--constraint-features
engineered` enables the full feature set as a diagnostic ablation; corrected
generators still apply, so this does not reproduce historical data exactly.

`--target blackbox` trains a fixed random forest on training data and distills its
predictions. Test **accuracy** uses ground-truth labels; **fidelity** uses held-out
teacher predictions, and teacher accuracy is reported separately. `--label-noise`
flips only training labels. Every method sees the same targets. Results include
class recalls, precision, F1, balanced accuracy, rule size, runtime, and seed means
and standard deviations. The tree depth bound is not a matched DNF rule budget;
these are basic reference models, not a state-of-the-art comparison.

`--include-reference` adds the original Python scoring backend. The dedicated
`benchmark_search_backends.py` checks identical rules and all 256 Boolean states
on three 8-feature tasks across three seeds while timing both backends.

### 3. MovieLens recommender

```bash
./run_movielens_exp.sh     # 200 users/items on ml-latest-small, strict thresholds
./run_movielens_full.sh    # all users/items on ml-latest-small
./run_movielens_large.sh   # all users/items on ml-latest
```

Or directly:

```bash
python src/movielens_experiment.py --data-path ../dataset/ml-latest-small \
    --max-users 200 --max-items 200 \
    --min-support 0.05 --min-confidence 0.7 --min-lift 1.5 --max-rules 30
```

Results are saved to `results/movielens/` (override with `--output-dir`).
The driver holds out the latest 20% of ratings globally before filtering the
catalogue or computing rating features. Evaluation uses unseen movies from the
training catalogue, includes users receiving no recommendations, and reports a
training-popularity baseline. Users need at least one unseen positive test item
in the catalogue; cold users/items and other exclusions are counted separately.
Precision@K uses K as its denominator, including when fewer movies are returned.

On Windows, run the project environment directly from PowerShell:

```powershell
.\.venv\Scripts\python.exe src/movielens_experiment.py --data-path ../dataset/ml-latest-small --max-users -1 --max-items -1 --min-support 0.02 --min-confidence 0.6 --min-lift 1.2 --max-rules 50 --output-dir results/movielens_full_temporal
```

`-1` removes the size caps; the minimum training interaction filter still applies.
The output includes rules, aggregate metrics with split hashes/configuration, and
per-user metrics. The saved rules belong to the model actually evaluated.
See [the measured MovieLens development results](docs/MOVIELENS_EXPERIMENT_2026-09-10.md).
`ml-latest-small` is a development snapshot, not a stable publication benchmark.

## Tests

```bash
python -m pytest tests/ -v
```

The suite covers the core Boolean LFIO learner, the explicit-negation extension (including the paper's claims that negation and full feature-value LFIO recover the same Boolean boundary, and that canonicalization removes duplicate candidates), the Sudoku/N-Queens dataset generators, and a benchmark smoke test. MovieLens regression tests cover temporal splitting, training-only features, seen-item exclusion, abstention-aware metrics, rule serialization, and CLI settings.

## Methods

### LFIO (Learning from Interpretations and Observations)

An ILP algorithm that learns Horn clauses from input–output examples:

1. **Generates initial rules** from positive examples as ground clauses
2. **Applies LGG (Least General Generalization)** to generalize compatible rules
3. **Uses subsumption checking** to remove redundant rules
4. **Applies inverse resolution** for rule refinement
5. **Filters by coverage and precision** to keep only high-quality rules

Implementations: `LFIO` (general engine), `LFIOBoolean` (binary feature matrices), and `LFIOBlackBox` (mimics a trained black box's predictions).

```python
from src.lfio_learner import LFIOBoolean, LFIOBlackBox

lfio = LFIOBoolean(min_coverage=5, min_precision=0.9)
lfio.fit(X_train, y_train, feature_names)
lfio.print_rules()

lfio_bb = LFIOBlackBox(black_box_model, min_coverage=0.05, min_fidelity=0.9)
lfio_bb.fit(X_train, feature_names)
lfio_bb.print_rules()
```

### LFIO with Explicit Boolean Negation

`LFIOBooleanWithNegation` (in `src/lfio_extended.py`) represents body literals as
triples `(feature, value, negated)` and evaluates `¬(x_f = v)` by direct inequality
over complete Boolean interpretations. Since `¬(x_f = 0) ≡ (x_f = 1)`, equivalent
tests are canonicalized while display polarity is retained. The default `bitset`
backend precomputes literal coverage and counts conjunction matches with integer
AND/population-count operations. `backend="python"` retains exhaustive scalar
scoring for reproducibility. Both preserve candidate ordering and tie-breaking.
Non-Boolean and missing values are rejected instead of silently changing negation
semantics. Canonicalization savings depend on feature count and rule length;
without explicit negation, the duplicate-reduction statistic is zero.

The synthetic neural pipeline now uses separate training/validation/test sets,
seeds PyTorch as well as data generation, and restores an independent copy of the
best validation checkpoint. `--seed` controls reproducibility; `--use-negation`
also appears in method comparisons and works with `--rule all`.

See [the research and environment review](docs/RESEARCH_REVIEW_2026-09-10.md) and
[recorded validation](docs/validation/RESULTS.md) for sources, measured results,
and limitations. The manuscript's experiment section uses the revised protocol.

```python
from src.lfio_extended import LFIOBooleanWithNegation

lfio_neg = LFIOBooleanWithNegation(
    min_coverage=3, min_precision=0.9, max_literals=3,
    allow_negation=True, canonicalize=True, prefer_negated_complement=True,
)
lfio_neg.fit(X_train, y_train, feature_names)
lfio_neg.print_rules()   # e.g. valid <- no_row_duplicates, NOT has_box_duplicates
```

### Baselines

- **Decision tree surrogate** (`rule_extractor.py`) — fits a tree to mimic the black box and reads rules off tree paths.
- **Beam-search ILP** (`ilp_learner.py`) — greedy conjunctive rule search with coverage/precision thresholds.

## Example Output

```
True Rule: y = 1 if (x1 AND x2) OR (x3 AND NOT x4), else 0

Black Box Performance:
  Train Accuracy: 0.992
  Test Accuracy: 0.980

=== LFIO Black Box Rules ===
1. IF x1=1 AND x2=1 THEN y = 1
   Coverage: 59.08%, Fidelity: 100.00%
2. IF x3=1 AND x4=0 THEN y = 1
   Coverage: 100.00%, Fidelity: 100.00%

LFIO Performance:
  Test Accuracy: 1.000
  Fidelity to Black Box: 1.000
```
