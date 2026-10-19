# FastSRS

Source code for the paper *"Fast Rashomon Sets of Sparse Rule Sets"* by Cristina Molero-Río, Boxuan Li, Tong Wang, and Cynthia Rudin, *Machine Learning* 115, 165 (2026). [https://doi.org/10.1007/s10994-026-07108-9](https://doi.org/10.1007/s10994-026-07108-9)

FastSRS learns short, accurate rule sets for binary classification using simulated annealing on a regularized objective. Two flavors are provided:

- **Optimal rule set** (`fastsrs.optimal.ORS`, also exported as `fastsrs.ORS`) — finds a single sparse rule set `R` that minimizes the paper's regularized objective
  `L(R) = (1 - acc(R)) + c1·n_conditions(R) + c2·n_rules(R) + c3·n_values(R)`,
  where `n_rules` is the number of rules (conjunctions), `n_conditions` the total number of conditions (literals) across rules, and `n_values` the total number of attribute values those conditions mention. `c1`, `c2`, `c3` are the paper's `C1`, `C2`, `C3`.
- **ε-Rashomon set** (`fastsrs.rashomon_epsilon.ORS`) — additionally records the rule set proposed at every SA iteration so you can extract the ε-Rashomon set `{R : L(R) ≤ (1+ε)·L(R*)}`, where `R*` is the best rule set found.

A few research variants are also included; see [Package layout](#5-package-layout) and [Reproducibility](#6-reproducibility).

## 1. Installation

FastSRS is a regular Python package (`fastsrs`) and requires Python ≥ 3.9.

### From PyPI

```bash
pip install fastsrs
```

### From GitHub (latest development version)

```bash
pip install "git+https://github.com/mmolerous/FastSRS.git"
```

### From a local clone (for development)

```bash
git clone https://github.com/mmolerous/FastSRS.git
cd FastSRS
pip install -e ".[dev]"      # editable install + pytest/build/twine
pytest                       # quick smoke tests on datasets/heart.csv
```

### Dependencies

`numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib` and [`pyfim`](https://borgelt.net/pyfim.html) (the `fim` frequent-itemset miner) are installed automatically. `pyfim` ships as a C source distribution, so pip needs a C compiler to build it: `gcc`/`clang` on Linux and macOS, or the *Microsoft C++ Build Tools* on Windows. If you would rather not compile, install a prebuilt binary first and then install FastSRS on top:

```bash
conda install -c conda-forge pyfim
pip install fastsrs
```

The exact environment used for the paper's experiments is recorded in [`environment.yml`](https://github.com/mmolerous/FastSRS/blob/main/environment.yml) / [`requirements.txt`](https://github.com/mmolerous/FastSRS/blob/main/requirements.txt) (`conda env create -f environment.yml`); it is not needed to use the package.

## 2. Data format

FastSRS expects an all-binary CSV with a final integer `Class` column (0/1). Each non-target column is a 0/1 indicator. The included [`datasets/`](https://github.com/mmolerous/FastSRS/tree/main/datasets/) directory has seven prepared datasets — `adult`, `compas`, `diabetes`, `fico`, `heart`, `invehicle`, `recidivism` — plus one simulated dataset.

The included CSVs follow these conventions for the indicator-column names:

- **Binary attribute** with values `{a, b}` ⇒ one column `attr_b` (the larger / last-sorted value) and its negation `attr_notb`.
- **Categorical attribute** with values `{a, b, c, …}` ⇒ a column `attr_v` and its negation `attr_notv` for *every* value `v`.
- **Numerical attribute** ⇒ for each of `Nlevel-1` quantile thresholds `t` (default 9 thresholds): `attr_<=t` and `attr_>t`.

## 3. Preparing your own data

`util.preprocess_data` converts a raw `pandas.DataFrame` (or a path to a CSV) into the binary format above. Columns are auto-detected by dtype unless you override them: 2-unique-value → binary, `object`/`category` → categorical, otherwise → numerical.

### Minimal example (clean numeric data)

```python
import pandas as pd
from fastsrs import preprocess_data

raw = pd.DataFrame({
    'sex':  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],         # binary
    'cp':   ['a','b','c','a','b','c','a','b','c','a'],  # categorical (object dtype)
    'age':  [25, 30, 45, 50, 33, 60, 28, 41, 55, 38],   # numerical
    'Class':[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
})

df = preprocess_data(raw, target_col='Class', Nlevel=4)
# Optionally save: preprocess_data(raw, target_col='Class', output_path='datasets/mydata.csv')
```

Resulting columns:

```
sex_1, sex_not1,
cp_a, cp_nota, cp_b, cp_notb, cp_c, cp_notc,
age_<=30.75, age_>30.75, age_<=39.5, age_>39.5, age_<=48.75, age_>48.75,
Class
```

### Overriding auto-detection

Auto-detection treats numeric columns with > 2 unique values as numerical (quantile-binarized). If a numeric integer column is actually categorical (e.g. `cp` with codes 1/2/3/4 in the heart data), pass it explicitly:

```python
preprocess_data(
    raw_df,
    target_col='num',
    binary=['sex', 'fbs', 'exang'],
    categorical=['cp', 'restecg', 'slope', 'ca', 'thal'],
    numerical=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
)
```

### Non-integer class labels

Use `class_map` to map raw labels to 0/1:

```python
preprocess_data(raw, target_col='income',
                class_map={' >50K': 1, ' <=50K': 0})
```

`class_map` accepts a dict or any callable (e.g. `lambda v: 0 if v == 0 else 1` to binarize a multi-class target).

### Other knobs

| parameter | default | purpose |
|---|---|---|
| `Nlevel` | 10 | number of quantiles for numerical binarization (produces `Nlevel-1` thresholds) |
| `include_negations` | `True` | emit both `_v` and `_notv` (and both `_<=t` and `_>t`) |
| `dropna` | `True` | drop rows with NaN in any feature column |
| `missing_value`, `missing_label` | `None` | rename a category in the resulting column names after encoding (e.g. `' ?' → ' int'` for adult.csv) |
| `output_path` | `None` | if given, also write the result to this CSV path |

The `preprocess_data` function reproduces the existing `datasets/heart.csv` and `datasets/adult.csv` exactly (same shape, same cell values) when invoked with the matching options.

## 4. Usage

All examples below assume you have a binary CSV such as [`datasets/heart.csv`](https://github.com/mmolerous/FastSRS/blob/main/datasets/heart.csv). `fastsrs.load_binary_csv` reads it into the `(X, y, col_ls)` triple that `ORS` expects and drops constant columns; the standalone scripts [`test_FastSRS.py`](https://github.com/mmolerous/FastSRS/blob/main/test_FastSRS.py) and [`test_FastSRS_Rashomon_epsilon.py`](https://github.com/mmolerous/FastSRS/blob/main/test_FastSRS_Rashomon_epsilon.py) show the equivalent manual pandas code and can be run directly from the repository root.

### 4.1 Optimal sparse rule set — [`test_FastSRS.py`](https://github.com/mmolerous/FastSRS/blob/main/test_FastSRS.py)

```python
import numpy as np
import random
from fastsrs import *              # ORS, load_binary_csv and the util helpers

# Read dataset
X, y, col_ls = load_binary_csv('datasets/heart.csv')

# Set parameters
supp, maxlen, Nrules = 5, 2, 2000
method = 'fpgrowth'
Niteration, q = 500, 0.25
c2 = 0.001; c1 = c2 / 4; c3 = c1 / 3

# Run FastSRS
random.seed(1); np.random.seed(1)
model = ORS(X, y, col_ls, method)
model.set_parameters(c1=c1, c2=c2, c3=c3)
model.set_fixed_bounds()
model.generate_rules(supp, maxlen, Nrules, method=method, criteria='precision')
grs, maps = model.train(Niteration, q, False)
merge_interval(model, grs)
merge_logical(model, grs)

# Print rules
print("\n===== RULE SET =====")
model.printMRS(grs)

# Compute metrics
Yhat = predict_MRS(grs, X)
TP, FP, TN, FN = getConfusion(Yhat, y)
acc = float(TP + TN) / (TP + TN + FP + FN)
nrules = calculate_rules(grs)
nconditions = calculate_conditions(grs)
nvalues = calculate_values(model, grs)
objvalue = (1 - acc) + model.c2 * nrules + model.c1 * nconditions + model.c3 * nvalues

print("\n===== RESULTS =====")
print('acc', acc)
print('nrules', nrules)
print('nconditions', nconditions)
print('nvalues', nvalues)
print('objvalue', objvalue)
```

Parameters (defaults in the snippet are the ones used in the paper's experiments):

| parameter | meaning |
|---|---|
| `method` | candidate-rule miner passed to `ORS(...)` and `generate_rules(...)`. Use `'fpgrowth'` (FP-growth over the positive examples, as in the paper). Any other value takes an experimental random-forest path. |
| `maxlen` | maximum number of conditions in a mined candidate rule (paper: 2). |
| `Nrules` | size of the candidate pool kept after screening; the miner's output is screened with the Theorem-1 negative-support bound and the best `Nrules` rules by precision are kept (paper: 2000). |
| `supp` | accepted for API compatibility but **not used** by the FP-growth miner, which mines rules whose positive support is at least `max(5 %, Theorem-2 bound)`. |
| `criteria` | accepted for API compatibility; screening always ranks candidates by **precision** (as in the paper). Passing `'IG'` has no effect. |
| `Niteration` | number of simulated-annealing iterations (paper: 500). |
| `q` | exploration probability: with probability `q` an SA move picks a rule/condition uniformly at random instead of greedily by score (paper: 0.25). |
| `c1`, `c2`, `c3` | regularization per condition, per rule and per value (the paper's `C1`, `C2`, `C3`). |

`set_fixed_bounds()` must be called after `set_parameters(...)`: it computes the Theorem-1 (maximum negative support) and Theorem-2 (minimum positive support) bounds from `c1 + c2 + c3` that prune the candidate space. `merge_interval` and `merge_logical` post-process the returned rule set into the readable form printed below (`(thal:7.0 or 6.0)`, `(oldpeak:>1.9)`).

Typical output on `heart` with the parameters above (roughly 6–11 rules, 83–88 % training accuracy, objective ≈0.14–0.18; one example run):

```text
===== RULE SET =====
rule 0:(ca:2.0),(thalach:<=170.0),
rule 1:(cp:4.0),(ca:not0.0),
rule 2:(thal:7.0),(oldpeak:>1.9),
...

===== RESULTS =====
acc 0.8619528619528619
nrules 7
nconditions 18
nvalues 18
objvalue 0.1510471380471381
```

> **Note on reproducibility.** `seed(1)` and `np.random.seed(1)` are set at the start of `train()` and `generate_rules()`, but rule screening uses `joblib.Parallel(n_jobs=-1)` and worker results are collected in completion order. The order of `self.rules` therefore varies slightly between runs, which can change which rule set the SA loop ends up with. Every run gives a valid near-optimal sparse rule set for the objective above; exact rules and numbers will differ from the snippet.

### 4.2 ε-Rashomon set of sparse rule sets — [`test_FastSRS_Rashomon_epsilon.py`](https://github.com/mmolerous/FastSRS/blob/main/test_FastSRS_Rashomon_epsilon.py)

`fastsrs.rashomon_epsilon.ORS.train()` returns one extra value, `Rset`: one entry `[rules, objective, training_accuracy]` for the candidate rule set proposed at *every* SA iteration, whether or not the move was accepted (so `len(Rset) == Niteration`). `fastsrs.get_epsilon_rashomon(Rset, eps)` returns the entries with `objective ≤ (1+eps)·min_objective`, de-duplicated up to rule and value order — the ε-Rashomon set of the paper, restricted to the rule sets the SA run actually visited. Entries in `Rset` are *not* merged; call `merge_interval`/`merge_logical` on a member before printing it.

Two diversity measures from the paper are provided: `prediction_diversity(R1, R2, X)` is the fraction of examples on which the two rule sets predict differently, and `structural_diversity(R1, R2)` is one minus the Jaccard similarity of the sets of attribute values used by the two rule sets.

```python
import numpy as np
import random
from fastsrs import load_binary_csv, merge_interval, merge_logical, predict_MRS, getConfusion, \
    calculate_rules, calculate_conditions, calculate_values, get_epsilon_rashomon, \
    prediction_diversity, structural_diversity
from fastsrs.rashomon_epsilon import ORS   # Rashomon-collecting variant of the learner

# Read dataset (same as above)
X, y, col_ls = load_binary_csv('datasets/heart.csv')

supp, maxlen, Nrules = 5, 2, 2000
method = 'fpgrowth'
Niteration, q = 500, 0.25
c2 = 0.001; c1 = c2 / 4; c3 = c1 / 3

random.seed(1); np.random.seed(1)
model = ORS(X, y, col_ls, method)
model.set_parameters(c1=c1, c2=c2, c3=c3)
model.set_fixed_bounds()
model.generate_rules(supp, maxlen, Nrules, method=method, criteria='precision')
grs, Rset, maps = model.train(Niteration, q, False)        # NOTE: 3-tuple return
merge_interval(model, grs); merge_logical(model, grs)

# Optimal model
print("===== Optimal RULE SET =====")
model.printMRS(grs)
Yhat = predict_MRS(grs, X)
TP, FP, TN, FN = getConfusion(Yhat, y)
acc = float(TP + TN) / (TP + TN + FP + FN)
nrules = calculate_rules(grs)
nconditions = calculate_conditions(grs)
nvalues = calculate_values(model, grs)
objvalue = (1 - acc) + model.c2 * nrules + model.c1 * nconditions + model.c3 * nvalues
print("\n===== Metrics for the optimal RULE SET =====")
print('acc', acc); print('nrules', nrules)
print('nconditions', nconditions); print('nvalues', nvalues)
print('objvalue', objvalue)

# ε-Rashomon set
print("\n===== RASHOMON SET for a given ε =====")
eps = 0.05
print("ε:", eps)
Rset_eps = get_epsilon_rashomon(Rset, eps)
print("Size of the ε-Rashomon set:", len(Rset_eps))

# Compare two random members of the ε-Rashomon set
print("\n===== Metrics for two random models R1 and R2 from the ε-RASHOMON SET =====")
k1, k2 = random.sample(range(len(Rset_eps)), 2)
R1 = Rset_eps[k1][0]    # each Rset entry is (rules, objective, 1-Error)
R2 = Rset_eps[k2][0]
merge_interval(model, R1); merge_logical(model, R1)
merge_interval(model, R2); merge_logical(model, R2)

print("===== R1 ====="); model.printMRS(R1)
print("===== R2 ====="); model.printMRS(R2)
print('Prediction diversity R1-R2:', prediction_diversity(R1, R2, X))
print('Structural diversity R1-R2:', structural_diversity(R1, R2))
```

Typical output on `heart` (values vary run-to-run, see note in §4.1):

```text
===== Metrics for the optimal RULE SET =====
acc         ≈ 0.84–0.87
nrules      ≈ 6–11
nconditions ≈ 15–26
nvalues     ≈ 15–28
objvalue    ≈ 0.15–0.18

===== RASHOMON SET for a given ε =====
ε: 0.05
Size of the ε-Rashomon set: typically 4–20 unique rule sets
```

### 4.3 Behavior under extreme regularization

If `c1+c2+c3` is so large that no candidate rule satisfies the Theorem-1/Theorem-2 support bounds, the rule miner produces nothing and `train()` short-circuits to an **empty rule set**, which predicts class 0 (the negative class) for every example, and emits a warning. The Rashomon variants additionally return an empty Rashomon set. No exception is raised, and downstream helpers (`predict_MRS`, `calculate_*`, `get_epsilon_rashomon`) all handle the empty case cleanly.

## 5. Package layout

The learner lives in `src/fastsrs/`. Each research variant is a submodule that defines its own `ORS` class with the same interface; import the one you need.

| import | purpose |
|---|---|
| `from fastsrs import ORS` (= `fastsrs.optimal`) | optimal sparse rule set (main method) |
| `fastsrs.rashomon_epsilon` | ε-Rashomon set: `train()` also returns `Rset` |
| `fastsrs.rashomon_nsize` | size-bounded Rashomon set: `train(RS, RP, Niteration, q, ...)` returns `(rules, Rset, maps)` where `Rset` holds at most `RS` best-so-far rule sets collected during the last `(1-RP)` fraction of SA iterations |
| `fastsrs.two_step` | two-step (warm-start) training from the paper: SA runs on a 25 % subsample of the data for the first 25 % of iterations, then continues on the full data |
| `fastsrs.two_step_rashomon_epsilon` | two-step trainer + ε-Rashomon-set collection, every iteration evaluated on the full data so both phases are comparable |
| `fastsrs.nobounds` | ablation: no Theorem-1/2 bounds during screening |
| `fastsrs.proprules` | rule-count statistics through the screening pipeline |
| `fastsrs.util` | `preprocess_data`, `predict_MRS`, `merge_interval`, `merge_logical`, `calculate_*`, diversity measures, `get_epsilon_rashomon` (all re-exported from `fastsrs`) |
| `fastsrs.data` | `load_binary_csv` |

`pip install -e .` from a clone installs the package in editable mode, so edits under `src/fastsrs/` are picked up without reinstalling. `pytest` runs the smoke tests in [`tests/`](https://github.com/mmolerous/FastSRS/tree/main/tests/); `python -m build` produces the sdist and wheel in `dist/`.

## 6. Reproducibility

To replicate the results from the main paper, install the package (see [Installation](#1-installation)) and run the following scripts from a clone of the repository. Each script locates the repository root automatically via its own `pathcode` variable (override it if you move the script), reads `datasets/<dataname>.csv`, and writes to `results/`. Set the `dataname` variable near the top of each file to the dataset you want to run.

| script | purpose |
|---|---|
| `run_FastSRS.py` | optimal sparse rule set (main results) |
| `run_FastSRS_two_step.py` | optimal model with two-step training (subsample warm-up + full data) |
| `run_FastSRS_nobounds.py` | ablation: no Theorem-1/2 bounds during screening |
| `run_FastSRS_proportionrules.py` | rule-count statistics through the screening pipeline |
| `run_FastSRS_robustness_study_features.py` | robustness study, perturbing features |
| `run_FastSRS_robustness_study_class.py` | robustness study, perturbing labels |
| `run_FastSRS_Rashomon_epsilon.py` | ε-Rashomon set of sparse rule sets |
| `run_FastSRS_Rashomon_nsize.py` | size-bounded Rashomon set |

The variant module `fastsrs.two_step_rashomon_epsilon` combines the two-step trainer with ε-Rashomon-set collection, evaluating every iteration on the full data so phase-1 (subsample) and phase-2 (full) entries are directly comparable.

## License

This project is released under the [MIT License](https://github.com/mmolerous/FastSRS/blob/main/LICENSE).

## Contact

- Cristina Molero-Río (mmolero@us.es)
- Boxuan Li (bl3011@columbia.edu)

## Citing this work

If you use FastSRS in your research, please cite the paper:

> Molero-Río, C., Li, B., Wang, T., & Rudin, C. (2026). Fast Rashomon Sets of Sparse Rule Sets. *Machine Learning*, 115(7), 165. https://doi.org/10.1007/s10994-026-07108-9

```bibtex
@article{MoleroRio2026FastSRS,
  title     = {Fast {R}ashomon Sets of Sparse Rule Sets},
  author    = {Molero-R{\'i}o, Cristina and Li, Boxuan and Wang, Tong and Rudin, Cynthia},
  journal   = {Machine Learning},
  volume    = {115},
  number    = {7},
  pages     = {165},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/s10994-026-07108-9},
  url       = {https://doi.org/10.1007/s10994-026-07108-9}
}
```
