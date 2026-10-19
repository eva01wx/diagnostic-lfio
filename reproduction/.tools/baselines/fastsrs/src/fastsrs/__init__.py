"""FastSRS: Fast Rashomon Sets of Sparse Rule Sets.

Source code for the paper *"Fast Rashomon Sets of Sparse Rule Sets"* by
Cristina Molero-Río, Boxuan Li, Tong Wang, and Cynthia Rudin.

Quick start::

    from fastsrs import ORS, load_binary_csv, merge_interval, merge_logical

    X, y, col_ls = load_binary_csv("datasets/heart.csv")
    model = ORS(X, y, col_ls, "fpgrowth")
    model.set_parameters(c1=0.00025, c2=0.001, c3=0.00025 / 3)
    model.set_fixed_bounds()
    model.generate_rules(5, 2, 2000, method="fpgrowth", criteria="precision")
    grs, maps = model.train(500, 0.25, False)
    merge_interval(model, grs)
    merge_logical(model, grs)
    model.printMRS(grs)

Submodules
----------
optimal
    ``ORS`` learner for a single optimal sparse rule set (re-exported here).
rashomon_epsilon
    ``ORS`` variant whose ``train()`` also returns every rule set visited so an
    epsilon-Rashomon set can be extracted with :func:`get_epsilon_rashomon`.
rashomon_nsize
    ``ORS`` variant that collects a size-bounded Rashomon set.
two_step
    ``ORS`` variant with two-step training (subsample warm-up, then full data).
two_step_rashomon_epsilon
    Two-step trainer combined with epsilon-Rashomon-set collection.
nobounds
    Ablation: no Theorem-1/2 bounds during rule screening.
proprules
    Rule-count statistics through the screening pipeline.
util
    Preprocessing, prediction, metric and diversity helpers.
"""

__version__ = "0.1.2"

from .util import (
    preprocess_data,
    predict_MRS,
    predict_MRS_mix,
    getConfusion,
    accuracy,
    merge_interval,
    merge_logical,
    calculate_rules,
    calculate_conditions,
    calculate_values,
    prediction_diversity,
    structural_diversity,
    canonical_form,
    get_epsilon_rashomon,
)
from .data import load_binary_csv
from .optimal import ORS

__all__ = [
    "__version__",
    "ORS",
    "load_binary_csv",
    "preprocess_data",
    "predict_MRS",
    "predict_MRS_mix",
    "getConfusion",
    "accuracy",
    "merge_interval",
    "merge_logical",
    "calculate_rules",
    "calculate_conditions",
    "calculate_values",
    "prediction_diversity",
    "structural_diversity",
    "canonical_form",
    "get_epsilon_rashomon",
]
