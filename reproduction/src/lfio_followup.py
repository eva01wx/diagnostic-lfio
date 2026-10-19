"""Controls and numerical predicates for the prospectively fixed follow-up.

RefinedLFIO itself is unchanged. The control changes only how many programs
from each training grid cell can enter validation selection.
"""
import numpy as np

from .lfio_research import RefinedLFIO


class OneSlotLFIO(RefinedLFIO):
    """One training-selected program per depth/penalty/head validation cell."""

    def _fit_variants(self, pool, positive, negative, n_samples, alpha):
        variants = list(super()._fit_variants(pool, positive, negative, n_samples, alpha))
        selected = min(variants, key=lambda v: (
            v[2][-1], sum(len(r.literals) for r in v[1]), len(v[1])))
        name, rules, history, audit = selected
        audit = dict(audit, training_variants=len(variants),
                     validation_policy="one training-objective winner per grid cell")
        yield name, rules, history, audit


def threshold_vocabulary(fitting, quantiles):
    """Fit unsupervised thresholds on fitting rows only; keep median nesting.

    Duplicate threshold values for a feature are removed. Constant predicate
    columns are retained consistently, including when a quantile is an endpoint.
    Each executable threshold test is one condition, including two tests that
    bound an interval on one original feature.
    """
    fitting = np.asarray(fitting, dtype=float)
    if fitting.ndim != 2 or not np.isfinite(fitting).all() or not len(fitting):
        raise ValueError("Expected nonempty finite numeric fitting matrix")
    return [(j, float(t)) for j in range(fitting.shape[1])
            for t in np.unique(np.quantile(fitting[:, j], quantiles))]


def apply_vocabulary(X, vocabulary):
    X = np.asarray(X, dtype=float)
    return np.column_stack([X[:, j] > t for j, t in vocabulary]).astype(np.uint8)


def numeric_tree_dnf(model):
    """Shorter-head leaf-path DNF, preserving sklearn's float32 input semantics."""
    tree = model.tree_
    paths = {int(c): [] for c in model.classes_}

    def visit(node, path):
        if tree.children_left[node] < 0:
            cls = int(model.classes_[tree.value[node].ravel().argmax()])
            paths[cls].append(path)
            return
        condition = [int(tree.feature[node]), float(tree.threshold[node])]
        visit(tree.children_left[node], path + [[*condition, False]])
        visit(tree.children_right[node], path + [[*condition, True]])

    visit(0, [])
    head = min((0, 1), key=lambda c: (sum(map(len, paths.get(c, []))),
                                     len(paths.get(c, [])), c))
    rules = paths.get(head, [])
    return dict(head=head, default_class=1-head, rules=rules,
                total_literals=sum(map(len, rules)),
                all_leaf_literals=sum(len(r) for rs in paths.values() for r in rs),
                input_cast="float32")


def predict_numeric_tree(export, X):
    # sklearn casts input to float32 before walking the tree.
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    covered = np.zeros(len(X), dtype=bool)
    for rule in export["rules"]:
        hit = np.ones(len(X), dtype=bool)
        for feature, threshold, greater in rule:
            hit &= (X[:, feature] > threshold) if greater else (X[:, feature] <= threshold)
        covered |= hit
    return np.where(covered, export["head"], export["default_class"])
