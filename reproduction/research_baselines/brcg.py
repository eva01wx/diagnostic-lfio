"""Thin, audited adapter for the pinned AIX360 BRCG-light implementation."""
import numpy as np
import pandas as pd

from .vendor_brcg.boolean_rule_cg import BooleanRuleCG


def literal_frame(X):
    columns = pd.MultiIndex.from_tuples([(f'x{j}', '==', v)
                                        for j in range(X.shape[1]) for v in (0, 1)])
    return pd.DataFrame(np.column_stack([X[:, j] == v for j in range(X.shape[1])
                                        for v in (0, 1)]).astype(int), columns=columns)


def export_brcg(model):
    z = model.z.loc[:, model.w > .5]
    rules = [[[int(feature[1:]), int(value)] for feature, _, value in
              z.index[z[column] > 0]] for column in z.columns]
    return dict(representation='single-head DNF with opposite default',
                head=0 if model.CNF else 1, default_class=1 if model.CNF else 0,
                rules=rules, total_literals=sum(map(len, rules)),
                column_generation_iterations=model.it,
                generated_columns=model.z.shape[1],
                implementation='AIX360 BRCG-light; beam search and heuristic rounding')


def predict_brcg(export, X):
    covered = np.zeros(len(X), dtype=bool)
    for rule in export['rules']:
        hit = np.ones(len(X), dtype=bool)
        for feature, value in rule:
            hit &= X[:, feature] == value
        covered |= hit
    return np.where(covered, export['head'], export['default_class'])


def grid():
    for penalty in (.0005, .002, .01, .05, .2):
        for cnf in (False, True):
            config = dict(lambda0=penalty, lambda1=penalty, CNF=cnf,
                          iterMax=25, timeMax=10, K=10, D=5, B=5,
                          eps=1e-6, solver='CLARABEL', silent=True)
            yield config, BooleanRuleCG(**config)
