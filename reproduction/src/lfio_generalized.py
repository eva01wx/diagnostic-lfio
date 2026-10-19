"""Diagnostic LFIO with an explicit, separately auditable generalization stage."""
import numpy as np

from .lfio_diagnostic import DiagnosticLFIO
from .rule_generalization import RuleGeneralizer, RuleProgram, _matrix, _labels


class GeneralizedLFIO:
    """Fit a rule surrogate, then compress or generalize its explicit rules.

    mode='exact' is lossless over the supported input domain. mode='calibrated'
    requires a fourth, separate data partition. Selection and calibration rows
    must not overlap fitting rows or each other. Disjointness alone does not
    establish iid sampling or protection under distribution shift.
    """
    def __init__(self, literal_budget=8, total_score_budget=1200000, random_state=0,
                 *, mode='exact', epsilon=.02, delta=.05, confidence='simultaneous'):
        if mode not in ('exact', 'calibrated'):
            raise ValueError('Choose exact or calibrated deployment')
        self.literal_budget, self.total_score_budget = literal_budget, total_score_budget
        self.random_state, self.mode = random_state, mode
        self.epsilon, self.delta, self.confidence = epsilon, delta, confidence

    def fit(self, X, y, X_selection, y_selection, *, X_calibration=None,
            y_calibration=None, feature_names=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError('Expected numerical fitting inputs')
        X = _matrix(X, X.shape[1]); y = _labels(y, len(X))
        selection = _matrix(X_selection, X.shape[1])
        sy = _labels(y_selection, len(selection))
        partitions = [X, selection]
        if self.mode == 'calibrated':
            if X_calibration is None or y_calibration is None:
                raise ValueError('Calibrated generalization requires a separate calibration partition')
            calibration = _matrix(X_calibration, X.shape[1])
            cy = _labels(y_calibration, len(calibration))
            partitions.append(calibration)
        elif X_calibration is not None or y_calibration is not None:
            raise ValueError('Exact mode does not use calibration observations')
        for i, first in enumerate(partitions):
            for second in partitions[i+1:]:
                joined = np.concatenate((first, second))
                _, inverse = np.unique(joined, axis=0, return_inverse=True)
                if set(inverse[:len(first)]) & set(inverse[len(first):]):
                    raise ValueError('Fitting, selection and calibration raw groups must be disjoint')
        self.feature_names_ = ([f'x{j}' for j in range(X.shape[1])] if feature_names is None
                               else list(feature_names))
        if len(self.feature_names_) != X.shape[1]:
            raise ValueError('Feature name count differs from input width')
        self.base_ = DiagnosticLFIO(self.literal_budget, self.total_score_budget,
                                   self.random_state).fit(X, y, selection, sy)
        base = self.base_.model_
        symbolic = dict(head=base.head_, default_class=1-base.head_,
            vocabulary=self.base_.vocabulary_,
            rules=[list(rule.literals) for rule in base._selected])
        self.original_ = RuleProgram.from_lfio(symbolic, X.shape[1])
        self.generalizer_ = RuleGeneralizer(mode=self.mode, epsilon=self.epsilon,
            delta=self.delta, confidence=self.confidence).fit(
                self.original_, X if self.mode == 'calibrated' else None,
                y if self.mode == 'calibrated' else None)
        if self.mode == 'calibrated':
            self.generalizer_.calibrate(calibration, cy)
        self.program_ = self.generalizer_.program_
        if self.program_.complexity[0] > self.literal_budget:
            raise AssertionError('Generalization exceeded the deployment condition budget')
        return self

    def predict(self, X):
        if not hasattr(self, 'program_'):
            raise ValueError('fit must be called first')
        return self.program_.predict(X)

    def get_rules(self):
        if not hasattr(self, 'program_'):
            raise ValueError('fit must be called first')
        rules = []
        for body in self.program_.rules:
            text = ' AND '.join(f'{self.feature_names_[a.feature]} {">" if a.greater else "<="} {a.threshold!r}'
                                for a in body) or 'TRUE'
            rules.append(f'IF {text} THEN class {self.program_.head}')
        return dict(rules=rules, default_class=1-self.program_.head)

    def explain(self, row):
        """Return the executable decision trace; this is not a causal explanation."""
        if not hasattr(self, 'program_'):
            raise ValueError('fit must be called first')
        values = _matrix(np.asarray(row, dtype=float).reshape(1, -1),self.program_.n_features)[0]
        trace = []
        for index, body in enumerate(self.program_.rules):
            conditions, matched = [], True
            for atom in body:
                result = bool(values[atom.feature] > atom.threshold)
                if not atom.greater:
                    result = not result
                conditions.append(dict(feature=self.feature_names_[atom.feature],
                    value=float(values[atom.feature]), operator='>' if atom.greater else '<=',
                    threshold=atom.threshold, satisfied=result))
                if not result:
                    matched = False; break
            trace.append(dict(rule=index, conditions=conditions, matched=matched))
            if matched:
                return dict(prediction=self.program_.head, matched_rule=index,
                            used_default=False, trace=trace)
        return dict(prediction=1-self.program_.head, matched_rule=None, used_default=True, trace=trace)

    def export(self):
        if not hasattr(self, 'program_'):
            raise ValueError('fit must be called first')
        return dict(feature_names=self.feature_names_, original=self.original_.export(),
            generalized=self.generalizer_.export(), controller=self.base_.audit_)
