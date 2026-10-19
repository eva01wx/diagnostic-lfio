"""Exact numerical feature projection and sound predictions without those inputs.

For a single-head DNF F and forgotten raw coordinates S, possible = exists S F
and certain = forall S F. Quantification ranges over all finite real values,
not an empirical range or independent Boolean threshold assignments. No
cross-feature feasibility constraints are assumed. A resource-limited universal
construction falls back to false and is explicitly marked conservative.
"""
import math

import numpy as np

from .rule_generalization import (RuleProgram, normalize, _box, _body, _Work,
                                 _ProofLimit, _subtract, _difference)


class _BoxLimit(Exception):
    pass


def _constant(parent, value):
    return RuleProgram(((),) if value else (), parent.head, parent.n_features)


def _project(program, features):
    # Normalize FIRST: dropping an impossible interval must not make it possible.
    clean = normalize(program)
    return normalize(RuleProgram(tuple(tuple(a for a in r if a.feature not in features)
                                       for r in clean.rules), clean.head, clean.n_features))


def _complement(program, work, max_boxes, trace):
    full = tuple((-math.inf, math.inf) for _ in range(program.n_features))
    remaining, peak, started = [full], 1, work.used
    for rule in normalize(program).rules:
        cover = _box(rule, program.n_features)
        pieces = []
        for box in remaining:
            pieces.extend(_subtract(box, cover, work))
            peak = max(peak, len(pieces))
            if len(pieces) > max_boxes:
                raise _BoxLimit
        remaining = pieces
        if not remaining:
            break
    result = normalize(RuleProgram(tuple(_body(b) for b in remaining),
                                   program.head, program.n_features))
    trace.append(dict(operation='complement_union_by_interval_subtraction',
        input_sha256=program.fingerprint, output_sha256=result.fingerprint,
        box_subtractions=work.used-started, peak_boxes=peak))
    return result


class FeatureForgetter:
    """Compile necessary/sufficient conditions after globally forgetting features.

    fit takes an exported numerical RuleProgram, not training observations.
    predict returns the original head, its opposite default, or -1 (abstain).
    Forgotten columns may contain NaN and are never consulted. Alternatively,
    retained_only=True accepts just the ordered retained coordinates.

    Certificates concern this surrogate under its numerical semantics, not its
    teacher, causal effects or the correctness of the underlying classification.
    """
    def __init__(self, *, max_nodes=50000, max_boxes=4096):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (max_nodes, max_boxes)):
            raise ValueError('Resource limits must be positive integers')
        self.max_nodes, self.max_boxes = max_nodes, max_boxes

    def fit(self, program, features):
        if not isinstance(program, RuleProgram):
            raise ValueError('Expected a numerical RuleProgram')
        features = tuple(features)
        if any(isinstance(f, bool) or not isinstance(f, (int, np.integer))
               or not 0 <= f < program.n_features for f in features):
            raise ValueError('Forgotten features must be valid raw-coordinate indices')
        self.original_ = program
        self.features_ = tuple(sorted(set(map(int, features))))
        self.retained_features_ = tuple(f for f in range(program.n_features) if f not in self.features_)
        self.possible_ = _project(program, self.features_)
        trace = [dict(operation='normalize_then_existentially_project_raw_coordinates',
            input_sha256=program.fingerprint, output_sha256=self.possible_.fingerprint)]
        work, exact, reason = _Work(self.max_nodes), True, None
        used = {a.feature for r in normalize(program).rules for a in r}
        if not used.intersection(self.features_):
            self.certain_ = self.possible_
            trace.append(dict(operation='unused_coordinate_identity',
                output_sha256=self.certain_.fingerprint))
        else:
            try:
                negative = _complement(program, work, self.max_boxes, trace)
                projected_negative = _project(negative, self.features_)
                trace.append(dict(operation='existentially_project_complement',
                    input_sha256=negative.fingerprint, output_sha256=projected_negative.fingerprint))
                self.certain_ = _complement(projected_negative, work, self.max_boxes, trace)
            except (_ProofLimit, _BoxLimit) as error:
                exact = False
                reason = 'box subtraction limit' if isinstance(error, _ProofLimit) else 'intermediate box limit'
                self.certain_ = _constant(program, False)
        self.audit_ = dict(status='exact' if exact else 'conservative', possible_exact=True,
            certain_exact=exact, reason=reason, box_subtractions=work.used,
            max_nodes=self.max_nodes, max_boxes=self.max_boxes, trace=trace,
            quantified_domain='all finite real-valued completions; no cross-coordinate constraints',
            interval_semantics='exclusive lower, inclusive upper; unbounded ends allowed',
            guarantee='certain implies original head; original head implies possible, for every completion',
            probabilistic_assumptions_required=False,
            globally_minimum_description_claim=False)
        for result in (self.possible_, self.certain_):
            assert not any(a.feature in self.features_ for r in result.rules for a in r)
        return self

    def _inputs(self, X, retained_only):
        if not hasattr(self, 'original_'):
            raise ValueError('fit must be called first')
        X = np.asarray(X, dtype=float)
        width = len(self.retained_features_) if retained_only else self.original_.n_features
        if X.ndim != 2 or X.shape[1] != width:
            raise ValueError('Input width must match the declared full or retained coordinate order')
        retained = X if retained_only else X[:, self.retained_features_]
        if not np.isfinite(retained).all():
            raise ValueError('Retained feature values must be finite')
        filled = np.zeros((len(X), self.original_.n_features))
        filled[:, self.retained_features_] = retained
        return filled

    def bounds(self, X, *, retained_only=False):
        values = self._inputs(X, retained_only)
        head = self.original_.head
        certain = self.certain_.predict(values) == head
        possible = self.possible_.predict(values) == head
        assert np.all(~certain | possible)
        prediction = np.full(len(values), -1, dtype=np.int8)
        prediction[certain] = head
        prediction[~possible] = 1-head
        return dict(certain_head=certain, possible_head=possible,
                    uncertain=possible & ~certain, prediction=prediction)

    def predict(self, X, *, retained_only=False):
        return self.bounds(X, retained_only=retained_only)['prediction']

    def get_rules(self, feature_names=None):
        if not hasattr(self, 'original_'):
            raise ValueError('fit must be called first')
        names = list(feature_names) if feature_names is not None else [f'x{j}' for j in range(self.original_.n_features)]
        if len(names) != self.original_.n_features:
            raise ValueError('Feature name count differs from input width')
        def formula(program):
            return ' OR '.join('(' + (' AND '.join(
                f'{names[a.feature]} {">" if a.greater else "<="} {a.threshold!r}' for a in r) or 'TRUE') + ')'
                for r in program.rules) or 'FALSE'
        lower, upper = formula(self.certain_), formula(self.possible_)
        return dict(certain_head=lower, possible_head=upper, certain_default=f'NOT ({upper})',
                    abstain=f'({upper}) AND NOT ({lower})', head=self.original_.head,
                    default_class=1-self.original_.head, certain_is_exact=self.audit_['certain_exact'])

    def explain(self, row, *, retained_only=False):
        """Explain a partial prediction and provide symbolic completion regions.

        Regions use interval endpoints, not rounded sample witnesses. Their
        unbounded endpoints do not denote infinite-valued model inputs.
        """
        values = self._inputs(np.asarray(row, dtype=float).reshape(1, -1), retained_only)
        bound = self.bounds(values)
        prediction = int(bound['prediction'][0])
        status = ('certain_head' if bound['certain_head'][0] else
                  'certain_default' if not bound['possible_head'][0] else
                  'ambiguous' if self.audit_['certain_exact'] else 'unresolved')
        compatible = []
        for rule in normalize(self.original_).rules:
            if all(bool(values[0, a.feature] > a.threshold) == a.greater
                   for a in rule if a.feature not in self.features_):
                compatible.append(tuple(a for a in rule if a.feature in self.features_))
        completions = normalize(RuleProgram(tuple(compatible), self.original_.head, self.original_.n_features))
        head_region = _box(completions.rules[0], self.original_.n_features) if completions.rules else None
        work, default_status = _Work(self.max_nodes), 'exact'
        try:
            default_region = _difference(_constant(self.original_, True), completions, work)
        except _ProofLimit:
            default_region, default_status = None, 'unknown: box subtraction limit'
        def export_region(region):
            return None if region is None else {str(f):dict(
                lower=None if not math.isfinite(region[f][0]) else region[f][0],
                upper=None if not math.isfinite(region[f][1]) else region[f][1],
                lower_inclusive=False, upper_inclusive=True) for f in self.features_}
        return dict(prediction=prediction, status=status, forgotten_features=list(self.features_),
            retained_values={str(f):float(values[0,f]) for f in self.retained_features_},
            head_completion_region=export_region(head_region),
            default_completion_region=export_region(default_region),
            default_region_search=default_status,
            scope=self.audit_['quantified_domain'],
            interpretation='Surrogate prediction under missing inputs, not a causal or teacher explanation')

    def export(self):
        if not hasattr(self, 'original_'):
            raise ValueError('fit must be called first')
        return dict(representation='numerical feature forgetting envelope', original=self.original_.export(),
            forgotten_features=list(self.features_), retained_features=list(self.retained_features_),
            possible=self.possible_.export(), certain=self.certain_.export(), audit=self.audit_)

    @classmethod
    def from_export(cls, record):
        """Recompute a certificate; do not trust a serialized success flag."""
        if record.get('representation') != 'numerical feature forgetting envelope':
            raise ValueError('Unsupported forgetting representation')
        audit = record['audit']
        result = cls(max_nodes=audit['max_nodes'], max_boxes=audit['max_boxes']).fit(
            RuleProgram.from_export(record['original']), record['forgotten_features'])
        if result.export() != record:
            raise ValueError('Forgetting record does not match recomputed semantics')
        return result
