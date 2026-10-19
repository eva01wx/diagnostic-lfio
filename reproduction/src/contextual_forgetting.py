"""Partial-input queries without compiling a global forgetting envelope.

Background knowledge is an explicit conjunction of rational difference bounds
x_left - x_right <= bound (or <); None denotes the constant zero. All arithmetic
used to decide these constraints is exact. No learned correlation is inferred.
The unrestricted fast path shares residual DNF queries, not global formulas.
"""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import json
import math

import numpy as np

from .rule_generalization import (Atom, RuleProgram, normalize, _box, _Work,
                                 _ProofLimit, _difference)


def rational(value):
    # Fraction(float) preserves the actual binary64 boundary, including extremes.
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError('Bounds must be finite')
        return Fraction(float(value))
    return Fraction(value)


@dataclass(frozen=True)
class DifferenceBound:
    left: int | None
    right: int | None
    bound: object = 0
    strict: bool = False

    def __post_init__(self):
        for index in (self.left, self.right):
            if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
                raise ValueError('Constraint coordinates must be nonnegative integers or None')
        if not isinstance(self.strict, bool):
            raise ValueError('strict must be Boolean')
        object.__setattr__(self, 'bound', rational(self.bound))

    def export(self):
        return [self.left, self.right, str(self.bound), self.strict]


class _Budget:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def tick(self):
        if self.used >= self.limit:
            raise _ProofLimit
        self.used += 1


def difference_witness(bounds, features, budget):
    """Bellman--Ford over (rational, infinitesimal coefficient) weights.

    A zero-weight cycle containing a strict edge is inconsistent. For SAT,
    substitute a positive rational epsilon small enough to satisfy every edge.
    Return exact rational values, or None for a proved negative cycle.
    """
    budget.tick()
    zero = features
    edges = [(zero if b.right is None else b.right,
              zero if b.left is None else b.left, b.bound, -int(b.strict)) for b in bounds]
    distance = [(Fraction(0), 0) for _ in range(features+1)]
    for _ in range(features+1):
        changed = False
        for source, target, value, strict in edges:
            budget.tick()
            candidate = (distance[source][0]+value, distance[source][1]+strict)
            if candidate < distance[target]:
                distance[target] = candidate
                changed = True
        if not changed:
            break
    else:
        return None
    epsilon = Fraction(1)
    for source, target, value, strict in edges:
        constant = distance[source][0]+value-distance[target][0]
        coefficient = distance[source][1]+strict-distance[target][1]
        if constant > 0 and coefficient < 0:
            epsilon = min(epsilon, constant/(-coefficient*2))
    origin = distance[zero][0]+distance[zero][1]*epsilon
    values = [a+b*epsilon-origin for a, b in distance[:-1]]
    # The returned witness is checked with exact arithmetic, never a tolerance.
    for bound in bounds:
        left = Fraction(0) if bound.left is None else values[bound.left]
        right = Fraction(0) if bound.right is None else values[bound.right]
        assert left-right < bound.bound if bound.strict else left-right <= bound.bound
    return values


def _atom_bound(atom, negate=False):
    greater = atom.greater != negate
    return (DifferenceBound(None, atom.feature, -rational(atom.threshold), True)
            if greater else DifferenceBound(atom.feature, None, rational(atom.threshold)))


def _box_witness(box):
    values = []
    for lo, hi in box:
        if math.isfinite(lo) and math.isfinite(hi):
            value = (rational(lo)+rational(hi))/2
        elif math.isfinite(lo):
            value = rational(lo)+1
        elif math.isfinite(hi):
            value = rational(hi)
        else:
            value = Fraction(0)
        values.append(value)
    return values


class ContextualForgetter:
    """Certify the source class, abstain (-1), or reject infeasibility (-2).

    ``query`` distinguishes SAT/UNSAT/UNKNOWN for both classes. Its certificates
    hold only under the supplied background bounds. Cached entries store exact
    hidden-coordinate witnesses; retained input values are never replaced.
    """
    def __init__(self, *, max_nodes=50000, max_cache=4096):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (max_nodes, max_cache)):
            raise ValueError('Limits must be positive integers')
        self.max_nodes, self.max_cache = max_nodes, max_cache

    def fit(self, program, features, constraints=()):
        if not isinstance(program, RuleProgram):
            raise ValueError('Expected RuleProgram')
        features = tuple(features)
        if any(isinstance(f, bool) or not isinstance(f, (int, np.integer))
               or not 0 <= f < program.n_features for f in features):
            raise ValueError('Invalid forgotten coordinate')
        self.original_, self.clean_ = program, normalize(program)
        self.features_ = tuple(sorted(set(map(int, features))))
        self.retained_features_ = tuple(f for f in range(program.n_features) if f not in self.features_)
        self.constraints_ = tuple(constraints)
        for bound in self.constraints_:
            if not isinstance(bound, DifferenceBound) or any(
                    f is not None and f >= program.n_features for f in (bound.left, bound.right)):
                raise ValueError('Invalid background constraint')
        self.context_retained_ = tuple(sorted({f for b in self.constraints_ for f in (b.left, b.right)
                                               if f in self.retained_features_}))
        self._cache = OrderedDict()
        self.stats_ = dict(queries=0, cache_hits=0, solved_patterns=0, work=0, unknown_patterns=0,
                           peak_cache_entries=0, evictions=0)
        return self

    def _residual(self, row):
        residual = tuple(sorted(set(tuple(a for a in rule if a.feature in self.features_)
                     for rule in self.clean_.rules if all(
                         bool(row[a.feature] > a.threshold) == a.greater
                         for a in rule if a.feature not in self.features_))))
        return ((),) if () in residual else residual

    def _unconstrained(self, residual):
        source = RuleProgram(residual, self.original_.head, self.original_.n_features)
        head = _box_witness(_box(residual[0], source.n_features)) if residual else None
        work = _Work(self.max_nodes)
        try:
            box = _difference(RuleProgram(((),), source.head, source.n_features), source, work)
            default = None if box is None else _box_witness(box)
            default_status = 'UNSAT' if box is None else 'SAT'
        except _ProofLimit:
            default, default_status = None, 'UNKNOWN'
        return ('SAT' if head is not None else 'UNSAT', head, default_status, default, work.used)

    def _contextual(self, residual, row):
        bounds = list(self.constraints_)
        for f in self.context_retained_:
            value = rational(row[f])
            bounds.extend((DifferenceBound(f, None, value), DifferenceBound(None, f, -value)))
        work = _Budget(self.max_nodes)
        def head_search():
            for rule in residual:
                witness = difference_witness(bounds+[_atom_bound(a) for a in rule],
                                             self.original_.n_features, work)
                if witness is not None:
                    return witness
            return None
        def default_search(index, current):
            witness = difference_witness(current, self.original_.n_features, work)
            if witness is None or index == len(residual):
                return witness
            for atom in residual[index]:
                result = default_search(index+1, current+[_atom_bound(atom, negate=True)])
                if result is not None:
                    return result
            return None
        result = []
        for search in (head_search, lambda: default_search(0, bounds)):
            try:
                witness = search()
                result.extend(('SAT' if witness is not None else 'UNSAT', witness))
            except _ProofLimit:
                result.extend(('UNKNOWN', None))
        return (*result, work.used)

    def query(self, row, *, retained_only=False):
        return self._query(row, retained_only=retained_only, details=True)

    def _query(self, row, *, retained_only, details):
        if not hasattr(self, 'original_'):
            raise ValueError('fit must be called first')
        row = np.asarray(row, dtype=float)
        expected = len(self.retained_features_) if retained_only else self.original_.n_features
        if row.shape != (expected,):
            raise ValueError('Input must be one row with the declared width')
        retained = row if retained_only else row[list(self.retained_features_)]
        if not np.isfinite(retained).all():
            raise ValueError('Retained inputs must be finite')
        values = np.zeros(self.original_.n_features)
        values[list(self.retained_features_)] = retained
        residual = self._residual(values)
        key = (residual, tuple(float(values[f]) for f in self.context_retained_))
        self.stats_['queries'] += 1
        if key in self._cache:
            self.stats_['cache_hits'] += 1
            record = self._cache.pop(key)
        else:
            result = (self._contextual(residual, values) if self.constraints_
                      else self._unconstrained(residual))
            head, hw, default, dw, work = result
            record = (head, tuple(str(hw[f]) for f in self.features_) if hw is not None else None,
                      default, tuple(str(dw[f]) for f in self.features_) if dw is not None else None)
            self.stats_['solved_patterns'] += 1
            self.stats_['work'] += work
            self.stats_['unknown_patterns'] += int('UNKNOWN' in (head, default))
        self._cache[key] = record
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
            self.stats_['evictions'] += 1
        self.stats_['peak_cache_entries'] = max(self.stats_['peak_cache_entries'], len(self._cache))
        head, hw, default, dw = record
        if head == default == 'UNSAT':
            prediction, status = -2, 'infeasible'
        elif head == 'SAT' and default == 'UNSAT':
            prediction, status = self.original_.head, 'certain_head'
        elif head == 'UNSAT' and default == 'SAT':
            prediction, status = 1-self.original_.head, 'certain_default'
        else:
            prediction = -1
            status = 'ambiguous' if head == default == 'SAT' else 'unresolved'
        if not details:
            return prediction
        def witness(hidden):
            return None if hidden is None else {str(f):v for f,v in zip(self.features_, hidden)}
        return dict(prediction=prediction, status=status, head_status=head, default_status=default,
                    head_witness=witness(hw), default_witness=witness(dw),
                    retained_values={str(f):str(rational(values[f])) for f in self.retained_features_})

    def predict(self, X, *, retained_only=False):
        X = np.asarray(X, dtype=float)
        expected = len(self.retained_features_) if retained_only else self.original_.n_features
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError('Expected a matrix of the declared width')
        return np.asarray([self._query(row, retained_only=retained_only, details=False) for row in X], dtype=np.int8)

    def export(self):
        return dict(representation='contextual numerical query specification',
                    original=self.original_.export(), forgotten_features=list(self.features_),
                    constraints=[b.export() for b in self.constraints_],
                    max_nodes=self.max_nodes, max_cache=self.max_cache,
                    arithmetic='exact rational difference bounds; strict cycles checked',
                    scope='source surrogate under explicitly supplied background constraints')

    @classmethod
    def from_export(cls, record):
        result = cls(max_nodes=record['max_nodes'], max_cache=record['max_cache']).fit(
            RuleProgram.from_export(record['original']), record['forgotten_features'],
            [DifferenceBound(*b) for b in record['constraints']])
        if result.export() != record:
            raise ValueError('Invalid query specification')
        return result

    def verify_query(self, row, record, *, retained_only=False):
        """Replay SAT/UNSAT proofs, rather than trusting serialized flags."""
        fresh = self.from_export(self.export()).query(row, retained_only=retained_only)
        if fresh != record:
            raise ValueError('Query certificate does not match exact replay')
        return True

    def cache_bytes(self):
        """Canonical JSON payload bytes, explicitly not resident memory."""
        return sum(len(json.dumps([[[[a.feature, a.threshold, a.greater] for a in r]
                                    for r in key[0]], key[1], value]).encode())
                   for key, value in self._cache.items())
