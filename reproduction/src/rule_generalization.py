"""Explicit generalization of single-head numerical-threshold DNF surrogates.

Exact compression proves equivalence over real-valued inputs, not just observed
rows. Approximate generalization builds a monotone expansion path on grow data;
an independent calibration set can select a prefix using simultaneous one-sided
Clopper--Pearson bounds. Calibration independence and iid class-conditional
sampling are assumptions, not properties this module can infer from arrays.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True, order=True)
class Atom:
    feature: int
    threshold: float
    greater: bool = True


@dataclass(frozen=True)
class RuleProgram:
    rules: tuple
    head: int
    n_features: int

    def __post_init__(self):
        if self.head not in (0, 1) or self.n_features < 1:
            raise ValueError('Expected a binary head and positive feature count')
        for rule in self.rules:
            for atom in rule:
                if (not isinstance(atom, Atom) or not isinstance(atom.feature, int)
                        or not 0 <= atom.feature < self.n_features
                        or not math.isfinite(atom.threshold)
                        or not isinstance(atom.greater, bool)):
                    raise ValueError('Invalid finite threshold atom')

    @classmethod
    def from_export(cls, record):
        if record.get('representation') != 'single-head numerical DNF':
            raise ValueError('Unsupported representation')
        if record.get('input_cast') != 'float64':
            raise ValueError('Only finite float64 input semantics are supported')
        if record.get('default_class') != 1-record['head']:
            raise ValueError('Default must complement the common head')
        return cls(tuple(tuple(Atom(int(f), float(t), bool(g)) for f, t, g in rule)
                         for rule in record['rules']), int(record['head']), int(record['n_features']))

    @classmethod
    def from_lfio(cls, record, n_features):
        if record.get('default_class') != 1-record['head']:
            raise ValueError('Expected a common head and opposite default')
        vocabulary = record['vocabulary']
        rules = tuple(tuple(Atom(int(vocabulary[f][0]), float(vocabulary[f][1]), bool(v))
                            for f, v in rule) for rule in record['rules'])
        return cls(rules, int(record['head']), int(n_features))

    def export(self):
        return dict(representation='single-head numerical DNF', input_cast='float64',
                    n_features=self.n_features, head=self.head, default_class=1-self.head,
                    rules=[[[a.feature, a.threshold, a.greater] for a in r] for r in self.rules])

    def predict(self, X):
        X = _matrix(X, self.n_features)
        covered = np.zeros(len(X), dtype=bool)
        for rule in self.rules:
            hit = np.ones(len(X), dtype=bool)
            for atom in rule:
                value = X[:, atom.feature] > atom.threshold
                hit &= value if atom.greater else ~value
            covered |= hit
        return np.where(covered, self.head, 1-self.head).astype(np.uint8)

    @property
    def complexity(self):
        atoms = [a for rule in self.rules for a in rule]
        return (len(atoms), len(self.rules), len({a.feature for a in atoms}),
                len({(a.feature, a.threshold) for a in atoms}))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.export(), sort_keys=True).encode()).hexdigest()


def _matrix(X, features):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != features or not np.isfinite(X).all():
        raise ValueError('Expected a finite numerical matrix of the specified width')
    return X


def _labels(y, size):
    y = np.asarray(y)
    if y.shape != (size,) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected one binary teacher output per row')
    return y.astype(np.uint8)


def _box(rule, features):
    # Every coordinate denotes (lower, upper]. Strictness is never rounded.
    bounds = [[-math.inf, math.inf] for _ in range(features)]
    for atom in rule:
        if atom.greater:
            bounds[atom.feature][0] = max(bounds[atom.feature][0], atom.threshold)
        else:
            bounds[atom.feature][1] = min(bounds[atom.feature][1], atom.threshold)
    return tuple(map(tuple, bounds)) if all(lo < hi for lo, hi in bounds) else None


def _body(box):
    return tuple(sorted([Atom(j, lo, True) for j, (lo, _) in enumerate(box) if math.isfinite(lo)]
                        + [Atom(j, hi, False) for j, (_, hi) in enumerate(box) if math.isfinite(hi)]))


def _program(rules, parent):
    return RuleProgram(tuple(sorted(set(map(tuple, rules)))), parent.head, parent.n_features)


def normalize(program):
    """Remove contradictory clauses, duplicates and implied same-feature atoms."""
    return _program([_body(box) for r in program.rules
                     if (box := _box(r, program.n_features)) is not None], program)


class _ProofLimit(Exception):
    pass


class _Work:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def tick(self):
        if self.used >= self.limit:
            raise _ProofLimit
        self.used += 1


def _subtract(box, cover, work):
    work.tick()
    intersection = tuple((max(a, c), min(b, d)) for (a, b), (c, d) in zip(box, cover))
    if any(lo >= hi for lo, hi in intersection):
        return [box]
    residual, core = [], list(box)
    for j, (lo, hi) in enumerate(intersection):
        left, right = core[j]
        if left < lo:
            piece = list(core); piece[j] = (left, lo)
            residual.append(tuple(piece)); core[j] = (lo, right)
        left, right = core[j]
        if hi < right:
            piece = list(core); piece[j] = (hi, right)
            residual.append(tuple(piece)); core[j] = (left, hi)
    return residual


def _difference(left, right, work):
    covers = [box for rule in right.rules if (box := _box(rule, right.n_features)) is not None]
    for rule in left.rules:
        target = _box(rule, left.n_features)
        if target is None:
            continue
        residual = [target]
        for cover in covers:
            pieces = []
            for box in residual:
                pieces.extend(_subtract(box, cover, work))
            residual = pieces
            if not residual:
                break
        if residual:
            return residual[0]
    return None


def verify_relation(before, after, *, equivalent=True, max_nodes=20000):
    """Prove equality or head-region containment, with an explicit unknown state.

    Containment means before's head region is a subset of after's, for either
    head polarity. An uncovered box is a real-domain counterexample region.
    Reaching the work cap is never reported as a successful proof.
    """
    if before.head != after.head or before.n_features != after.n_features:
        raise ValueError('Programs must share the head and feature domain')
    if not isinstance(max_nodes, int) or max_nodes < 0:
        raise ValueError('max_nodes must be a nonnegative integer')
    work = _Work(max_nodes)
    try:
        difference = _difference(before, after, work)
        direction = 'before_not_after'
        if difference is None and equivalent:
            difference = _difference(after, before, work)
            direction = 'after_not_before'
        return dict(status='proved' if difference is None else 'disproved',
                    relation='equivalence' if equivalent else 'head-region containment',
                    nodes=work.used, counterexample_box=(None if difference is None else
                        [[None if not math.isfinite(v) else v for v in pair] for pair in difference]),
                    direction=None if difference is None else direction,
                    scope='all real-valued inputs under the exported strict-threshold semantics')
    except _ProofLimit:
        return dict(status='unknown', relation='equivalence' if equivalent else 'head-region containment',
                    nodes=work.used, reason='proof node limit')


def _hull(first, second, features):
    a, b = _box(first, features), _box(second, features)
    return _body(tuple((min(x, z), max(y, w)) for (x, y), (z, w) in zip(a, b)))


def _neighbors(program, *, include_deletion=False):
    seen = set()
    for i, rule in enumerate(program.rules):
        if include_deletion:
            proposal = _program(program.rules[:i]+program.rules[i+1:], program)
            if proposal.fingerprint not in seen:
                seen.add(proposal.fingerprint)
                yield 'remove_covered_rule', proposal
        for j in range(len(rule)):
            rules = list(program.rules); rules[i] = rule[:j]+rule[j+1:]
            proposal = normalize(_program(rules, program))
            if proposal.fingerprint not in seen:
                seen.add(proposal.fingerprint)
                yield 'drop_condition', proposal
    for i, j in itertools.combinations(range(len(program.rules)), 2):
        hull = _hull(program.rules[i], program.rules[j], program.n_features)
        proposal = normalize(_program([r for k, r in enumerate(program.rules) if k not in (i, j)]+[hull], program))
        if proposal.fingerprint not in seen:
            seen.add(proposal.fingerprint)
            yield 'merge_interval_hull', proposal


def boolean_simplify(program):
    """Classical absorption and complementary-pair merging; no threshold order."""
    current = _program([tuple(sorted(set(r))) for r in program.rules], program)
    while True:
        proposals = []
        for i, j in itertools.combinations(range(len(current.rules)), 2):
            first, second = set(current.rules[i]), set(current.rules[j])
            replacement = None
            if first <= second or second <= first:
                replacement = tuple(sorted(first if first <= second else second))
            elif len(first-second) == len(second-first) == 1:
                a, b = next(iter(first-second)), next(iter(second-first))
                if a.feature == b.feature and a.threshold == b.threshold and a.greater != b.greater:
                    replacement = tuple(sorted(first & second))
            if replacement is not None:
                proposal = _program([r for k, r in enumerate(current.rules) if k not in (i, j)]+[replacement], current)
                if proposal.complexity < current.complexity:
                    proposals.append(proposal)
        if not proposals:
            return current
        current = min(proposals, key=lambda p: (p.complexity, p.fingerprint))


def exact_compress(program, *, max_nodes=20000, total_nodes=500000):
    """Greedy, globally verified compression; not a globally minimum DNF."""
    current = normalize(program)
    trace = [dict(operation='normalize_intervals', before=program.export(), after=current.export(),
                  proof='coordinatewise interval identity')]
    used, checked, unknown = 0, 0, 0
    while used < total_nodes:
        proposals = sorted(_neighbors(current, include_deletion=True),
                           key=lambda item: (item[1].complexity, item[1].fingerprint, item[0]))
        accepted = False
        for operation, candidate in proposals:
            if candidate.complexity >= current.complexity:
                continue
            proof = verify_relation(current, candidate, max_nodes=min(max_nodes, total_nodes-used))
            used += proof['nodes']; checked += 1
            unknown += int(proof['status'] == 'unknown')
            if proof['status'] == 'proved':
                trace.append(dict(operation=operation, before=current.export(), after=candidate.export(), proof=proof))
                current, accepted = candidate, True
                break
            if used >= total_nodes:
                break
        if not accepted:
            break
    return current, dict(trace=trace, exact_equivalence=True, nodes=used,
                         proposals_checked=checked, unknown_proofs=unknown,
                         exhausted=used >= total_nodes, optimal_size_claim=False)


def _harm(original, candidate, y, head):
    risk = y == 1-head
    count = int(risk.sum())
    harmful = int(((original == y) & (candidate != y) & risk).sum())
    return harmful, count


def _upper(successes, trials, delta):
    if not trials or successes == trials:
        return 1.
    return float(beta.ppf(1-delta, successes+1, trials-successes))


class RuleGeneralizer:
    """Fit an explicit generalization path, then optionally calibrate it.

    epsilon bounds *two-class balanced-error increase* relative to the original
    surrogate. Monotone head expansion can harm only the opposite teacher class;
    its conditional harm bound is divided by two. This does not bound teacher
    accuracy or total surrogate error, and gives no guarantee under distribution
    shift. confidence='empirical' is an observed calibration constraint only.
    """
    def __init__(self, *, mode='exact', epsilon=.02, delta=.05,
                 confidence='simultaneous', max_nodes=20000, total_nodes=500000):
        if mode not in ('exact', 'train', 'calibrated'):
            raise ValueError('mode must be exact, train or calibrated')
        if confidence not in ('simultaneous', 'empirical'):
            raise ValueError('Unknown calibration confidence mode')
        if not 0 <= epsilon < .5 or not 0 < delta < 1 or max_nodes < 1 or total_nodes < 1:
            raise ValueError('Invalid risk or proof allowance')
        self.mode, self.epsilon, self.delta = mode, float(epsilon), float(delta)
        self.confidence, self.max_nodes, self.total_nodes = confidence, int(max_nodes), int(total_nodes)

    def fit(self, program, X_grow=None, y_grow=None):
        if not isinstance(program, RuleProgram):
            raise ValueError('Expected a RuleProgram')
        self.original_ = program
        self.exact_, exact_audit = exact_compress(program, max_nodes=self.max_nodes, total_nodes=self.total_nodes)
        self.path_ = [self.exact_]
        self.audit_ = dict(exact=exact_audit, mode=self.mode, epsilon=self.epsilon, delta=self.delta,
                           confidence=self.confidence, path=[], calibration_used=False,
                           guarantee='global equivalence to original surrogate', original_sha256=program.fingerprint)
        if self.mode != 'exact':
            X = _matrix(X_grow, program.n_features); y = _labels(y_grow, len(X))
            base_prediction = program.predict(X)
            current = self.exact_
            while current.complexity[0] > 0:
                feasible = []
                for operation, candidate in _neighbors(current):
                    if candidate.complexity >= current.complexity:
                        continue
                    harmful, count = _harm(base_prediction, candidate.predict(X), y, program.head)
                    # Budget is cumulative against the original, never per move.
                    if count and harmful/count <= 2*self.epsilon+1e-12:
                        feasible.append((candidate.complexity, harmful/count, candidate.fingerprint, operation, candidate))
                if not feasible:
                    break
                _, _, _, operation, candidate = min(feasible)
                containment = verify_relation(current, candidate, equivalent=False, max_nodes=self.max_nodes)
                if containment['status'] != 'proved':
                    # An unproved expansion never enters the calibration path.
                    break
                compressed, proof = exact_compress(candidate, max_nodes=self.max_nodes, total_nodes=self.total_nodes)
                self.audit_['path'].append(dict(operation=operation, before=current.export(),
                    expanded=candidate.export(), after=compressed.export(), containment=containment, exact=proof))
                self.path_.append(compressed); current = compressed
        self.program_ = self.path_[-1] if self.mode == 'train' else self.exact_
        if self.mode == 'train':
            self.audit_['guarantee'] = 'grow-sample cumulative harm constraint only'
        self.audit_['selected_index'] = len(self.path_)-1 if self.mode == 'train' else 0
        self.audit_['path_sha256'] = [p.fingerprint for p in self.path_]
        return self

    def calibrate(self, X, teacher_outputs):
        """Select from a grow-only path; never modify it using calibration data.

        For the simultaneous bound, the original learner, all path construction
        and hyperparameter choices must be independent of these calibration
        observations. Calling this method repeatedly does not permit adaptively
        rebuilding a path on the same calibration set.
        """
        if self.mode != 'calibrated' or not hasattr(self, 'path_'):
            raise ValueError('Fit a calibrated generalizer before calibration')
        if self.audit_['calibration_used']:
            raise ValueError('Calibration is one-shot; fit a fresh instance for another study')
        X = _matrix(X, self.original_.n_features); y = _labels(teacher_outputs, len(X))
        before = self.original_.predict(X)
        family_size = max(1, len(self.path_)-1)
        rows, feasible = [], [0]
        for index, candidate in enumerate(self.path_[1:], 1):
            harmful, count = _harm(before, candidate.predict(X), y, self.original_.head)
            upper = _upper(harmful, count, self.delta/family_size)
            bound = upper/2 if self.confidence == 'simultaneous' else (harmful/(2*count) if count else .5)
            accept = bool(count and bound <= self.epsilon+1e-12)
            rows.append(dict(index=index, harmful=harmful, opposite_class_samples=count,
                harm_upper=upper, balanced_error_increase_bound=bound,
                accepted=accept, comparison_error_probability=self.delta/family_size))
            if accept:
                feasible.append(index)
        selected = min(feasible, key=lambda i: (self.path_[i].complexity, i))
        self.program_ = self.path_[selected]
        self.audit_.update(calibration_used=True, selected_index=selected, calibration=rows,
            family_size=family_size, guarantee=('simultaneous class-conditional binomial harm bound; independent iid calibration required'
                if self.confidence == 'simultaneous' else 'calibration-sample cumulative harm constraint only'))
        return self

    def predict(self, X):
        if not hasattr(self, 'program_'):
            raise ValueError('fit must be called first')
        return self.program_.predict(X)

    def export(self):
        if not hasattr(self, 'program_'):
            raise ValueError('fit must be called first')
        return dict(program=self.program_.export(), audit=self.audit_,
                    path=[p.export() for p in self.path_])
