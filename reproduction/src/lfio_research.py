"""Validation-selected Boolean rule sets with an explicit error/length objective.

This is a propositional learner, not a new non-monotonic logic semantics.
Candidate coverage is cached in packed integers; both decision polarities are
considered. Greedy additions and deletions monotonically reduce the stated
training objective. Validation chooses a model without consulting test data.
"""
from dataclasses import dataclass
import hashlib
import itertools
import math
import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score


@dataclass(frozen=True)
class Candidate:
    literals: tuple
    bits: int


@dataclass(frozen=True)
class PreparedCandidates:
    """Training-input-only coverage cache, reusable across condition budgets."""
    fingerprint: str
    configuration: tuple
    candidates: tuple
    effective_depths: tuple
    statistics: tuple


class ValidatedLFIO:
    """Historical compatibility class; current studies expose RefinedLFIO only."""
    def __init__(self, depths=(1, 2, 3, 4, 5), penalties=(0.0, 0.05, 0.2, 0.8),
                 polarities=(0, 1), max_rules=32, min_support=3,
                 candidate_budget=50000, prune=True, literal_budget=None):
        self.depths = tuple(depths)
        self.penalties = tuple(penalties)
        self.polarities = tuple(polarities)
        self.max_rules = max_rules
        self.min_support = min_support
        self.candidate_budget = candidate_budget
        self.prune = prune
        self.literal_budget = literal_budget

    def _candidate_configuration(self):
        return (self.depths, self.min_support, self.candidate_budget)

    @staticmethod
    def _input_fingerprint(X):
        return hashlib.sha256(str(X.shape).encode() + X.tobytes()).hexdigest()

    def prepare_candidates(self, X):
        """Cache exhaustive supported candidates without labels or validation data."""
        X, _ = self._check(X)
        candidates = self._candidates(X)
        return PreparedCandidates(self._input_fingerprint(X), self._candidate_configuration(),
                                  tuple(candidates), tuple(self.effective_depths_),
                                  tuple(self.search_stats_.items()))

    def _count_work(self, name, amount):
        if not hasattr(self, 'search_work_'):
            self.search_work_ = {}
        self.search_work_[name] = self.search_work_.get(name, 0) + amount

    @staticmethod
    def _bits(mask):
        return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')

    @staticmethod
    def _check(X, y=None):
        X = np.asarray(X)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isin(X, [0, 1]).all():
            raise ValueError('X must be a nonempty complete Boolean matrix')
        if y is not None:
            y = np.asarray(y)
            if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
                raise ValueError('y must contain one Boolean label per row')
        return X.astype(np.uint8, copy=False), y

    def _candidates(self, X):
        features = X.shape[1]
        requested = sorted(set(min(int(d), features) for d in self.depths))
        budget_depth = 0
        total = 0
        for depth in range(1, max(requested) + 1):
            needed = math.comb(features, depth) * 2 ** depth
            if total + needed > self.candidate_budget:
                break
            total += needed
            budget_depth = depth
        if not budget_depth:
            raise ValueError('candidate_budget cannot cover all single literals')
        self.effective_depths_ = [d for d in requested if d <= budget_depth]
        if not self.effective_depths_:
            self.effective_depths_ = [budget_depth]
        maximum = max(self.effective_depths_)
        literal_bits = {(f, v): self._bits(X[:, f] == v)
                        for f in range(features) for v in (0, 1)}
        candidates = []
        all_bits = (1 << len(X)) - 1
        evaluated = 0
        for size in range(1, maximum + 1):
            for columns in itertools.combinations(range(features), size):
                for values in itertools.product((0, 1), repeat=size):
                    evaluated += 1
                    bits = all_bits
                    for col, val in zip(columns, values):
                        bits &= literal_bits[col, val]
                    if bits.bit_count() >= self.min_support:
                        candidates.append(Candidate(tuple(zip(columns, values)), bits))
        self.search_stats_ = {'enumerated_candidates': evaluated,
                              'supported_candidates': len(candidates),
                              'effective_depths': self.effective_depths_,
                              'candidate_budget': self.candidate_budget}
        return candidates

    @staticmethod
    def _union(rules):
        bits = 0
        for rule in rules:
            bits |= rule.bits
        return bits

    @staticmethod
    def _objective(rules, positive, negative, wp, wn, penalty):
        covered = ValidatedLFIO._union(rules)
        return (wp * (positive & ~covered).bit_count()
                + wn * (negative & covered).bit_count()
                + penalty * sum(len(r.literals) + 1 for r in rules))

    def _fit_one(self, pool, positive, negative, n_samples, alpha):
        np_, nn = positive.bit_count(), negative.bit_count()
        wp, wn = n_samples / (2 * max(1, np_)), n_samples / (2 * max(1, nn))
        penalty = alpha * math.log2(max(2, n_samples))
        covered, rules = 0, []
        objective_path = [wp * np_]
        for _ in range(self.max_rules):
            self._count_work('greedy_candidate_visits', len(pool))
            best, best_gain = None, 1e-12
            used_literals = sum(len(r.literals) for r in rules)
            remaining_p = positive & ~covered
            remaining_n = negative & ~covered
            for candidate in pool:
                if (self.literal_budget is not None and
                        used_literals + len(candidate.literals) > self.literal_budget):
                    continue
                tp = (candidate.bits & remaining_p).bit_count()
                if not tp:
                    continue
                fp = (candidate.bits & remaining_n).bit_count()
                gain = wp * tp - wn * fp - penalty * (len(candidate.literals) + 1)
                # Enumeration prefers shorter rules, then lexicographic literals.
                if gain > best_gain:
                    best, best_gain = candidate, gain
            if best is None:
                break
            rules.append(best)
            covered |= best.bits
            objective_path.append(self._objective(rules, positive, negative, wp, wn, penalty))
        if self.prune:
            improved = True
            while improved and rules:
                improved = False
                current = self._objective(rules, positive, negative, wp, wn, penalty)
                for index in range(len(rules) - 1, -1, -1):
                    proposed = rules[:index] + rules[index + 1:]
                    loss = self._objective(proposed, positive, negative, wp, wn, penalty)
                    if loss <= current + 1e-10:
                        rules = proposed
                        current = loss
                        objective_path.append(loss)
                        improved = True
        return rules, objective_path

    def _fit_variants(self, pool, positive, negative, n_samples, alpha):
        rules, history = self._fit_one(pool, positive, negative, n_samples, alpha)
        yield 'greedy', rules, history, {}

    @staticmethod
    def _predict_rules(X, rules, head):
        output = np.full(len(X), 1 - head, dtype=np.uint8)
        for rule in rules:
            matches = np.ones(len(X), dtype=bool)
            for feature, value in rule.literals:
                matches &= X[:, feature] == value
            output[matches] = head
        return output

    def fit(self, X, y, X_val, y_val, feature_names=None, *, prepared=None):
        started = time.perf_counter()
        X, y = self._check(X, y)
        X_val, y_val = self._check(X_val, y_val)
        if X_val.shape[1] != X.shape[1]:
            raise ValueError('Validation feature count differs from training')
        if not self.depths or min(self.depths) < 1 or not self.penalties or min(self.penalties) < 0:
            raise ValueError('Provide positive depths and nonnegative penalties')
        if not self.polarities or not set(self.polarities) <= {0, 1}:
            raise ValueError('polarities must be chosen from 0 and 1')
        if self.max_rules < 1 or self.min_support < 1:
            raise ValueError('max_rules and min_support must be positive')
        if self.literal_budget is not None and (not isinstance(self.literal_budget, (int, np.integer))
                                                or self.literal_budget < 0):
            raise ValueError('literal_budget must be a nonnegative integer or None')
        self.n_features_in_ = X.shape[1]
        self.feature_names = list(feature_names) if feature_names is not None else [f'x{i}' for i in range(X.shape[1])]
        if len(self.feature_names) != X.shape[1]:
            raise ValueError('Feature-name count does not match X')
        self.selection_trace_ = []
        self.optimization_ = {}
        self.search_work_ = {}
        if len(np.unique(y)) == 1:
            self.head_ = 1 - int(y[0])
            self._selected = []
            self.rules = []
            self.selected_config_ = {'constant_prediction': int(y[0])}
            self.objective_path_ = [0.0]
            self.search_stats_ = {'enumerated_candidates': 0, 'supported_candidates': 0}
            self.fit_seconds_ = time.perf_counter() - started
            return self
        if prepared is None:
            candidates = self._candidates(X)
        else:
            if (not isinstance(prepared, PreparedCandidates)
                    or prepared.fingerprint != self._input_fingerprint(X)
                    or prepared.configuration != self._candidate_configuration()):
                raise ValueError('Prepared candidate cache does not match fitting inputs/configuration')
            candidates = prepared.candidates
            self.effective_depths_ = list(prepared.effective_depths)
            self.search_stats_ = dict(prepared.statistics)
        bits = {cls: self._bits(y == cls) for cls in (0, 1)}
        best_key, best_model = None, None
        for depth in self.effective_depths_:
            pool = [r for r in candidates if len(r.literals) <= depth]
            for alpha in self.penalties:
                for head in self.polarities:
                    for strategy, rules, history, audit in self._fit_variants(
                            pool, bits[head], bits[1 - head], len(X), alpha):
                        prediction = self._predict_rules(X_val, rules, head)
                        if len(np.unique(y_val)) == 2:
                            score = float(balanced_accuracy_score(y_val, prediction))
                        else:
                            score = float(np.mean(prediction == y_val))
                        literals = sum(len(r.literals) for r in rules)
                        config = {'depth': depth, 'alpha': alpha, 'head': head,
                                  'validation_balanced_accuracy': score,
                                  'rules': len(rules), 'total_literals': literals,
                                  'strategy': strategy, 'literal_budget': self.literal_budget}
                        self.selection_trace_.append(dict(config, optimization=audit))
                        # Complexity breaks validation ties; test labels are never used.
                        key = (score, -literals, -len(rules), -depth, alpha, -head)
                        if best_key is None or key > best_key:
                            best_key, best_model = key, (rules, head, config, history, audit)
        (self._selected, self.head_, self.selected_config_, self.objective_path_,
         self.optimization_) = best_model
        self.rules = [{'literals': [(int(f), 1 if v == 0 else int(v), v == 0)
                                   for f, v in rule.literals], 'head': self.head_}
                      for rule in self._selected]
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def predict(self, X):
        if not hasattr(self, 'n_features_in_'):
            raise ValueError('fit must be called first')
        X, _ = self._check(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from training')
        return self._predict_rules(X, self._selected, self.head_)

    def export(self):
        return {'default_class': 1 - self.head_, 'rules': self.rules,
                'selected_config': self.selected_config_, 'search': self.search_stats_,
                'objective_path': self.objective_path_, 'selection_trace': self.selection_trace_,
                'literal_budget': self.literal_budget, 'optimization': self.optimization_,
                'search_work': self.search_work_}


class RefinedLFIO(ValidatedLFIO):
    """Two training-only starts with bounded add/delete/single-rule exchange.

    The pure start avoids irreversible false-positive coverage from early broad
    rules. Exchange can retract that coverage. Both initial programs and their
    repaired versions are validation candidates: better training loss need not
    generalize better. No global-optimality or human-interpretability claim is
    made. The historical base class remains available for old result readers.
    """

    def __init__(self, *, pure_start=True, max_refinement_steps=8, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(max_refinement_steps, int) or max_refinement_steps < 0:
            raise ValueError('max_refinement_steps must be a nonnegative integer')
        self.pure_start = pure_start
        self.max_refinement_steps = max_refinement_steps

    def _refine(self, rules, pool, positive, negative, n_samples, alpha):
        rules = list(rules)
        wp = n_samples / (2 * max(1, positive.bit_count()))
        wn = n_samples / (2 * max(1, negative.bit_count()))
        penalty = alpha * math.log2(max(2, n_samples))
        history = [self._objective(rules, positive, negative, wp, wn, penalty)]
        moves, stopped = [], 'step_limit'
        for _ in range(self.max_refinement_steps):
            size = sum(len(r.literals) for r in rules)
            current = history[-1]
            selected_literals = {r.literals for r in rules}
            best_key = (current, size, len(rules))
            best = None
            # Prefix/suffix unions make removing each rule linear in rule count.
            prefix, suffix = [0], [0] * (len(rules) + 1)
            for rule in rules:
                prefix.append(prefix[-1] | rule.bits)
            for index in range(len(rules) - 1, -1, -1):
                suffix[index] = suffix[index + 1] | rules[index].bits
            for removed in [None, *range(len(rules))]:
                self._count_work('repair_candidate_visits', len(pool) + int(removed is not None))
                covered = prefix[-1] if removed is None else prefix[removed] | suffix[removed + 1]
                kept_size = size - (len(rules[removed].literals) if removed is not None else 0)
                kept_count = len(rules) - int(removed is not None)
                remaining_p, remaining_n = positive & ~covered, negative & ~covered
                base_loss = (wp * remaining_p.bit_count() + wn * (negative & covered).bit_count()
                             + penalty * (kept_size + kept_count))
                candidates = itertools.chain([None] if removed is not None else [], pool)
                for candidate in candidates:
                    new_size = kept_size + (len(candidate.literals) if candidate else 0)
                    new_count = kept_count + int(candidate is not None)
                    if new_count > self.max_rules or (self.literal_budget is not None
                                                      and new_size > self.literal_budget):
                        continue
                    loss = base_loss
                    if candidate:
                        if candidate.literals in selected_literals and (removed is None or candidate != rules[removed]):
                            continue
                        loss += (-wp * (candidate.bits & remaining_p).bit_count()
                                 + wn * (candidate.bits & remaining_n).bit_count()
                                 + penalty * (len(candidate.literals) + 1))
                    # Equal-loss moves are allowed only when the program shrinks.
                    better = loss < best_key[0] - 1e-10
                    simpler = abs(loss - best_key[0]) <= 1e-10 and (new_size, new_count) < best_key[1:]
                    if better or simpler:
                        best_key, best = (loss, new_size, new_count), (removed, candidate)
            if best is None:
                stopped = 'local_optimum'
                break
            removed, candidate = best
            if removed is not None:
                rules.pop(removed)
            if candidate is not None:
                rules.append(candidate)
            actual = self._objective(rules, positive, negative, wp, wn, penalty)
            if actual > current + 1e-8:
                raise AssertionError('Rule refinement increased the training objective')
            history.append(actual)
            moves.append('add' if removed is None else 'delete' if candidate is None else 'replace')
        return rules, history, {'moves': moves, 'stop_reason': stopped,
                                'max_refinement_steps': self.max_refinement_steps,
                                'neighborhood': 'add/delete/one-for-one replacement in enumerated pool'}

    def _fit_variants(self, pool, positive, negative, n_samples, alpha):
        starts = [('greedy', pool)]
        if self.pure_start:
            starts.append(('pure', [c for c in pool if not c.bits & negative]))
        seen = set()
        for label, initial_pool in starts:
            rules, history = self._fit_one(initial_pool, positive, negative, n_samples, alpha)
            signature = tuple(sorted(r.literals for r in rules))
            if signature in seen:
                continue
            seen.add(signature)
            yield label, rules, history, {'stage': 'initial', 'initialization': label}
            if self.max_refinement_steps:
                repaired, tail, audit = self._refine(rules, pool, positive, negative, n_samples, alpha)
                audit['initialization'] = label
                if audit['moves']:
                    yield label + '+repair', repaired, history + tail[1:], audit
