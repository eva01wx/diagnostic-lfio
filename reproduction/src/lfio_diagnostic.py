"""Prospective diagnostic allocation with absolute neighbor-scoring limits.

The search primitive adapts the historical ScoredWorkLFIO implementation.
The controller allocates fitting work using an internal holdout; it never
receives test inputs, labels, or teacher predictions. Empirical collision
bounds and held-out action witnesses have deliberately different meanings.
"""
import hashlib
import heapq
import math
import time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .lfio_research import RefinedLFIO
from .lfio_budgeted import mdl_vocabulary
from .lfio_followup import apply_vocabulary

class AbsoluteWorkLFIO(RefinedLFIO):
    """One training winner per cell under a shared scored-neighbor allowance.

Both policies use ordinary and pure initializations, then randomized greedy
restarts (uniform choice among the three best improving additions). The repair
policy spends part of the same allowance on add/delete/replace moves. A score
unit is a feasible program-neighbor objective evaluation, including deletions.
It is not a CPU-time unit. Exhaustion retains the best evaluated feasible move.
"""

    def __init__(self, *, cell_score_budget=200000, allocate_repair=True, random_state=0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(cell_score_budget, int) or cell_score_budget < 1:
            raise ValueError("cell_score_budget must be a positive integer")
        self.cell_score_budget = cell_score_budget
        self.allocate_repair = allocate_repair
        self.random_state = int(random_state)

    def _fit_variants(self, pool, positive, negative, n_samples, alpha):
        wp = n_samples / (2 * max(1, positive.bit_count()))
        wn = n_samples / (2 * max(1, negative.bit_count()))
        penalty = alpha * math.log2(max(2, n_samples))
        limit = self.cell_score_budget
        used, restarts, repair_moves = 0, 0, 0
        baseline = wp * positive.bit_count()
        best_rules, best_history, best_loss, best_size = [], [baseline], baseline, (0, 0)
        best_origin = "constant"
        seed_bytes = f"{self.random_state}|{positive}|{alpha}|{len(pool)}|{self.literal_budget}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
        rng = np.random.default_rng(seed)

        def objective(rules):
            return self._objective(rules, positive, negative, wp, wn, penalty)

        def remember(rules, history, origin):
            nonlocal best_rules, best_history, best_loss, best_size, best_origin
            loss = history[-1]
            size = (sum(len(r.literals) for r in rules), len(rules))
            if loss < best_loss - 1e-10 or (abs(loss-best_loss) <= 1e-10 and size < best_size):
                best_rules, best_history = list(rules), list(history)
                best_loss, best_size, best_origin = loss, size, origin

        def greedy(initial_pool, randomize, origin):
            nonlocal used
            rules, history, covered, size = [], [baseline], 0, 0
            for _ in range(self.max_rules):
                proposals = []
                remaining_p, remaining_n = positive & ~covered, negative & ~covered
                for index, candidate in enumerate(initial_pool):
                    if used >= limit:
                        break
                    if self.literal_budget is not None and size + len(candidate.literals) > self.literal_budget:
                        continue
                    used += 1
                    gain = (wp * (candidate.bits & remaining_p).bit_count()
                            - wn * (candidate.bits & remaining_n).bit_count()
                            - penalty * (len(candidate.literals)+1))
                    if gain > 1e-12:
                        item = (gain, -index, candidate)
                        width = 3 if randomize else 1
                        if len(proposals) < width:
                            heapq.heappush(proposals, item)
                        elif item[:2] > proposals[0][:2]:
                            heapq.heapreplace(proposals, item)
                if not proposals:
                    break
                proposals.sort(key=lambda item: (-item[0], -item[1]))
                choice = int(rng.integers(min(3, len(proposals)))) if randomize else 0
                candidate = proposals[choice][2]
                rules.append(candidate)
                covered |= candidate.bits
                size += len(candidate.literals)
                history.append(objective(rules))
                remember(rules, history, origin)
                if used >= limit:
                    break
            if self.prune:
                changed = True
                while changed and rules and used < limit:
                    changed = False
                    for index in range(len(rules)-1, -1, -1):
                        if used >= limit:
                            break
                        proposed = rules[:index] + rules[index+1:]
                        used += 1
                        loss = objective(proposed)
                        if loss <= history[-1] + 1e-10:
                            rules = proposed
                            history.append(loss)
                            remember(rules, history, origin)
                            changed = True
            return rules, history

        def repair(rules, history, origin):
            nonlocal used, repair_moves
            rules = list(rules)
            for _ in range(self.max_refinement_steps):
                if used >= limit:
                    break
                size = sum(len(r.literals) for r in rules)
                key, move = (history[-1], size, len(rules)), None
                selected = {r.literals for r in rules}
                prefix, suffix = [0], [0]*(len(rules)+1)
                for rule in rules:
                    prefix.append(prefix[-1] | rule.bits)
                for index in range(len(rules)-1, -1, -1):
                    suffix[index] = suffix[index+1] | rules[index].bits
                for removed in [None, *range(len(rules))]:
                    covered = prefix[-1] if removed is None else prefix[removed] | suffix[removed+1]
                    kept_size = size - (len(rules[removed].literals) if removed is not None else 0)
                    kept_count = len(rules) - int(removed is not None)
                    remaining_p, remaining_n = positive & ~covered, negative & ~covered
                    base = (wp*remaining_p.bit_count() + wn*(negative & covered).bit_count()
                            + penalty*(kept_size+kept_count))
                    neighbors = ([None] if removed is not None else []) + pool
                    for candidate in neighbors:
                        if used >= limit:
                            break
                        new_size = kept_size + (len(candidate.literals) if candidate else 0)
                        new_count = kept_count + int(candidate is not None)
                        if new_count > self.max_rules or (self.literal_budget is not None and new_size > self.literal_budget):
                            continue
                        if candidate and candidate.literals in selected:
                            # Replacing a rule by itself is not a scored neighbor.
                            continue
                        used += 1
                        loss = base
                        if candidate:
                            loss += (-wp*(candidate.bits & remaining_p).bit_count()
                                     + wn*(candidate.bits & remaining_n).bit_count()
                                     + penalty*(len(candidate.literals)+1))
                        if (loss < key[0]-1e-10 or
                                (abs(loss-key[0]) <= 1e-10 and (new_size, new_count) < key[1:])):
                            key, move = (loss, new_size, new_count), (removed, candidate)
                    if used >= limit:
                        break
                if move is None:
                    break
                removed, candidate = move
                if removed is not None:
                    rules.pop(removed)
                if candidate is not None:
                    rules.append(candidate)
                actual = objective(rules)
                if actual > history[-1] + 1e-8:
                    raise AssertionError("Repair increased the objective")
                history.append(actual)
                repair_moves += 1
                remember(rules, history, origin+"+repair")

        pure_pool = [c for c in pool if not c.bits & negative]
        # If no rule fits the condition ceiling, scoring cannot consume work.
        usable = any(self.literal_budget is None or len(c.literals) <= self.literal_budget for c in pool)
        while used < limit and usable:
            start_used = used
            is_pure = restarts == 1 and self.pure_start
            initial_pool = pure_pool if is_pure else pool
            origin = "pure" if is_pure else "greedy" if restarts == 0 else "randomized_greedy"
            rules, history = greedy(initial_pool, restarts >= 2, origin)
            if self.allocate_repair:
                repair(rules, history, origin)
            restarts += 1
            if used == start_used and not is_pure:
                break
        self._count_work("scored_neighbor_evaluations", used)
        audit = dict(policy="repair allocation" if self.allocate_repair else "greedy restarts",
                     score_budget=limit, scores_used=used, cell_score_budget=self.cell_score_budget,
                     initializations=restarts, repair_moves_evaluated_path=repair_moves,
                     selected_origin=best_origin, training_objective=best_loss,
                     validation_policy="one training-objective winner per cell",
                     random_state=self.random_state, seed_digest=seed)
        yield "bounded_repair" if self.allocate_repair else "bounded_restarts", best_rules, best_history, audit


def balanced_collision(X, y):
    """Exact *sample* balanced error floor for an unrestricted partition decoder."""
    X, y = RefinedLFIO._check(X, y)
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    ones = np.bincount(inverse, weights=y, minlength=len(counts))
    zeros = counts-ones
    n0, n1 = max(1, int((y == 0).sum())), max(1, int((y == 1).sum()))
    error = np.minimum(zeros/(2*n0), ones/(2*n1)).sum()
    return dict(error_floor=float(error), patterns=len(counts),
                repeated_sample_fraction=float(counts[counts > 1].sum()/len(y)),
                singleton_sample_fraction=float((counts == 1).sum()/len(y)),
                scope="empirical; unrestricted decoder; no population or condition-budget guarantee")


def paired_gain(y, before, after):
    """Balanced paired improvement and a descriptive one-standard-error score.

    This is an allocation heuristic, not a confidence interval: its models were
    selected on the same internal holdout. External test data assess the policy.
    """
    y, before, after = map(np.asarray, (y, before, after))
    means, variances = [], []
    for label in np.unique(y):
        change = ((after[y == label] == label).astype(float)
                  - (before[y == label] == label).astype(float))
        means.append(float(change.mean()))
        variances.append(float(change.var(ddof=1)/len(change)) if len(change) > 1 else 0.)
    gain = float(np.mean(means))
    se = float(np.sqrt(sum(variances))/len(means))
    return dict(gain=gain, standard_error=se, allocation_score=gain-se)


class DiagnosticLFIO:
    """Choose stop, search, or vocabulary enrichment before outer validation.

    Four internal probes share 1/3 of an absolute scoring allowance: a small
    median-vocabulary fit, additional median search, an MDL-vocabulary fit, and
    a doubled-condition-budget capacity witness. The last probe is diagnostic
    only and can never become an inadmissible deployed model. Remaining work
    fits exactly one selected language at the original condition budget.
    """
    def __init__(self, literal_budget=8, total_score_budget=1200000,
                 random_state=0, minimum_gain=.01, force_action=None):
        if literal_budget < 0 or total_score_budget < 120:
            raise ValueError("Invalid condition or scoring budget")
        if force_action not in (None, 'stop', 'search', 'vocabulary'):
            raise ValueError("Unknown forced action")
        self.literal_budget = int(literal_budget)
        self.total_score_budget = int(total_score_budget)
        self.random_state = int(random_state)
        self.minimum_gain = float(minimum_gain)
        self.force_action = force_action

    @staticmethod
    def median_vocabulary(X):
        return [(j, float(t)) for j, t in enumerate(np.median(X, axis=0))]

    def _learner(self, budget, scores, *, seed_offset=0, repair=False):
        # Six outer validation candidates, independent of the chosen action.
        return AbsoluteWorkLFIO(depths=(2,), penalties=(0., .05, .2),
            polarities=(0, 1), literal_budget=budget, candidate_budget=75000,
            pure_start=False, allocate_repair=repair, max_refinement_steps=8,
            random_state=self.random_state+seed_offset,
            cell_score_budget=max(1, scores//6))

    def fit(self, X, y, X_val, y_val):
        started = time.perf_counter()
        X, X_val = np.asarray(X, float), np.asarray(X_val, float)
        y, y_val = np.asarray(y, int), np.asarray(y_val, int)
        if (X.ndim != 2 or X_val.ndim != 2 or X.shape[1] != X_val.shape[1]
                or y.shape != (len(X),) or y_val.shape != (len(X_val),)
                or not np.isfinite(X).all() or not np.isfinite(X_val).all()
                or not np.isin(y, [0, 1]).all() or not np.isin(y_val, [0, 1]).all()):
            raise ValueError("Expected finite numeric matrices and Boolean teacher labels")
        self.audit_ = dict(total_score_budget=self.total_score_budget,
            selection_uses="fitting-only internal holdout; outer validation selects six final cells",
            minimum_gain=self.minimum_gain, probes={}, scores_used=0)
        # Group identical raw rows so an internal witness cannot exploit duplicates.
        _, groups = np.unique(X, axis=0, return_inverse=True)
        if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 4 and len(np.unique(groups)) >= 4:
            a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=self.random_state)
                        .split(X, y, groups))
            median = self.median_vocabulary(X[a])
            rich, _ = mdl_vocabulary(X[a], y[a])
            base_cap = self.total_score_budget//24
            probe_specs = [('stop', median, self.literal_budget, base_cap),
                           ('search', median, self.literal_budget, 3*base_cap),
                           ('vocabulary', rich, self.literal_budget, base_cap),
                           ('capacity', median, 2*self.literal_budget, base_cap)]
            predictions = {}
            self.audit_['inner_fit_indices'] = a.tolist()
            self.audit_['inner_holdout_indices'] = b.tolist()
            for name, vocab, budget, cap in probe_specs:
                model = self._learner(budget, cap).fit(
                    apply_vocabulary(X[a], vocab), y[a],
                    apply_vocabulary(X[b], vocab), y[b])
                predictions[name] = model.predict(apply_vocabulary(X[b], vocab))
                used = model.search_work_.get('scored_neighbor_evaluations', 0)
                self.audit_['scores_used'] += used
                self.audit_['probes'][name] = dict(score_cap=cap, scores_used=used,
                    budget=budget, vocabulary=vocab, selected_config=model.selected_config_,
                    total_literals=sum(len(r.literals) for r in model._selected))
            for name in ('search', 'vocabulary', 'capacity'):
                self.audit_['probes'][name].update(paired_gain(y[b], predictions['stop'], predictions[name]))
            candidates = [(self.audit_['probes'][name]['allocation_score'], name)
                          for name in ('search', 'vocabulary')]
            score, action = max(candidates)
            if score <= self.minimum_gain:
                action = 'stop'
            capacity = self.audit_['probes']['capacity']['allocation_score'] > self.minimum_gain
            self.audit_['capacity_witness'] = bool(capacity)
            self.audit_['capacity_witness_is_certificate'] = False
        else:
            action = 'stop'
            self.audit_['probe_skip_reason'] = 'insufficient class or group support'
        self.audit_['recommended_action'] = action
        if self.force_action is not None:
            action = self.force_action
        self.audit_['action'] = action
        self.vocabulary_ = (mdl_vocabulary(X, y)[0] if action == 'vocabulary'
                            else self.median_vocabulary(X))
        Xb, Vb = apply_vocabulary(X, self.vocabulary_), apply_vocabulary(X_val, self.vocabulary_)
        remaining = self.total_score_budget-self.audit_['scores_used']
        # Stopping deliberately saves work; compare both maximum allowances and cost.
        final_cap = min(remaining, self.total_score_budget//6) if action == 'stop' else remaining
        self.model_ = self._learner(self.literal_budget, final_cap).fit(Xb, y, Vb, y_val)
        self.audit_['final_score_cap'] = final_cap
        self.audit_['scores_used'] += self.model_.search_work_.get('scored_neighbor_evaluations', 0)
        if self.audit_['scores_used'] > self.total_score_budget:
            raise AssertionError('Absolute work allowance exceeded')
        self.audit_['collision'] = balanced_collision(Xb, y)
        self.audit_['fit_seconds'] = time.perf_counter()-started
        return self

    def predict(self, X):
        return self.model_.predict(apply_vocabulary(np.asarray(X, float), self.vocabulary_))

    def export(self):
        return dict(vocabulary=self.vocabulary_, controller=self.audit_, model=self.model_.export())
