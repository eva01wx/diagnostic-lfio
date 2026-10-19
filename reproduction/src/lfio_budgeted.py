"""Optional scored-work allocation and teacher-informed predicates for Refined LFIO.

The historical implementation and its recorded experiments remain unchanged.
All search allocation and predicate construction below use fitting data only.
"""
import hashlib
import heapq
import math

import numpy as np

from .lfio_research import RefinedLFIO


class ScoredWorkLFIO(RefinedLFIO):
    """One training winner per cell under a shared scored-neighbor allowance.

Both policies use ordinary and pure initializations, then randomized greedy
restarts (uniform choice among the three best improving additions). The repair
policy spends part of the same allowance on add/delete/replace moves. A score
unit is a feasible program-neighbor objective evaluation, including deletions.
It is not a CPU-time unit. Exhaustion retains the best evaluated feasible move.
"""

    def __init__(self, *, work_multiplier=32, allocate_repair=True, random_state=0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(work_multiplier, int) or work_multiplier < 1:
            raise ValueError("work_multiplier must be a positive integer")
        self.work_multiplier = work_multiplier
        self.allocate_repair = allocate_repair
        self.random_state = int(random_state)

    def _fit_variants(self, pool, positive, negative, n_samples, alpha):
        wp = n_samples / (2 * max(1, positive.bit_count()))
        wn = n_samples / (2 * max(1, negative.bit_count()))
        penalty = alpha * math.log2(max(2, n_samples))
        limit = self.work_multiplier * len(pool)
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
                     score_budget=limit, scores_used=used, work_multiplier=self.work_multiplier,
                     initializations=restarts, repair_moves_evaluated_path=repair_moves,
                     selected_origin=best_origin, training_objective=best_loss,
                     validation_policy="one training-objective winner per cell",
                     random_state=self.random_state, seed_digest=seed)
        yield "bounded_repair" if self.allocate_repair else "bounded_restarts", best_rules, best_history, audit


def mdl_vocabulary(fitting, teacher_labels, max_extra_cuts=2, min_leaf=3):
    """Median plus at most two teacher-label MDL cuts per numerical feature.

Uses the Fayyad--Irani binary MDL acceptance criterion. Best-first truncation,
minimum leaf support and retaining the median are explicit adaptations. This
is an application of established discretization, not a new MDL criterion.
"""
    X = np.asarray(fitting, dtype=float)
    y = np.asarray(teacher_labels)
    if (X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all()
            or y.shape != (len(X),) or not np.isin(y, [0, 1]).all()):
        raise ValueError("Expected finite fitting inputs and aligned Boolean teacher labels")
    if not isinstance(max_extra_cuts, int) or max_extra_cuts < 0 or min_leaf < 1:
        raise ValueError("Invalid cut or support limit")

    def entropy(ones, counts):
        fraction = np.divide(ones, counts)
        result = np.zeros_like(fraction, dtype=float)
        mask = (fraction > 0) & (fraction < 1)
        result[mask] = -fraction[mask]*np.log2(fraction[mask]) - (1-fraction[mask])*np.log2(1-fraction[mask])
        return result

    vocabulary, audit = [], []
    for feature in range(X.shape[1]):
        order = np.argsort(X[:, feature], kind="stable")
        values, target = X[order, feature], y[order].astype(int)
        prefix = np.r_[0, np.cumsum(target)]
        median = float(np.median(values))

        def proposal(lo, hi):
            n = hi-lo
            ones = int(prefix[hi]-prefix[lo])
            if n < 2*min_leaf or ones in (0, n):
                return None
            positions = np.arange(lo+min_leaf, hi-min_leaf+1)
            positions = positions[values[positions-1] < values[positions]]
            if not len(positions):
                return None
            left_n, right_n = positions-lo, hi-positions
            left_p = prefix[positions]-prefix[lo]
            right_p = ones-left_p
            parent_h = float(entropy(np.array([ones]), np.array([n]))[0])
            left_h, right_h = entropy(left_p, left_n), entropy(right_p, right_n)
            gains = parent_h - (left_n*left_h+right_n*right_h)/n
            index = int(np.argmax(gains))
            cut = int(positions[index])
            k_left = 1 + int(0 < left_p[index] < left_n[index])
            k_right = 1 + int(0 < right_p[index] < right_n[index])
            delta = math.log2(7) - (2*parent_h-k_left*left_h[index]-k_right*right_h[index])
            bound = (math.log2(n-1)+delta)/n
            gain = float(gains[index])
            if gain <= bound + 1e-12:
                return None
            lower, upper = float(values[cut-1]), float(values[cut])
            threshold = lower/2 + upper/2
            if not lower <= threshold < upper:
                threshold = lower
            return dict(lo=lo, hi=hi, cut=cut, threshold=threshold, samples=n,
                        gain=gain, mdl_bound=float(bound), priority=n*gain)

        pending, accepted = [], []
        first = proposal(0, len(values))
        if first:
            pending.append(first)
        while pending and len(accepted) < max_extra_cuts:
            pending.sort(key=lambda p: (-p["priority"], p["threshold"]))
            current = pending.pop(0)
            accepted.append(current)
            for lo, hi in ((current["lo"], current["cut"]), (current["cut"], current["hi"])):
                child = proposal(lo, hi)
                if child:
                    pending.append(child)
        thresholds = sorted({median, *(item["threshold"] for item in accepted)})
        vocabulary.extend((feature, float(t)) for t in thresholds)
        audit.append(dict(feature=feature, median=median, accepted=accepted, thresholds=thresholds))
    return vocabulary, audit
