"""Selection-only cost/coverage constrained coordinate hiding.

These are audited greedy heuristics, not globally optimal subset algorithms.
Only active source coordinates are eligible; hiding unused inputs earns nothing.
"""
import numpy as np

from .contextual_forgetting import ContextualForgetter


POLICIES = ('Fixed', 'Frequency', 'Impact', 'Coverage', 'CostAware')


def select_features(program, selection, costs, *, policies=POLICIES,
                    targets=(.7, .8, .9), max_nodes=50000, max_cache=4096,
                    max_candidates=512):
    selection = np.asarray(selection, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if selection.ndim != 2 or selection.shape[1] != program.n_features or not len(selection):
        raise ValueError('Nonempty selection matrix required')
    if not np.isfinite(selection).all() or costs.shape != (program.n_features,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Finite inputs and positive per-coordinate costs required')
    if not targets or any(not 0 <= t <= 1 for t in targets) or any(p not in POLICIES for p in policies):
        raise ValueError('Invalid coverage targets or policies')
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError('Candidate budget must be a positive integer')
    active = sorted({a.feature for rule in program.rules for a in rule})
    frequency = {f:sum(a.feature == f for r in program.rules for a in r) for f in active}
    candidates = {}
    def evaluate(features):
        features = tuple(sorted(features))
        if features not in candidates:
            if len(candidates) >= max_candidates:
                return None
            engine = ContextualForgetter(max_nodes=max_nodes, max_cache=max_cache).fit(program, features)
            predictions = engine.predict(selection)
            candidates[features] = dict(features=list(features), coverage=float(np.mean(predictions >= 0)),
                hidden_cost=float(costs[list(features)].sum()),
                patterns=engine.stats_['solved_patterns'], unknown_patterns=engine.stats_['unknown_patterns'])
        return candidates[features]
    evaluate(())
    for f in active:
        evaluate((f,))
    impact = {f:1-candidates[(f,)]['coverage'] if (f,) in candidates else 1. for f in active}
    choices = []
    # Independent path for each target; every accepted extension is measured.
    for policy in policies:
        for target in targets:
            hidden, exhausted = (), False
            while len(hidden) < len(active):
                remaining = [f for f in active if f not in hidden]
                if policy in ('Fixed', 'Frequency', 'Impact'):
                    order = sorted(remaining, key=(lambda f:f) if policy == 'Fixed' else
                                   (lambda f:(frequency[f],f)) if policy == 'Frequency' else
                                   (lambda f:(impact[f],f)))
                    accepted = None
                    for f in order:
                        record = evaluate((*hidden,f))
                        if record is None:
                            exhausted = True
                            break
                        if record['coverage'] >= target:
                            accepted = record
                            break
                else:
                    options = []
                    for f in remaining:
                        record = evaluate((*hidden,f))
                        if record is None:
                            exhausted = True
                            break
                        if record['coverage'] >= target:
                            loss = max(0., candidates[tuple(hidden)]['coverage']-record['coverage'])
                            key = ((-record['coverage'],f) if policy == 'Coverage' else
                                   (loss/costs[f],-costs[f],-record['coverage'],f))
                            options.append((key,record))
                    accepted = min(options,key=lambda pair:pair[0])[1] if options else None
                if accepted is None:
                    break
                hidden = tuple(accepted['features'])
            choices.append(dict(policy=policy,target=float(target),**candidates[hidden],
                                candidate_budget_exhausted=exhausted))
    return dict(active_features=active,costs=costs.tolist(),singleton_impact=impact,
                frequency=frequency,choices=choices,candidates=list(candidates.values()),
                candidate_limit=max_candidates,query_limits=dict(max_nodes=max_nodes,max_cache=max_cache),
                guarantee='selection coverage constraint only; test coverage is not guaranteed',
                optimality_claim=False)
