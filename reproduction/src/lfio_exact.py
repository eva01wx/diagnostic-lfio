"""Small finite-vocabulary MILP oracle for search diagnostics, not a scalable learner."""
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from src.lfio_research import ValidatedLFIO


def solve_pool(X, y, pool, *, head=1, alpha=0., literal_budget=None,
               max_rules=32, time_limit=30., max_candidates=2000):
    """Bound the SAME weighted error/length objective in a fixed candidate pool.

    Each rule is a binary variable. Each distinct observed assignment has a
    coverage variable constrained to the OR of its covering selected rules.
    Duplicates retain their class multiplicities. Only status 0 certifies an
    optimum; a time limit is reported together with any incumbent/lower bound.
    """
    X, y = ValidatedLFIO._check(X, y)
    if head not in (0, 1) or alpha < 0 or max_rules < 0 or time_limit <= 0:
        raise ValueError('Invalid exact-search parameters')
    if literal_budget is not None and literal_budget < 0:
        raise ValueError('literal_budget must be nonnegative')
    if len(pool) > max_candidates:
        raise ValueError('Exact diagnostic candidate limit exceeded')
    np_, nn = int(np.sum(y == head)), int(np.sum(y != head))
    wp, wn = len(y) / (2 * max(1, np_)), len(y) / (2 * max(1, nn))
    penalty = alpha * math.log2(max(2, len(y)))
    domain, inverse = np.unique(X, axis=0, return_inverse=True)
    positives = np.bincount(inverse, weights=y == head, minlength=len(domain))
    negatives = np.bincount(inverse, weights=y != head, minlength=len(domain))
    m, n = len(pool), len(domain)
    lengths = np.array([len(r.literals) for r in pool])
    cost = np.r_[penalty * (lengths + 1), wn * negatives - wp * positives]
    constant = float(wp * np_)
    rows, cols, values, upper = [], [], [], []

    def constraint(indices, coefficients, bound):
        row = len(upper)
        rows.extend([row] * len(indices))
        cols.extend(indices)
        values.extend(coefficients)
        upper.append(bound)

    coverage = np.zeros((n, m), dtype=bool)
    for j, rule in enumerate(pool):
        mask = np.ones(n, dtype=bool)
        for feature, value in rule.literals:
            mask &= domain[:, feature] == value
        coverage[:, j] = mask
        # Independently validate that the oracle and fitted learner use the same coverage.
        if ValidatedLFIO._bits(mask[inverse]) != rule.bits:
            raise ValueError('Candidate coverage does not match the fitting data')
    for i in range(n):
        active = np.flatnonzero(coverage[i]).tolist()
        constraint([m + i, *active], [1., *([-1.] * len(active))], 0.)
        for j in active:
            constraint([j, m + i], [1., -1.], 0.)
    constraint(list(range(m)), [1.] * m, max_rules)
    if literal_budget is not None:
        constraint(list(range(m)), lengths.tolist(), literal_budget)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(upper), m + n)).tocsc()
    result = milp(cost, integrality=np.ones(m + n), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(matrix, -np.inf, upper),
                  options={'time_limit': float(time_limit), 'mip_rel_gap': 0.})
    output = {'status': int(result.status), 'message': str(result.message),
              'optimal': result.status == 0, 'candidate_count': m,
              'head': head, 'alpha': alpha, 'literal_budget': literal_budget,
              'scope': 'training objective in the supplied finite vocabulary only',
              'objective': None, 'lower_bound': None, 'selected_indices': []}
    lower = getattr(result, 'mip_dual_bound', None)
    if lower is not None and np.isfinite(lower):
        output['lower_bound'] = float(lower + constant)
    if result.x is not None:
        selected = np.flatnonzero(result.x[:m] > .5).tolist()
        rules = [pool[j] for j in selected]
        if len(rules) > max_rules or (literal_budget is not None and
                                      sum(len(r.literals) for r in rules) > literal_budget):
            raise AssertionError('MILP returned an infeasible program')
        p = ValidatedLFIO._bits(y == head)
        q = ValidatedLFIO._bits(y != head)
        objective = ValidatedLFIO._objective(rules, p, q, wp, wn, penalty)
        if not np.isclose(objective, result.fun + constant, atol=1e-6):
            raise AssertionError('MILP and executable-rule objectives disagree')
        output.update(objective=objective, selected_indices=selected,
                      rules=len(rules), total_literals=sum(len(r.literals) for r in rules))
    return output
