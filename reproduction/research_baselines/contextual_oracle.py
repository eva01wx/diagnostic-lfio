"""Small-case independent oracle: threshold cells plus Fourier--Motzkin.

Uses neither difference-graph relaxation nor interval subtraction. Exponential
elimination is intentional: this verifier is for small controlled cases only.
"""
from fractions import Fraction
import itertools
import math


def _feasible(rows, features):
    for coordinate in features:
        positive, negative, zero = [], [], []
        for coefficients, bound, strict in rows:
            value = coefficients.get(coordinate, 0)
            (positive if value > 0 else negative if value < 0 else zero).append((coefficients,bound,strict))
        rows = zero
        for a, b, strict_a in positive:
            for c, d, strict_c in negative:
                left, right = a[coordinate], -c[coordinate]
                coefficients = {f:a.get(f,0)/left+c.get(f,0)/right for f in set(a)|set(c) if f != coordinate}
                rows.append(({f:v for f,v in coefficients.items() if v}, b/left+d/right, strict_a or strict_c))
        for coefficients, bound, strict in rows:
            if not coefficients and (bound < 0 or (bound == 0 and strict)):
                return False
    return all(bound > 0 or (bound == 0 and not strict) for _,bound,strict in rows)


def constrained_bounds(program, row, hidden, constraints=(), *, max_cells=10000):
    hidden = tuple(sorted(hidden))
    cells = []
    for f in hidden:
        cuts = sorted({a.threshold for r in program.rules for a in r if a.feature == f})
        points = [-math.inf]+cuts+[math.inf]
        cells.append(list(zip(points[:-1],points[1:])))
    if math.prod(map(len,cells)) > max_cells:
        raise ValueError('Oracle cell cap')
    base = []
    for bound in constraints:
        coefficients, value = {}, Fraction(bound.bound)
        for f, sign in ((bound.left,1),(bound.right,-1)):
            if f is None:
                continue
            if f in hidden:
                coefficients[f] = coefficients.get(f,Fraction(0))+sign
            else:
                value -= sign*Fraction(float(row[f]))
        base.append(({f:v for f,v in coefficients.items() if v},value,bound.strict))
    possible, default, legal = False, False, 0
    for intervals in itertools.product(*cells):
        region = dict(zip(hidden,intervals))
        rows = list(base)
        for f,(lo,hi) in region.items():
            if math.isfinite(lo): rows.append(({f:Fraction(-1)},-Fraction(lo),True))
            if math.isfinite(hi): rows.append(({f:Fraction(1)},Fraction(hi),False))
        if not _feasible(rows, hidden):
            continue
        legal += 1
        hit = any(all((region[a.feature][0] >= a.threshold if a.greater else region[a.feature][1] <= a.threshold)
                      if a.feature in region else bool(row[a.feature]>a.threshold) == a.greater for a in rule)
                  for rule in program.rules)
        possible |= hit
        default |= not hit
    prediction = (-2 if not legal else program.head if not default else
                  1-program.head if not possible else -1)
    return dict(prediction=prediction, possible_head=possible, possible_default=default, feasible_cells=legal)
