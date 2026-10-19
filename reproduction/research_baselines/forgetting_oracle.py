"""Independent threshold-cell enumeration for checking feature projections.

This verifier evaluates bodies directly and does not use interval subtraction,
complement construction or any projection implementation. Each forgotten
coordinate is partitioned at every threshold used by the original program.
"""
import itertools
import math

import numpy as np


def enumeration_bounds(program, X, features, *, max_cells=100000):
    X = np.asarray(X, dtype=float)
    features = tuple(sorted(set(features)))
    cells = []
    for feature in features:
        cuts = sorted({a.threshold for r in program.rules for a in r if a.feature == feature})
        endpoints = [-math.inf] + cuts + [math.inf]
        cells.append(list(zip(endpoints[:-1], endpoints[1:])))
    count = math.prod(map(len, cells))
    if count > max_cells:
        raise ValueError('Independent enumeration cell limit')
    retained_masks = []
    for rule in program.rules:
        mask = np.ones(len(X), dtype=bool)
        for atom in rule:
            if atom.feature not in features:
                test = X[:, atom.feature] > atom.threshold
                mask &= test if atom.greater else ~test
        retained_masks.append(mask)
    possible, certain = np.zeros(len(X), dtype=bool), np.ones(len(X), dtype=bool)
    for combination in itertools.product(*cells):
        intervals = dict(zip(features, combination))
        hit = np.zeros(len(X), dtype=bool)
        for rule, mask in zip(program.rules, retained_masks):
            # A threshold atom is constant on each (lo, hi] cell, including endpoints.
            satisfied = all((intervals[a.feature][0] >= a.threshold) if a.greater
                            else (intervals[a.feature][1] <= a.threshold)
                            for a in rule if a.feature in intervals)
            if satisfied:
                hit |= mask
        possible |= hit
        certain &= hit
    return dict(possible_head=possible, certain_head=certain, cells=count)
