# Contextual, cost-aware and query-based forgetting

This exploratory extension reuses the known six-dataset cohort. No new teacher
or surrogate is trained, and no result is presented as independent validation.
Freeze this protocol and the implementation hashes before running this version.
Earlier source locks, records, generated reports and ZIP releases are preserved.

## Semantics and limits

For retained r, forgotten s and explicit background C, define
V=exists s C, U=exists s(C and F), L=V and not exists s(C and not F).
Certify head only when C and F is SAT and C and not F is UNSAT; reverse
the tests for default. Both SAT means ambiguous, both UNSAT means infeasible
(-2), and any unresolved required proof yields abstention (-1). Never treat
infeasibility as a vacuously true head certificate or UNKNOWN as UNSAT.

Support conjunctions of rational difference constraints x_i-x_j <= c or < c,
with a constant-zero node. Equality is two opposite bounds. Binary64 inputs
and thresholds are interpreted as their exact rational values. Bellman--Ford
with rational/infinitesimal pairs detects strict zero cycles and supplies
checked rational witnesses. This is a restricted numerical background language,
not a general linear/Boolean constraint solver. No constraints are inferred
from benchmark correlations, observed ranges or labels.

Without background constraints, group queries by their residual hidden-input
DNF, caching exact head/default witnesses. Check the default class by bounded
interval subtraction for that residual, without materializing a global envelope.
The contextual path branches over violated body atoms and solves exact
difference constraints. Maximum work is 50,000 operations per residual query;
cache capacity is 4,096 entries with LRU eviction. Unknown is explicit.
Cache exports are specifications; query certificates are verified by fresh replay.

## A: selection experiment

Use all 648 Original base/budget programs from rule_generalization_20261016:
six datasets, seeds 7001--7003, RF/HGB, 0/20% training label noise, three bases,
budgets 8/16/32. BoundedGeneralize remains in the earlier report as a separate
source variant; this experiment isolates changes to forgetting Original.

Objective: maximize sum of hidden active-coordinate costs subject to selection
certified coverage >= q, per-query work/cache limits, and at most 512 distinct
candidate subsets per profile/program. q is 0.7, 0.8 or 0.9. Algorithms are
greedy heuristics; no subset-optimality or population coverage guarantee.
The empty subset is an admissible fallback and is counted in every mean.

Compare Fixed (index order), Frequency (fewest source atoms), Impact (least
singleton selection uncertainty), Coverage (best measured next coverage), and
CostAware (least marginal coverage loss per unit cost, then larger cost).
All policies skip infeasible additions, accept only measured feasible subsets,
and break ties by indices. They share candidate evaluations within a profile.
Costs are controlled scenarios, not claimed deployment measurements: uniform
cost 1, and heterogeneous cost 1+(coordinate index modulo 5). Costs on unused
coordinates earn no reward. Save all candidates, scores and final choices
before test inference. Test rows never determine choices or targets.

Primary descriptive comparison: CostAware versus Coverage, heterogeneous costs,
q=0.8, dataset-equal hidden active-cost fraction. Also report hidden count,
test certified coverage, class-conditional coverage, zero-hiding fraction,
selection constraint satisfaction, test shortfall frequency and budget exhaustion.
Use a descriptive paired bootstrap over the six dataset means, 10,000 draws,
seed 20261016. Repeated fits/queries do not increase independent sample size.

## B: background constraints

Controlled F=(x0>0), hide x0, with y=x1 in {-2,-1,0,1,2}: omitted C,
true equality x0=y, one-sided order x0>=y, strict band y-1/4<x0<=y+1/4,
wrong equality x0=y-2, and true equality plus y<=1 (infeasible retained y=2).
Compare predictions under each assumption with a domain where x0=y is known
by construction. Wrong-C disagreement demonstrates conditional scope, not
solver unsoundness. Add 100 seeded random three-coordinate DNF/constraint
systems, both heads, all three retained z values, for hidden {0}, {0,1},
and {0,1,2}. Independently enumerate threshold cells and use exact
Fourier--Motzkin elimination, rather than the query engine's graph algorithm.
No claim of improved real-data coverage under verified domain constraints.

## C: query versus global compilation

On every recorded Original projection from feature_forgetting_20261016_v2,
query all selection/test rows and compare with the saved compiled envelope.
Match exact envelopes, permit additional sound predictions where the old
compiler fell back, and independently check the first three test rows of each
projection using threshold-cell enumeration. Store all prediction vectors.
Also verify hidden-input independence and fresh query-certificate replay.

For the 32 largest old compiled representations (ties: path/projection index),
rerun global compilation and the query engine in the same worker/process.
Measure setup, first batch (first 64 test rows), warm batch and total CPU,
and Python allocation peak using tracemalloc. This is a deliberately difficult
representation subset, not a representative speed benchmark. Record global
expanded conditions, residual cache entries/payload bytes and unknown queries.
Python allocation peaks are not total RSS; serialized cache bytes are not human
reading complexity. No polynomial-time or general speedup claim.

## Verification and release

Check input/source SHA-256, replay every selection choice, saved test vector and
coverage, certify assigned results against fully observed source predictions,
and run independent small-cell oracles. Rebuild and visually review manuscript
and review PDF. Release code/data records under a new dated extension name,
retaining the earlier releases as provenance.

Foundations: Lin/Reiter and Lin; Doherty/Szalas 2024 for dual/background
conditions. Wang/Brewster DATA 2024 motivates task impact, without treating
graph-importance scores as logical certificates. Liberatore 2024/2025 and
DivKC 2026 motivate representation accounting, without transferring Horn
complexity results to numerical DNF or claiming their algorithms were reproduced.
