# Numerical feature forgetting: extension fixed before its evaluation

This is a post hoc extension of the known 2026-10-16 rule-generalization cohort.
It is not new external validation, new learner training, or registry preregistration.
The completed allocation/generalization source locks and records remain unchanged.

## Fixed inputs and semantics

Reuse all 648 base/budget fits (six datasets, seeds 7001--7003, RF/HGB,
0/20% teacher-training label noise, DiagnosticLFIO/MedianRestarts/MDLRestarts,
B=8/16/32). Apply the extension to both Original and BoundedGeneralize programs,
giving 1,296 source programs. Use the recorded fitting, selection and test
partitions. Calibration is not reused for a new probabilistic claim.

For the common-head DNF F, globally forget a set S of raw numerical coordinates.
The upper formula U=exists S F and lower formula L=forall S F contain none of
those coordinates. Quantify over all finite real completions; do not infer
admissible ranges or cross-feature constraints from observed data. Numerical
threshold dependencies on each raw coordinate are respected jointly. A head
prediction is certified by L, a default prediction by not U; otherwise abstain.
The deterministic guarantee is relative to the particular source surrogate,
not the teacher, reference label, or causal mechanism. BoundedGeneralize retains
its earlier, separate relation to Original and its calibration assumptions.

Normalize impossible bodies before projection. Construct U by deleting all
tests on S in all normalized bodies. Construct L as not(exists S not F) using
interval-box subtraction. Fixed limits: 50,000 subtractions and 4,096
intermediate boxes per compilation. On exhaustion retain exact U but use
L=false, explicitly conservative rather than claiming exact ambiguity.

## Queries and selection

Compile every singleton among features actually used by each source program.
An unused coordinate, if any, supplies a separately reported identity control;
do not dilute active-feature results with all unused coordinates.

Fixed1 forgets the smallest active index. Certainty1 selects the singleton
maximizing the fraction of selection rows assigned a certified class. Ties
prefer fewer combined tests in L and U, then fewer bodies, then coordinate
indices. Certainty2 keeps Certainty1's choice and evaluates adding each other
active feature jointly from the original program, using the same ordering.
Fixed2 forgets the first two active indices. If fewer than two features are
active, do not relabel a singleton as a two-feature query; report eligible
counts and restrict pair comparisons accordingly. Empty active sets are
identity controls. Compile every candidate needed by these policies, and
persist all formulas, choices and fitting medians before test inference.

## Endpoints and comparisons

Primary descriptive endpoint: dataset-equal test certified coverage for
Original under Certainty1, paired with Fixed1 on the same source programs.
Report six-dataset cluster bootstrap intervals (10,000 replicates, seed
20261016), with no claim of additional independent dataset units.

Secondary endpoints: Certainty2/Fixed2 coverage, all-active-singleton coverage,
head/default coverage, certain-exact completion rate, contradictions with the
source surrogate on assigned test rows, projection sizes, compilation CPU and
the fraction whose combined L/U condition count exceeds the source condition
count. Forgetting fewer coordinates need not yield a shorter representation.

Compare two forced-prediction references on the same selected missing features:
head if U else default (existential-only), and source prediction after filling
missing coordinates with fitting-set medians. Report disagreements with the
fully observed source and teacher macro fidelity. They do not abstain and do
not carry our completion-invariant guarantee; their unconditional fidelity
must not be compared with selective fidelity as if the denominators agreed.
No imputation baseline is claimed globally unsound merely because a finite
test fold has no disagreements.

## Verification and limits

Recompute all exports and verify them independently by enumerating every
forgotten-coordinate threshold cell on both selection and test retained rows.
The checker uses direct body evaluation and no interval-complement routine.
Exact L/U must match enumeration; conservative L must be contained in the true
universal region. Assigned predictions must agree with the source at every
recorded complete row. Reconstruct policies from selection data and source
locks; test scores never select coordinates. Replay saved vectors and medians.
Tests also exhaust low-dimensional threshold cells/boundaries, both head
polarities, contradictions, empty/all feature sets, limits, and tampered exports.

Logical forgetting and the necessary/sufficient-condition duality are classical
(Lin and Reiter 1994; Lin 2001). The extension contributes executable numerical
semantics, bounded construction, partial-input certificates and an auditable
integration/evaluation. No minimum explanation, teacher-level guarantee, human
comprehension, causal intervention, or universal compression claim is made.
