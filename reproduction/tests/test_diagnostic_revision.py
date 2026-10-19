import itertools
import numpy as np
import pytest
from src.lfio_diagnostic import AbsoluteWorkLFIO, DiagnosticLFIO, balanced_collision
from src.lfio_followup import apply_vocabulary
from research_baselines.prospective import (fast_export, fast_frame, fast_predict,
    native_neurules_export, native_neurules_predict)


def test_balanced_collision_matches_all_decoders_with_unequal_class_sizes():
    X = np.array([[0], [0], [0], [1], [1], [1], [1]])
    y = np.array([0, 0, 1, 0, 1, 1, 1])
    losses = []
    for values in itertools.product((0, 1), repeat=2):
        p = np.array(values)[X[:, 0]]
        losses.append(.5*np.mean(p[y == 0] != 0)+.5*np.mean(p[y == 1] != 1))
    assert balanced_collision(X, y)['error_floor'] == pytest.approx(min(losses))
    rich = np.column_stack([X, y])
    assert balanced_collision(rich, y)['error_floor'] == 0


def test_internal_action_does_not_use_outer_validation_and_respects_total_work():
    X = np.column_stack([np.arange(160), np.tile([0., 1.], 80)])
    y = (X[:, 0] < 39).astype(int)
    params = dict(literal_budget=4, total_score_budget=12000, random_state=29)
    first = DiagnosticLFIO(**params).fit(X, y, X, y)
    second = DiagnosticLFIO(**params).fit(X, y, X, 1-y)
    assert first.audit_['action'] == second.audit_['action']
    assert first.audit_['probes'] == second.audit_['probes']
    assert first.vocabulary_ == second.vocabulary_
    assert first.audit_['scores_used'] <= 12000
    assert first.audit_['action'] == 'vocabulary'
    assert sum(len(r.literals) for r in first.model_._selected) <= 4
    a, b = (first.audit_[key] for key in ('inner_fit_indices', 'inner_holdout_indices'))
    assert not set(a) & set(b)
    assert len(a)+len(b) == len(X)
    assert len(first.model_.selection_trace_) == 6
    assert len(first.audit_['probes']['capacity']['vocabulary']) == X.shape[1]


def test_absolute_caps_are_independent_of_pool_size_and_repair_policy():
    X = np.array(list(itertools.product((0, 1), repeat=4)))
    y = (X.sum(axis=1) >= 2).astype(int)
    for repair in (False, True):
        model = AbsoluteWorkLFIO(depths=(2,), penalties=(0.,), cell_score_budget=431,
            allocate_repair=repair, literal_budget=5).fit(X, y, X, y)
        assert model.search_work_['scored_neighbor_evaluations'] == 862
        assert all(c['optimization']['scores_used'] == 431 for c in model.selection_trace_)
        assert np.all(np.diff(model.objective_path_) <= 1e-8)


def test_controller_constant_target_and_zero_budget_are_executable():
    X = np.arange(100).reshape(50, 2)
    for label in (0, 1):
        model = DiagnosticLFIO(0, 1200).fit(X, np.full(50, label), X, np.full(50, label))
        np.testing.assert_array_equal(model.predict(X), label)
        assert model.audit_['scores_used'] == 0
    with pytest.raises(ValueError):
        DiagnosticLFIO(total_score_budget=0)


def test_capacity_witness_can_fire_without_deploying_the_larger_probe():
    rng = np.random.default_rng(91)
    bits = np.repeat(np.array(list(itertools.product((0, 1), repeat=2))), 80, axis=0)
    X = bits+rng.uniform(-.02, .02, size=bits.shape)
    y = bits[:, 0] ^ bits[:, 1]
    # XOR needs two two-literal bodies. The probe can use four tests, while
    # every deployed candidate must retain the original two-test allowance.
    model = DiagnosticLFIO(2, 12000, 9).fit(X, y, X, y)
    assert model.audit_['capacity_witness']
    assert not model.audit_['capacity_witness_is_certificate']
    assert sum(len(r.literals) for r in model.model_._selected) <= 2


def test_fastsrs_multivalue_export_keeps_disjunction_and_counts_each_test():
    raw = np.array([[0., 3.], [2., 1.], [4., 1.]])
    vocabulary = [(0, 1.), (0, 3.), (1, 2.)]
    matrix, names = fast_frame(raw, vocabulary)
    export = fast_export([{'x0': [1, 2], 'x1': [4]}], 1, vocabulary, names)
    np.testing.assert_array_equal(fast_predict(export, raw),
        ((matrix[:, [1, 2]].sum(axis=1) > 0) & (matrix[:, 4] > 0)).astype(int))
    assert export['total_literals'] == 3 and export['native_conditions'] == 2


def test_native_neurules_export_preserves_strict_float32_interval_boundaries():
    import torch
    from research_baselines.adapters import neurules_modules
    models, _, _ = neurules_modules()
    hard = models.HardRuleList(
        torch.tensor([[[0., -1.], [1., 2.]]]),
        torch.tensor([[1.], [1.]]), torch.tensor([2., 1.]),
        torch.tensor([[0., 1.], [1., 0.]]))
    export = native_neurules_export(hard, {}, np.array([0.]), np.array([1.]))
    raw = np.array([[-1.], [0.], [np.nextafter(np.float32(0), np.float32(1))],
                    [.5], [1.], [2.]], dtype=np.float32)
    np.testing.assert_array_equal(hard(torch.tensor(raw)).numpy().argmax(axis=1),
                                  native_neurules_predict(export, raw))
    assert export['total_literals'] == 4
