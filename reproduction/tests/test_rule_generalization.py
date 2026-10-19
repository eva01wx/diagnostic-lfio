import itertools
import json

import numpy as np
import pytest

from src.rule_generalization import (Atom, RuleGeneralizer, RuleProgram,
    boolean_simplify, exact_compress, normalize, verify_relation)


def program(rules, head=1, features=2):
    return RuleProgram(tuple(tuple(Atom(*a) for a in r) for r in rules), head, features)


def test_complementary_bodies_merge_without_changing_unobserved_predictions():
    original = program([[(0, 0., True), (1, 0., True)],
                        [(0, 0., True), (1, 0., False)]])
    result, audit = exact_compress(original)
    assert result.complexity[:2] == (1, 1)
    assert result.rules == ((Atom(0, 0., True),),)
    assert audit['exact_equivalence']
    X = np.array(list(itertools.product((-1., 0., np.nextafter(0., 1.), 1.), repeat=2)))
    np.testing.assert_array_equal(original.predict(X), result.predict(X))


def test_threshold_implication_and_contradictions_use_raw_numeric_semantics():
    original = program([[(0, 2., True), (0, 1., True)],
                        [(1, 0., True), (1, 0., False)]])
    result, _ = exact_compress(original)
    assert result.rules == ((Atom(0, 2., True),),)
    assert boolean_simplify(original).complexity[0] > result.complexity[0]


def test_union_coverage_proof_handles_more_than_pairwise_subsumption():
    original = program([[(0, 0., False)], [(0, 0., True), (0, 1., False)], [(0, 1., True)]], features=1)
    result, audit = exact_compress(original)
    assert result.rules == ((),)
    assert audit['trace']
    np.testing.assert_array_equal(result.predict(np.array([[-9.], [0.], [1.], [9.]])), 1)


def test_fit_sample_equivalence_does_not_pass_a_global_proof():
    original = program([[(0, 0., True), (1, 0., True)]])
    relaxed = program([[(0, 0., True)]])
    X = np.array([[1., 1.], [-1., -1.]])
    np.testing.assert_array_equal(original.predict(X), relaxed.predict(X))
    proof = verify_relation(original, relaxed)
    assert proof['status'] == 'disproved'
    assert proof['direction'] == 'after_not_before'
    assert verify_relation(original, relaxed, equivalent=False)['status'] == 'proved'


def test_numeric_interval_hull_with_a_gap_is_not_an_equivalent_merge():
    original = program([[(0, 0., False)], [(0, 1., True)]], features=1)
    result, _ = exact_compress(original)
    assert result.complexity[:2] == (2, 2)
    assert result.predict(np.array([[.5]]))[0] == 0


def test_proof_exhaustion_is_unknown_and_never_authorizes_a_change():
    original = program([[(0, 0., True), (1, 0., True)]])
    relaxed = program([[(0, 0., True)]])
    assert verify_relation(original, relaxed, max_nodes=0)['status'] == 'unknown'
    result, audit = exact_compress(original, max_nodes=0, total_nodes=1)
    assert result == original
    assert audit['unknown_proofs'] > 0


@pytest.mark.parametrize('head', [0, 1])
def test_random_compression_agrees_on_all_threshold_cells_and_boundaries(head):
    rng = np.random.default_rng(20261016)
    grid = np.array(list(itertools.product((-2., -1., np.nextafter(-1., 0.), 0.,
                                          np.nextafter(0., 1.), 1., 2.), repeat=3)))
    for _ in range(30):
        rules = [[(int(rng.integers(3)), float(rng.integers(-1, 2)), bool(rng.integers(2)))
                  for _ in range(int(rng.integers(1, 4)))] for _ in range(int(rng.integers(1, 7))) ]
        original = program(rules, head=head, features=3)
        result, audit = exact_compress(original)
        assert result.complexity <= original.complexity
        np.testing.assert_array_equal(original.predict(grid), result.predict(grid))
        assert verify_relation(original, result, max_nodes=100000)['status'] == 'proved'
        for row in audit['trace'][1:]:
            assert verify_relation(RuleProgram.from_export(row['before']),
                                   RuleProgram.from_export(row['after']))['status'] == 'proved'


def test_monotone_generalization_harms_only_default_teacher_class():
    original = program([[(0, 0., True), (1, 0., True)]])
    X = np.array([[1., 1.]]*100+[[1., -1.]]*100+[[-1., -1.]]*100)
    y = np.array([1]*200+[0]*100)
    model = RuleGeneralizer(mode='train').fit(original, X, y)
    assert model.program_.complexity[0] == 1
    assert verify_relation(original, model.program_, equivalent=False)['status'] == 'proved'
    assert np.all(model.predict(X)[y == 1] >= original.predict(X)[y == 1])


def test_independent_calibration_rejects_hidden_overgeneralization():
    original = program([[(0, 0., True), (1, 0., True)]])
    grow = np.array([[1., 1.]]*50+[[1., -1.]]*50+[[-1., -1.]]*50)
    teacher = np.array([1]*100+[0]*50)
    model = RuleGeneralizer(mode='calibrated').fit(original, grow, teacher)
    assert len(model.path_) > 1
    path = [p.export() for p in model.path_]
    calib = np.array([[1., -1.]]*200+[[1., 1.]]*200)
    model.calibrate(calib, np.array([0]*200+[1]*200))
    assert model.program_ == model.exact_
    assert [p.export() for p in model.path_] == path
    with pytest.raises(ValueError, match='one-shot'):
        model.calibrate(calib, np.zeros(len(calib)))


def test_zero_errors_with_too_few_calibration_examples_is_not_a_certificate():
    original = program([[(0, 0., True), (1, 0., True)]])
    grow = np.array([[1., 1.]]*20+[[1., -1.]]*20+[[-1., -1.]]*20)
    teacher = np.array([1]*40+[0]*20)
    model = RuleGeneralizer(mode='calibrated').fit(original, grow, teacher)
    model.calibrate(np.array([[-1., -1.], [1., 1.]]), np.array([0, 1]))
    assert model.audit_['selected_index'] == 0
    assert model.audit_['calibration'][0]['harmful'] == 0
    assert model.audit_['calibration'][0]['balanced_error_increase_bound'] > .02


def test_large_clean_calibration_allows_a_shorter_model_and_json_roundtrip():
    original = program([[(0, 0., True), (1, 0., True)]])
    X = np.array([[1., 1.]]*1000+[[1., -1.]]*1000+[[-1., -1.]]*1000)
    y = np.array([1]*2000+[0]*1000)
    model = RuleGeneralizer(mode='calibrated').fit(original, X[:], y[:]).calibrate(X.copy(), y.copy())
    # These synthetic arrays check the numerical gate, not sampling independence.
    assert model.program_.complexity[0] == 1
    restored = RuleProgram.from_export(json.loads(json.dumps(model.export()))['program'])
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    assert model.audit_['calibration'][0]['balanced_error_increase_bound'] <= .02


def test_exported_strict_boundaries_and_input_validation():
    original = program([[(0, 0., True), (0, 1., False)]], head=0, features=1)
    X = np.array([[0.], [np.nextafter(0., 1.)], [1.], [np.nextafter(1., 2.)]])
    np.testing.assert_array_equal(original.predict(X), [1, 0, 0, 1])
    np.testing.assert_array_equal(RuleProgram.from_export(original.export()).predict(X), original.predict(X))
    with pytest.raises(ValueError):
        original.predict([[np.nan]])
    with pytest.raises(ValueError):
        original.predict([[0., 1.]])
    with pytest.raises(ValueError):
        RuleGeneralizer(epsilon=-.1)


def test_constant_programs_are_preserved_and_never_gain_complexity():
    for original in (program([], features=1), program([[]], features=1)):
        result = RuleGeneralizer().fit(original)
        assert result.program_.complexity <= original.complexity
        assert verify_relation(original, result.program_)['status'] == 'proved'
