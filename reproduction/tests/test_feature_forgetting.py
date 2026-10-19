import copy
import itertools
import json

import numpy as np
import pytest

from research_baselines.forgetting_oracle import enumeration_bounds
from src.feature_forgetting import FeatureForgetter
from src.rule_generalization import Atom, RuleProgram, verify_relation


def program(rules, head=1, features=3):
    return RuleProgram(tuple(tuple(Atom(*a) for a in r) for r in rules), head, features)


@pytest.mark.parametrize('head', [0, 1])
def test_global_projection_combines_both_polarities_and_retains_certainty(head):
    original = program([[(0, 0., True), (1, 0., True)],
                        [(0, 0., False), (2, 0., True)]], head)
    model = FeatureForgetter().fit(original, [0])
    X = np.array([[np.nan, 1., 1.], [np.nan, -1., -1.],
                  [np.nan, 1., -1.], [np.nan, -1., 1.]])
    np.testing.assert_array_equal(model.predict(X), [head, 1-head, -1, -1])
    lower = program([[(1, 0., True), (2, 0., True)]], head)
    upper = program([[(1, 0., True)], [(2, 0., True)]], head)
    assert verify_relation(lower, model.certain_)['status'] == 'proved'
    assert verify_relation(upper, model.possible_)['status'] == 'proved'
    assert model.audit_['certain_exact']
    assert all(a.feature != 0 for p in (model.certain_,model.possible_) for r in p.rules for a in r)


def test_contradictory_numeric_thresholds_are_removed_before_projection():
    original = program([[(0, 2., True), (0, 1., False)], [(1, 0., True)]])
    model = FeatureForgetter().fit(original, [0])
    np.testing.assert_array_equal(model.predict([[np.nan, -1., 0.], [np.nan, 1., 0.]]), [0, 1])
    assert model.possible_ == model.certain_


def test_unused_feature_identity_needs_no_complement_or_resource_allowance():
    original = program([[(0,0.,True),(1,0.,True)]])
    model = FeatureForgetter(max_nodes=1,max_boxes=1).fit(original,[2])
    assert model.audit_['certain_exact']
    assert model.audit_['box_subtractions'] == 0
    assert model.certain_ == model.possible_
    np.testing.assert_array_equal(model.predict([[1.,1.,np.nan],[-1.,1.,np.nan]]),[1,0])


def test_union_must_cover_all_hidden_intervals_including_strict_boundaries():
    complete = program([[(0, 0., False)], [(0, 0., True), (0, 1., False)], [(0, 1., True)]])
    assert FeatureForgetter().fit(complete, [0]).predict([[np.nan, 0., 0.]])[0] == 1
    gap = program([[(0, 0., False)], [(0, 1., True)]])
    assert FeatureForgetter().fit(gap, [0]).predict([[np.nan, 0., 0.]])[0] == -1


@pytest.mark.parametrize('head', [0, 1])
def test_random_projections_match_independent_enumeration_for_all_feature_sets(head):
    rng = np.random.default_rng(20261016)
    grid = np.array(list(itertools.product((-2., -1., np.nextafter(-1., 0.), 0.,
                                           np.nextafter(0., 1.), 1., 2.), repeat=3)))
    for _ in range(18):
        rules = [[(int(rng.integers(3)), float(rng.integers(-1, 2)), bool(rng.integers(2)))
                  for _ in range(int(rng.integers(0, 5)))] for _ in range(int(rng.integers(0, 7)))]
        original = program(rules, head)
        for length in range(4):
            for features in itertools.combinations(range(3), length):
                model = FeatureForgetter().fit(original, features)
                actual = model.bounds(grid)
                expected = enumeration_bounds(original, grid, features)
                assert model.audit_['certain_exact']
                for key in ('certain_head','possible_head'):
                    np.testing.assert_array_equal(actual[key], expected[key])
                covered = actual['prediction'] >= 0
                np.testing.assert_array_equal(actual['prediction'][covered], original.predict(grid)[covered])


def test_joint_and_sequential_forgetting_are_semantically_equal():
    original = program([[(0, 0., True), (1, 0., True), (2, 0., True)],
                        [(0, 0., False), (1, 0., False)]])
    joint = FeatureForgetter().fit(original, [0, 1])
    first = FeatureForgetter().fit(original, [0])
    existential = FeatureForgetter().fit(first.possible_, [1])
    universal = FeatureForgetter().fit(first.certain_, [1])
    assert verify_relation(joint.possible_, existential.possible_)['status'] == 'proved'
    assert verify_relation(joint.certain_, universal.certain_)['status'] == 'proved'


@pytest.mark.parametrize('kwargs', [dict(max_nodes=1), dict(max_boxes=1)])
def test_resource_exhaustion_is_conservative_and_does_not_claim_exact_universal(kwargs):
    original = program([[(0, 0., True), (1, 0., True)]])
    model = FeatureForgetter(**kwargs).fit(original, [0])
    assert model.audit_['status'] == 'conservative'
    assert not model.audit_['certain_exact']
    assert model.certain_.rules == ()
    np.testing.assert_array_equal(model.predict([[np.nan, 1., 0.], [np.nan, -1., 0.]]), [-1,0])
    assert model.explain([np.nan,1.,0.])['status'] == 'unresolved'


def test_explanation_supplies_two_opposite_completion_regions_and_ignores_hidden_values():
    original = program([[(0, 0., True), (1, 0., True)]])
    model = FeatureForgetter().fit(original, [0])
    explanation = model.explain([np.nan, 1., 0.])
    assert explanation['status'] == 'ambiguous'
    assert explanation['head_completion_region']['0']['lower'] == 0.
    assert explanation['default_completion_region']['0']['upper'] == 0.
    assert explanation == model.explain([1000000.,1.,0.])
    np.testing.assert_array_equal(model.predict([[np.nan,1.,0.]]), model.predict([[1.,0.]],retained_only=True))


def test_constant_programs_no_features_all_features_and_empty_rows():
    for original in (program([]),program([[]])):
        for features in ([],[0,1,2]):
            model = FeatureForgetter().fit(original,features)
            np.testing.assert_array_equal(model.predict([[np.nan]*3] if features else [[0.]*3]),
                                          original.predict([[0.]*3]))
            assert model.predict(np.empty((0,len(model.retained_features_))),retained_only=True).shape == (0,)
    model = FeatureForgetter().fit(program([[(0,0.,True)]]),[0,1,2])
    np.testing.assert_array_equal(model.predict(np.empty((2,0)),retained_only=True),[-1,-1])


def test_certificate_roundtrip_recomputes_and_rejects_tampering():
    model = FeatureForgetter().fit(program([[(0,0.,True),(1,0.,True)]]),[0])
    record = json.loads(json.dumps(model.export()))
    assert FeatureForgetter.from_export(record).export() == record
    tampered = copy.deepcopy(record);tampered['certain'] = tampered['possible']
    with pytest.raises(ValueError,match='recomputed'):
        FeatureForgetter.from_export(tampered)


def test_validation_and_domain_are_explicit():
    original = program([[(0,0.,True)]])
    for features in ([True],[-1],[3],[.5]):
        with pytest.raises(ValueError):FeatureForgetter().fit(original,features)
    with pytest.raises(ValueError):FeatureForgetter(max_nodes=0)
    with pytest.raises(ValueError):FeatureForgetter().predict([[1.,2.,3.]])
    model = FeatureForgetter().fit(original,[0])
    with pytest.raises(ValueError,match='Retained'):model.predict([[np.nan,np.nan,0.]])
    with pytest.raises(ValueError,match='width'):model.predict([[0.]])
    # Quantification is over real completions, wider than representable float64.
    extreme = FeatureForgetter().fit(program([[(0,np.finfo(float).max,True)]]),[0])
    assert extreme.predict([[np.nan,0.,0.]])[0] == -1
    assert 'finite real' in extreme.audit_['quantified_domain']
