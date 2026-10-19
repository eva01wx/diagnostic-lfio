import copy
from fractions import Fraction
import itertools
import json

import numpy as np
import pytest

from research_baselines.contextual_oracle import constrained_bounds
from research_baselines.forgetting_oracle import enumeration_bounds
from src.contextual_forgetting import ContextualForgetter, DifferenceBound as B
from src.forgetting_selection import select_features
from src.rule_generalization import Atom, RuleProgram


def program(rules, head=1, n=3):
    return RuleProgram(tuple(tuple(Atom(*a) for a in r) for r in rules),head,n)


@pytest.mark.parametrize('head',[0,1])
def test_true_context_recovers_both_classes_and_missing_context_abstains(head):
    source = program([[(0,0.,True)]],head)
    X = np.array([[np.nan,-1.,0.],[np.nan,0.,0.],[np.nan,1.,0.]])
    assert ContextualForgetter().fit(source,[0]).predict(X).tolist() == [-1]*3
    model = ContextualForgetter().fit(source,[0],[B(0,1),B(1,0)])
    assert model.predict(X).tolist() == [1-head,1-head,head]
    assert model.query([np.nan,1.,0.])['head_witness']['0'] == '1'


def test_infeasibility_is_distinct_from_abstention_and_vacuous_truth():
    source = program([[(0,0.,True)]])
    model = ContextualForgetter().fit(source,[0],[B(0,1),B(1,0),B(1,None,1)])
    result = model.query([np.nan,2.,0.])
    assert result['prediction'] == -2 and result['head_status'] == result['default_status'] == 'UNSAT'
    assert model.query([np.nan,0.,0.])['status'] == 'certain_default'


def test_strict_cycles_open_boundaries_and_exact_fraction_witnesses():
    source = program([[(0,0.,True)]])
    impossible = ContextualForgetter().fit(source,[0,1],[B(0,1,0,True),B(1,0)])
    assert impossible.query([np.nan,np.nan,0.])['prediction'] == -2
    model = ContextualForgetter().fit(source,[0],[B(None,0,0,True),B(0,None,'1/1000000000000000000000000000000',True)])
    result = model.query([np.nan,0.,0.])
    assert result['status'] == 'certain_head'
    assert 0 < Fraction(result['head_witness']['0']) < Fraction(1,10**30)


@pytest.mark.parametrize('head',[0,1])
def test_unrestricted_queries_match_independent_threshold_cell_enumeration(head):
    rng = np.random.default_rng(42616)
    X = np.asarray(list(itertools.product((-1.,0.,np.nextafter(0.,1.),1.),repeat=3)))
    for _ in range(18):
        rules = [[(int(rng.integers(3)),float(rng.integers(-1,2)),bool(rng.integers(2)))
                  for _ in range(int(rng.integers(5)))] for _ in range(int(rng.integers(6)))]
        source = program(rules,head)
        for length in range(4):
            for hidden in itertools.combinations(range(3),length):
                model = ContextualForgetter().fit(source,hidden)
                expected = enumeration_bounds(source,X,hidden)
                predictions = np.where(expected['certain_head'],head,
                                       np.where(~expected['possible_head'],1-head,-1))
                np.testing.assert_array_equal(model.predict(X),predictions)


def test_contextual_queries_match_independent_fourier_motzkin_oracle():
    rng = np.random.default_rng(82616)
    for _ in range(50):
        source = program([[(int(rng.integers(3)),float(rng.integers(-1,2)),bool(rng.integers(2)))
                           for _ in range(int(rng.integers(1,4)))] for _ in range(int(rng.integers(5)))])
        bounds = [B(int(rng.integers(3)),int(rng.integers(3)),int(rng.integers(-1,2)),bool(rng.integers(2))) for _ in range(3)]
        for hidden in ((0,),(0,1),(0,1,2),()):
            model = ContextualForgetter().fit(source,hidden,bounds)
            for row in ((-1.,0.,1.),(0.,0.,0.),(1.,-1.,0.)):
                actual = model.query(row)
                expected = constrained_bounds(source,row,hidden,bounds)
                assert actual['prediction'] == expected['prediction']
                for key,want in (('head_status',expected['possible_head']),('default_status',expected['possible_default'])):
                    assert actual[key] == ('SAT' if want else 'UNSAT')


def test_resource_limit_returns_unknown_never_a_false_certificate():
    source = program([[(0,0.,True)]])
    model = ContextualForgetter(max_nodes=1).fit(source,[0],[B(0,1)])
    assert model.query([np.nan,0.,0.])['status'] == 'unresolved'
    assert model.predict([[np.nan,0.,0.]]).tolist() == [-1]


def test_hidden_values_cache_retained_interface_and_evictions():
    source = program([[(0,0.,True),(1,0.,True)],[(0,0.,False),(2,0.,True)]])
    model = ContextualForgetter(max_cache=1).fit(source,[0])
    first = model.query([np.nan,1.,1.])
    assert model.query([999.,1.,1.]) == first
    assert model.query([1.,1.],retained_only=True) == first
    assert model.stats_['cache_hits'] == 2
    model.query([0.,-1.,-1.]); model.query([0.,1.,1.])
    assert model.stats_['evictions'] == 2
    assert model.cache_bytes() > 0
    assert model.predict(np.empty((0,2)),retained_only=True).shape == (0,)


def test_context_cache_uses_values_not_only_predicate_truth():
    source = program([[(0,0.,True)]])
    model = ContextualForgetter().fit(source,[0],[B(0,1),B(1,0)])
    assert model.query([np.nan,-1.,0.])['prediction'] == 0
    assert model.query([np.nan,1.,0.])['prediction'] == 1


def test_exact_rational_witness_at_maximum_float_does_not_overflow():
    maximum = np.finfo(float).max
    source = program([[(0,maximum,True)]])
    record = ContextualForgetter().fit(source,[0]).query([np.nan,0.,0.])
    assert record['status'] == 'ambiguous'
    assert Fraction(record['head_witness']['0']) > Fraction(maximum)


def test_roundtrip_and_certificate_replay_reject_tampered_status():
    source = program([[(0,0.,True)]])
    model = ContextualForgetter().fit(source,[0],[B(0,1),B(1,0)])
    exported = json.loads(json.dumps(model.export()))
    assert ContextualForgetter.from_export(exported).export() == exported
    row = [np.nan,1.,0.]; result=model.query(row)
    assert model.verify_query(row,result)
    altered=copy.deepcopy(result);altered['prediction']=0
    with pytest.raises(ValueError): model.verify_query(row,altered)


def test_cost_selection_changes_choice_and_respects_selection_constraints():
    source = program([[(0,0.,True),(1,0.,True)]],n=3)
    X = np.asarray([[1.,1.,0.],[-1.,1.,0.],[1.,-1.,0.],[-1.,-1.,0.]])
    selected=select_features(source,X,[1.,9.,100.],targets=(.5,1.))
    choices={(r['policy'],r['target']):r for r in selected['choices']}
    assert choices['CostAware',.5]['features'] == [1]
    assert choices['Coverage',.5]['features'] == [0]
    for choice in selected['choices']:
        assert choice['coverage'] >= choice['target']
        assert 2 not in choice['features']
        model=ContextualForgetter().fit(source,choice['features'])
        assert np.mean(model.predict(X)>=0) == choice['coverage']


def test_selection_candidate_budget_is_explicit_and_does_not_invent_evidence():
    source=program([[(0,0.,True),(1,0.,True)]])
    result=select_features(source,np.ones((2,3)),[1.,2.,3.],max_candidates=1)
    assert all(r['features'] == [] for r in result['choices'])
    assert any(r['candidate_budget_exhausted'] for r in result['choices'])


@pytest.mark.parametrize('kwargs',[dict(max_cache=0),dict(max_nodes=True)])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):ContextualForgetter(**kwargs)
