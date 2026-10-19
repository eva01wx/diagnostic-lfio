import json
import numpy as np
import pytest

from src.lfio_generalized import GeneralizedLFIO
from src.rule_generalization import RuleProgram, verify_relation


def data():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(400, 2))
    y = ((X[:, 0] > 0)&(X[:, 1] > 0)).astype(int)
    return X, y


def test_end_to_end_lossless_wrapper_exports_and_explains_both_classes():
    X,y=data()
    model=GeneralizedLFIO(4,6000,31).fit(X[:200],y[:200],X[200:300],y[200:300],
                                      feature_names=['length','width'])
    assert verify_relation(model.original_,model.program_)['status']=='proved'
    np.testing.assert_array_equal(model.original_.predict(X[300:]),model.predict(X[300:]))
    for row, prediction in zip(X[300:],model.predict(X[300:])):
        trace=model.explain(row)
        assert trace['prediction']==prediction
        assert trace['used_default']==(prediction!=model.program_.head)
    export=json.loads(json.dumps(model.export()))
    restored=RuleProgram.from_export(export['generalized']['program'])
    np.testing.assert_array_equal(restored.predict(X),model.predict(X))
    assert model.get_rules()['default_class']==1-model.program_.head


def test_calibrated_wrapper_requires_unused_groups_and_preserves_budget():
    X,y=data()
    with pytest.raises(ValueError,match='separate calibration'):
        GeneralizedLFIO(mode='calibrated').fit(X[:200],y[:200],X[200:300],y[200:300])
    with pytest.raises(ValueError,match='disjoint'):
        GeneralizedLFIO(mode='calibrated').fit(X[:200],y[:200],X[200:300],y[200:300],
            X_calibration=X[100:150],y_calibration=y[100:150])
    model=GeneralizedLFIO(4,6000,31,mode='calibrated').fit(
        X[:200],y[:200],X[200:300],y[200:300],X_calibration=X[300:],y_calibration=y[300:])
    assert model.program_.complexity[0]<=4
    assert model.generalizer_.audit_['calibration_used']


def test_wrapper_rejects_duplicate_selection_groups_and_checks_fit_state():
    X,y=data()
    with pytest.raises(ValueError,match='disjoint'):
        GeneralizedLFIO().fit(X[:200],y[:200],X[:30],y[:30])
    for method,args in [('predict',(X,)),('get_rules',()),('explain',(X[0],)),('export',())]:
        with pytest.raises(ValueError,match='fit must'):
            getattr(GeneralizedLFIO(),method)(*args)
