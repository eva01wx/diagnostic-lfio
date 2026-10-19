"""Auditable native-input baselines for the October prospective experiment."""
from contextlib import redirect_stdout
from collections import defaultdict
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
from joblib import Parallel

ROOT = Path(__file__).resolve().parents[1]
FASTSRS_COMMIT = '594904e79ec5916be1635f3495d017d76568bdc2'


def fast_frame(raw, vocabulary):
    columns, names = [], []
    for i, (feature, threshold) in enumerate(vocabulary):
        for value, sign in ((0, 'le'), (1, 'gt')):
            columns.append((raw[:, feature] > threshold) == value)
            names.append(f'x{feature}_{sign}{i}')
    return np.column_stack(columns).astype(int), names


def fast_export(rules, head, vocabulary, names):
    conditions = []
    for rule in rules:
        conditions.append([[[int(index)//2, int(index)%2] for index in indices]
                           for _, indices in sorted(rule.items())])
    return dict(representation='multi-value DNF', head=head, default_class=1-head,
                rules=conditions, vocabulary=vocabulary,
                total_literals=sum(len(alternatives) for rule in conditions for alternatives in rule),
                native_conditions=sum(map(len, conditions)), column_names=names,
                implementation='official FastSRS ORS', commit=FASTSRS_COMMIT)


def fast_predict(export, raw):
    vocabulary = export['vocabulary']
    bits = np.column_stack([raw[:, f] > t for f, t in vocabulary])
    covered = np.zeros(len(raw), bool)
    for rule in export['rules']:
        hit = np.ones(len(raw), bool)
        for alternatives in rule:
            hit &= np.logical_or.reduce([bits[:, j] == v for j, v in alternatives])
        covered |= hit
    return np.where(covered, export['head'], export['default_class'])


def fast_candidates(X, y, vocabulary):
    sys.path.insert(0, str(ROOT/'.tools/baselines/fastsrs/src'))
    import fastsrs.optimal as upstream
    from fastsrs.util import predict_MRS
    # Execution-only single-CPU override. Official screening and SA are unchanged.
    upstream.multiprocessing = SimpleNamespace(cpu_count=lambda: 1)
    upstream.Parallel = lambda **kwargs: Parallel(n_jobs=1)
    matrix, names = fast_frame(X, vocabulary)
    for penalty in (.0005, .005, .05):
        for head in (0, 1):
            target = (y == head).astype(int)
            model = upstream.ORS(matrix, target, names, 'fpgrowth')
            model.set_parameters(c1=penalty, c2=penalty, c3=penalty/3)
            model.set_fixed_bounds()
            with redirect_stdout(io.StringIO()):
                model.generate_rules(5, 3, 2000, method='fpgrowth', criteria='precision')
                # The public init argument avoids upstream random initialization
                # sampling seven rules from a candidate pool smaller than seven.
                init = []
                if model.rules:
                    first = model.rules[0]
                    initial = defaultdict(list)
                    for name in first:
                        initial.setdefault(name.split('_')[0], []).append(model.colindices[name])
                    init = [initial]
                rules, _ = model.train(2000, .3, False, init=init)
            export = fast_export(rules, head, vocabulary, names)
            native = np.where(predict_MRS(rules, matrix), head, 1-head)
            np.testing.assert_array_equal(native, fast_predict(export, X))
            yield export, dict(penalty=penalty, head=head, iterations=2000,
                mined_max_length=3, premined_limit=2000, mining_support_percent=5,
                initialization='first mined rule via official init argument',
                threading='one CPU', source_rule_count=len(model.rules))


def native_neurules_export(hard, config, mean, scale):
    lower = hard.cut_points[:, 0, :].detach().numpy()
    upper = hard.cut_points[:, 1, :].detach().numpy()
    weights = hard.and_weights.detach().numpy()
    classes = hard.rule_weights.detach().numpy().argmax(axis=1)
    rules = []
    for ri in hard.index_order.tolist():
        selected = np.flatnonzero(weights[ri] > 0)
        if not len(selected):
            continue  # Official HardRuleList skips all-disabled rules.
        rules.append(dict(conditions=[[int(fi), float(lower[fi, ri]), float(upper[fi, ri])]
                                      for fi in selected], head=int(classes[ri])))
    return dict(representation='ordered native interval list', ordered_rules=rules,
        default_class=0, rules=len(rules), total_literals=2*sum(len(r['conditions']) for r in rules),
        mean=mean.tolist(), scale=scale.tolist(), config=config,
        condition_counting='each strict lower and upper threshold counts once')


def native_neurules_predict(export, raw):
    # Match the official PyTorch float32 input and cut-point semantics exactly.
    X = ((raw-np.asarray(export['mean']))/np.asarray(export['scale'])).astype(np.float32)
    prediction, covered = np.zeros(len(X), int), np.zeros(len(X), bool)
    for rule in export['ordered_rules']:
        hit = np.ones(len(X), bool)
        for f, lower, upper in rule['conditions']:
            hit &= (X[:, f] > np.float32(lower)) & (X[:, f] < np.float32(upper))
        chosen = hit & ~covered
        prediction[chosen] = rule['head']
        covered |= hit
    return prediction


def native_neurules_candidates(X, y, V, seed):
    import torch
    from sklearn.preprocessing import StandardScaler
    from research_baselines.adapters import neurules_modules, seed_all
    torch.set_num_threads(1)
    models, configs, helpers = neurules_modules()
    scaler = StandardScaler().fit(X)
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    limits = helpers.get_data_limits(Xt)
    for count in (2, 4, 8, 10, 20, 30):
        seed_all(seed)
        config = dict(profile='official CLI on native scaled inputs', n_rules=count,
            n_epochs=500, lr=.01, batch_size=2048, min_support=.2, max_support=.9,
            lambd=.4, init='const', predicate_temperature=.2, selector_temperature=1.,
            optimizer_steps_per_batch=2)
        rc = configs.Rule_Config(X.shape[1], 2, count, predicate_temperature=.2,
                                selector_temperature=1., init='const')
        model = models.NeuralRuleList(rc, limits)
        # Keep upstream's continuous default; do not force Boolean discretization.
        tc = configs.Train_Config(seed=seed, n_epochs=500, lr=.01, batch_size=2048,
            min_support=.2, max_support=.9, lambd=.4, device='cpu')
        predicate = helpers.Temperature_Scheduler(250, rc.schedule_predicate_temperature)
        selector = helpers.Temperature_Scheduler(250, rc.schedule_selector_temperature)
        helpers.train_model(model, Xt, yt, torch.nn.CrossEntropyLoss(), tc,
            temperature_selector_schedule=selector, temperature_predicate_schedule=predicate,
            disable_tqdm=True)
        hard = model.eval().to_hard_rule_list()
        export = native_neurules_export(hard, config, scaler.mean_, scaler.scale_)
        with torch.no_grad():
            check = V[:128]
            native = hard(torch.tensor(scaler.transform(check), dtype=torch.float32)).numpy().argmax(axis=1)
        np.testing.assert_array_equal(native, native_neurules_predict(export, check))
        config['official_export_check_samples'] = len(check)
        yield export, config
