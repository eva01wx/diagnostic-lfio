"""Run official TT-Sparse and NeuRules code on a shared Boolean protocol.

Upstream source is never edited. Only data access, split plumbing and vectorized
hard-rule evaluation are adapted. Downloaded repositories remain in .tools.
"""
import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / '.tools/baselines'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_definitions(path, names, namespace):
    """Execute selected upstream definitions, excluding notebook/data side effects."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'Missing upstream definition in {path}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class BooleanEncoder:
    """Identity encoder: the same train-median Boolean features as every baseline."""
    class_names = ['0', '1']

    def __init__(self, dimensions):
        self.columns = [f'B{i}' for i in range(dimensions)]

    def get_feature_manifest(self):
        # The upstream resolver expects comparison descriptions, including negation.
        return {i: f'{column} == 1' for i, column in enumerate(self.columns)}

    def get_skip_feature_names(self):
        return []

    def transform(self, frame):
        return {'X_ltt': frame[self.columns].to_numpy(np.float32),
                'X_skip': np.empty((len(frame), 0), dtype=np.float32), 'y': None}


@contextmanager
def fixed_tt_split(train_size, validation_size):
    def split(indices, **kwargs):
        assert np.array_equal(indices, np.arange(train_size + validation_size))
        return indices[:train_size], indices[train_size:]
    with patch('tt_sparse.model.train_test_split', split):
        yield


def tt_predict(export, X):
    from tt_sparse.rules import RuleSet, predict_rules
    encoder = BooleanEncoder(X.shape[1])
    rules = RuleSet(**export['rule_set'])
    probabilities = predict_rules(rules, pd.DataFrame(X, columns=encoder.columns), encoder)
    return (probabilities >= .5).astype(int), probabilities


def tt_fit(X, y, V, v, seed, pilot=False, candidate_callback=None, architecture_grid=None):
    from tt_sparse import TTSparseModel, train, prune, explain
    start = time.perf_counter()
    encoder = BooleanEncoder(X.shape[1])
    data = {'X_ltt': np.vstack([X, V]).astype(np.float32), 'X_skip': None,
            'y': np.concatenate([y, v]).astype(np.float32)}
    trace, best, best_key = [], None, None
    grid = [(nodes, min(bits, X.shape[1])) for nodes, bits in
            (architecture_grid if architecture_grid is not None else [(32, 3), (32, 6)])]
    grid = list(dict.fromkeys(grid))
    for nodes, bits in grid:
        seed_all(seed)
        config = dict(num_nodes=nodes, n_bits=bits, epochs=70, patience=25,
                      batch_size=2048, lr=.005, tau=.1, finetune_epochs=30, max_iterations=80)
        model = TTSparseModel(X.shape[1], num_nodes=nodes, n_bits=bits, tau=.1,
                              num_classes=1, task_type='binary', skip_size=0)
        with fixed_tt_split(len(X), len(V)):
            training = train(model, data, epochs=config['epochs'], patience=25,
                             seed=seed, verbose=False)
            pruning = prune(model, data, seed=seed, verbose=False)
        rule_set = explain(model, encoder, training_data=None, use_xor=False)
        active = [r for r in rule_set.rules if r['kind'] == 'ltt']
        literals = sum(len(re.findall(r'\bB\d+\b', r['r'])) for r in active)
        operators = sum(sum(r['r'].count(op) for op in ('&', '|', '^', '~')) for r in active)
        export = {'representation': 'weighted DNF formulas; XOR simplification disabled',
                  'rule_set': vars(rule_set), 'rules': len(active), 'total_literals': literals,
                  'logical_operators': operators, 'features': sorted({int(i) for r in active
                   for i in re.findall(r'\bB(\d+)\b', r['r'])}),
                  'config': config, 'training': training, 'pruning': pruning}
        pred, probabilities = tt_predict(export, V)
        with torch.no_grad():
            neural = torch.sigmoid(model(torch.tensor(V, dtype=torch.float32))).numpy().reshape(-1)
        agreement = float(np.mean(pred == (neural >= .5)))
        error = float(np.max(np.abs(probabilities - neural)))
        if agreement != 1 or error > 1e-4:
            debug = ROOT / 'results/recent_baselines_pilot'
            debug.mkdir(exist_ok=True)
            torch.save(model, debug / 'tt_mismatch.pt')
            np.savez(debug / 'tt_mismatch_inputs.npz', V=V)
            raise AssertionError(f'TT rule/network mismatch: {agreement}, max probability error {error}')
        score = float(balanced_accuracy_score(v, pred))
        if candidate_callback is not None:
            candidate_callback(export, model, score)
        trace.append({'config': config, 'validation_balanced_accuracy': score,
                      'rules': len(active), 'total_literals': literals,
                      'symbolic_network_agreement': agreement, 'max_probability_error': error,
                      'training': training, 'pruning': pruning})
        key = (score, -literals, -len(active))
        if best_key is None or key > best_key:
            best_key, best = key, (export, model)
    return best[0], best[1], trace, time.perf_counter() - start


def neurules_modules():
    repo = REPOS / 'neurules'
    model_module = load_module(repo / 'neurules.py', 'upstream_neurules')
    configs = load_module(repo / 'configs.py', 'upstream_neurules_configs')
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    scope = dict(torch=torch, np=np, Adam=torch.optim.Adam, tqdm=tqdm,
                 DataLoader=DataLoader, TensorDataset=TensorDataset)
    load_definitions(repo / 'utils.py', ['get_data_loader', 'get_data_limits', 'Temperature_Scheduler'], scope)
    load_definitions(repo / 'training.py', ['train_model', 'eval_model', 'F1_Loss'], scope)
    return model_module, configs, SimpleNamespace(**scope)


def neurules_export(hard_model, config):
    """Exact predicates on the Boolean domain; impossible rules are recorded separately."""
    lower = hard_model.cut_points[:, 0, :].detach().numpy()
    upper = hard_model.cut_points[:, 1, :].detach().numpy()
    weights = hard_model.and_weights.detach().numpy()
    classes = hard_model.rule_weights.detach().numpy().argmax(axis=1)
    result, impossible, unconstrained = [], [], []
    for ri in hard_model.index_order.tolist():
        selected = np.flatnonzero(weights[ri] > 0)
        if not len(selected):
            unconstrained.append(ri)  # Official HardRuleList skips all-disabled rules.
            continue
        conditions, reachable = [], True
        for fi in selected:
            zero = lower[fi, ri] < 0 < upper[fi, ri]
            one = lower[fi, ri] < 1 < upper[fi, ri]
            if not zero and not one:
                reachable = False
                break
            if zero != one:
                conditions.append([int(fi), int(one)])
        if reachable:
            result.append({'upstream_index': ri, 'conditions': conditions,
                           'head': int(classes[ri]), 'class_logits': hard_model.rule_weights[ri].tolist()})
        else:
            impossible.append(ri)
    # Retain unreachable-after-tautology rules in export, but count the executable prefix.
    executable = []
    for rule in result:
        executable.append(rule)
        if not rule['conditions']:
            break
    return {'representation': 'ordered Boolean decision list', 'ordered_rules': executable,
            'unreachable_tail': result[len(executable):], 'impossible_rules': impossible,
            'all_disabled_rules': unconstrained, 'default_class': 0,
            'rules': len(executable), 'total_literals': sum(len(r['conditions']) for r in executable),
            'features': sorted({f for r in executable for f, _ in r['conditions']}), 'config': config,
            'raw': {'lower': lower.tolist(), 'upper': upper.tolist(), 'and_weights': weights.tolist(),
                    'index_order': hard_model.index_order.tolist(),
                    'rule_weights': hard_model.rule_weights.tolist()}}


def neurules_predict(export, X):
    pred = np.zeros(len(X), dtype=int)
    covered = np.zeros(len(X), dtype=bool)
    for rule in export['ordered_rules']:
        matches = np.ones(len(X), dtype=bool)
        for feature, value in rule['conditions']:
            matches &= X[:, feature] == value
        selected = matches & ~covered
        pred[selected] = rule['head']
        covered |= matches
    return pred, covered


def neurules_fit(X, y, V, v, seed, pilot=False, candidate_callback=None, rule_counts=(10, 20, 30)):
    start = time.perf_counter()
    models, configs, helpers = neurules_modules()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    Vt = torch.tensor(V, dtype=torch.float32)
    limits = helpers.get_data_limits(Xt)
    trace, best, best_key = [], None, None
    # 500 epochs is the official run_real_world.sh -> run_method.py default.
    for count in rule_counts:
        seed_all(seed)
        config = dict(n_rules=count, n_epochs=500, lr=.002, batch_size=-1,
                      predicate_temperature=.2, selector_temperature=1., min_support=.02,
                      temperature_reduction=5, optimizer_steps_per_batch=2)
        rc = configs.Rule_Config(X.shape[1], 2, count, predicate_temperature=.2, selector_temperature=1.)
        model = models.NeuralRuleList(rc, limits)
        model.discretizer.is_discrete = [True] * X.shape[1]
        tc = configs.Train_Config(seed=seed, n_epochs=500, device='cpu')
        predicate = helpers.Temperature_Scheduler(250, rc.schedule_predicate_temperature)
        selector = helpers.Temperature_Scheduler(250, rc.schedule_selector_temperature)
        helpers.train_model(model, Xt, yt, torch.nn.CrossEntropyLoss(), tc,
                            temperature_selector_schedule=selector,
                            temperature_predicate_schedule=predicate, disable_tqdm=True)
        model.eval()
        hard = model.to_hard_rule_list()
        export = neurules_export(hard, config)
        pred, coverage = neurules_predict(export, V)
        with torch.no_grad():
            official = hard(Vt).numpy().argmax(axis=1)
        agreement = float(np.mean(pred == official))
        if agreement != 1:
            raise AssertionError(f'NeuRules vectorization mismatch: {agreement}')
        score = float(balanced_accuracy_score(v, pred))
        if candidate_callback is not None:
            candidate_callback(export, model, score)
        trace.append({'config': config, 'validation_balanced_accuracy': score,
                      'rules': export['rules'], 'total_literals': export['total_literals'],
                      'official_hard_rule_agreement': agreement, 'coverage': float(coverage.mean())})
        key = score, -export['total_literals'], -export['rules']
        if best_key is None or key > best_key:
            best_key, best = key, (export, model)
    return best[0], best[1], trace, time.perf_counter() - start


def predict(method, export, X):
    return (tt_predict if method == 'TT-Sparse' else neurules_predict)(export, X)[0]


FIT = {'TT-Sparse': tt_fit, 'NeuRules': neurules_fit}
