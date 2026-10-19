"""Frozen October revision: independent native-teacher tasks and action allocation.

Run --prepare-only before starting any model fitting. Test inputs and labels are
only passed to exported predictors after each method's validation selection.
The data, sources, configuration, and official baseline files are hash-locked.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from src.lfio_diagnostic import DiagnosticLFIO, AbsoluteWorkLFIO, balanced_collision
from src.lfio_budgeted import mdl_vocabulary
from src.lfio_followup import apply_vocabulary, numeric_tree_dnf, predict_numeric_tree
from research_baselines.brcg import BooleanRuleCG, literal_frame, export_brcg, predict_brcg
from research_baselines.prospective import (fast_candidates, fast_predict,
    native_neurules_candidates, native_neurules_predict)

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/research/prospective_2026-10-15'
OUT = ROOT/'results/prospective_20261015'
DATASETS = ('rice', 'raisin', 'htru2', 'qsar', 'climate', 'retinopathy')
SEEDS = (6001, 6002, 6003)
TEACHERS = ('rf', 'hgb')
NOISE = (0., .2)
BUDGETS = (8, 16, 32)
METHODS = ('DiagnosticLFIO', 'MedianRestarts', 'MDLRestarts', 'MedianRepair',
           'NativeTree', 'BRCG-light', 'FastSRS', 'NeuRulesNative')
TOTAL_SCORES = 1200000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_data(name):
    folder = DATA/name
    if name in ('rice', 'raisin', 'retinopathy'):
        source = next(folder.rglob('*.arff'))
        records, _ = arff.loadarff(source)
        frame = pd.DataFrame(records)
        target = frame.iloc[:, -1]
        positive = {'rice': b'Cammeo', 'raisin': b'Besni', 'retinopathy': b'1'}[name]
        y = (target == positive).to_numpy(int)
        features = frame.iloc[:, :-1]
    elif name == 'htru2':
        frame = pd.read_csv(folder/'HTRU_2.csv', header=None)
        y, features = frame.iloc[:, -1].to_numpy(int), frame.iloc[:, :-1]
    elif name == 'qsar':
        frame = pd.read_csv(folder/'biodeg.csv', sep=';', header=None)
        y, features = (frame.iloc[:, -1] == 'RB').to_numpy(int), frame.iloc[:, :-1]
    elif name == 'climate':
        frame = pd.read_csv(folder/'pop_failures.dat', sep=r'\s+')
        y = frame.outcome.to_numpy(int)
        features = frame.drop(columns=['Study', 'Run', 'outcome'])
    else:
        raise ValueError(name)
    raw = features.to_numpy(float)
    assert np.isfinite(raw).all() and set(np.unique(y)) == {0, 1}
    return raw, y, list(map(str, features.columns))


def lfio_export(model, vocabulary):
    return dict(representation='numeric threshold DNF', vocabulary=vocabulary,
        head=model.head_, default_class=1-model.head_,
        rules=[[[int(f), int(v)] for f, v in rule.literals] for rule in model._selected],
        total_literals=sum(len(r.literals) for r in model._selected),
        model=model.export())


def predict_export(method, export, raw):
    if method == 'NativeTree':
        return predict_numeric_tree(export, raw)
    if method == 'FastSRS':
        return fast_predict(export, raw)
    if method == 'NeuRulesNative':
        return native_neurules_predict(export, raw)
    return predict_brcg(export, apply_vocabulary(raw, export['vocabulary']))


def size(export, method):
    return int(export['total_literals']), (export['rules'] if method == 'NeuRulesNative'
                                           else len(export['rules']))


def score(y, prediction):
    return float(balanced_accuracy_score(y, prediction)) if len(np.unique(y)) == 2 else float(np.mean(y == prediction))


def run_method(method, raw, y, val, vy, seed, folder):
    started, cpu = time.perf_counter(), time.process_time()
    candidates = []

    def record(export, config, fit_budget=None):
        conditions, rules = size(export, method)
        prediction = predict_export(method, export, val)
        candidates.append(dict(symbolic=export, config=config, fit_budget=fit_budget,
            total_literals=conditions, rules=rules,
            validation_balanced_fidelity=score(vy, prediction)))

    median = DiagnosticLFIO.median_vocabulary(raw)
    rich, audit = mdl_vocabulary(raw, y)
    if method == 'DiagnosticLFIO':
        for budget in BUDGETS:
            model = DiagnosticLFIO(budget, TOTAL_SCORES, seed).fit(raw, y, val, vy)
            export = lfio_export(model.model_, model.vocabulary_)
            export['controller'] = model.audit_
            record(export, model.model_.selected_config_, budget)
    elif method in ('MedianRestarts', 'MDLRestarts', 'MedianRepair'):
        vocabulary = rich if method == 'MDLRestarts' else median
        X, V = apply_vocabulary(raw, vocabulary), apply_vocabulary(val, vocabulary)
        for budget in BUDGETS:
            model = AbsoluteWorkLFIO(depths=(2,), penalties=(0., .05, .2),
                polarities=(0, 1), literal_budget=budget, candidate_budget=75000,
                pure_start=False, allocate_repair=method == 'MedianRepair',
                max_refinement_steps=8, random_state=seed,
                cell_score_budget=TOTAL_SCORES//6).fit(X, y, V, vy)
            export = lfio_export(model, vocabulary)
            record(export, model.selected_config_, budget)
    elif method == 'NativeTree':
        for leaves in (2, 3, 4, 6, 8, 16):
            model = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=3,
                class_weight='balanced', random_state=seed).fit(raw, y)
            export = numeric_tree_dnf(model)
            export['total_literals'] = sum(map(len, export['rules']))
            np.testing.assert_array_equal(model.predict(val), predict_numeric_tree(export, val))
            record(export, dict(max_leaf_nodes=leaves, min_samples_leaf=3, class_weight='balanced'))
    elif method == 'BRCG-light':
        X, V = literal_frame(apply_vocabulary(raw, rich)), literal_frame(apply_vocabulary(val, rich))
        for penalty in (.0005, .005, .05):
            for head in (0, 1):
                config = dict(lambda0=penalty, lambda1=penalty, CNF=head == 0,
                    iterMax=25, timeMax=10, K=10, D=3, B=5, eps=1e-6,
                    solver='CLARABEL', silent=True)
                model = BooleanRuleCG(**config)
                model.fit(X, y)
                export = export_brcg(model)
                export['vocabulary'] = rich
                np.testing.assert_array_equal(model.predict(V), predict_export(method, export, val))
                record(export, config)
    elif method == 'FastSRS':
        for export, config in fast_candidates(raw, y, rich):
            record(export, config)
    elif method == 'NeuRulesNative':
        for export, config in native_neurules_candidates(raw, y, val, seed):
            record(export, config)
    else:
        raise ValueError(method)
    # No test data have been supplied to this function.
    selections = []
    for budget in BUDGETS:
        feasible = [i for i, c in enumerate(candidates) if c['total_literals'] <= budget
                    and c['fit_budget'] in (None, budget)]
        winner = max(feasible, key=lambda i: (candidates[i]['validation_balanced_fidelity'],
            -candidates[i]['total_literals'], -candidates[i]['rules'], -i)) if feasible else None
        selections.append(dict(budget=budget, feasible=winner is not None, candidate_index=winner))
    result = dict(candidates=candidates, selections=selections,
        runtime_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
    write_json(folder/f'{method}_fit.json', result)
    return result


def one(task):
    name, seed, teacher_kind, noise, selected_methods = task
    folder = OUT/f'{name}_{seed}_{teacher_kind}_noise{round(noise*100):02d}'
    folder.mkdir(parents=True, exist_ok=True)
    raw, labels, names = load_data(name)
    _, groups = np.unique(raw, axis=0, return_inverse=True)
    folds = [b for _, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(raw, labels, groups)]
    te, va, tr = folds[0], folds[1], np.concatenate(folds[2:])
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(groups[a]) & set(groups[b])
    split_file = folder/'split.npz'
    if split_file.exists():
        split = np.load(split_file)
        for key, value in (('train', tr), ('validation', va), ('test', te)):
            np.testing.assert_array_equal(split[key], value)
        y, vy, ty = (split['teacher_'+key] for key in ('train', 'validation', 'test'))
    else:
        observed = labels[tr].copy()
        observed[np.random.default_rng(seed+13579).random(len(tr)) < noise] ^= 1
        if teacher_kind == 'rf':
            teacher = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                class_weight='balanced_subsample', random_state=seed, n_jobs=1)
        else:
            teacher = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31,
                l2_regularization=1., class_weight='balanced', random_state=seed)
        teacher.fit(raw[tr], observed)
        y, vy, ty = [teacher.predict(raw[a]).astype(int) for a in (tr, va, te)]
        np.savez_compressed(split_file, train=tr, validation=va, test=te,
            observed_train=observed, teacher_train=y, teacher_validation=vy, teacher_test=ty)
        protocol = dict(dataset=name, seed=seed, teacher=teacher_kind, noise=noise,
            samples=len(raw), features=raw.shape[1], feature_names=names,
            source_data_sha256=fingerprint(raw, labels), split_sha256=fingerprint(tr, va, te),
            teacher_predictions_sha256=fingerprint(y, vy, ty),
            teacher_accuracy=float(np.mean(ty == labels[te])),
            teacher_balanced_accuracy=score(labels[te], ty),
            train_class_counts=np.bincount(y, minlength=2).tolist(),
            validation_class_counts=np.bincount(vy, minlength=2).tolist(),
            test_class_counts=np.bincount(ty, minlength=2).tolist(),
            median_collision=balanced_collision(apply_vocabulary(raw[tr], DiagnosticLFIO.median_vocabulary(raw[tr])), y))
        write_json(folder/'protocol.json', protocol)
    for method in selected_methods:
        result_file = folder/f'{method}.json'
        if result_file.exists():
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit_file = folder/f'{method}_fit.json'
            record = (json.loads(fit_file.read_text()) if fit_file.exists()
                      else run_method(method, raw[tr], y, raw[va], vy, seed, folder))
        # Test inference starts only after validation choices were persisted.
        rows, predictions = [], {}
        for selected in record['selections']:
            row = dict(dataset=name, seed=seed, teacher=teacher_kind, noise=noise, method=method, **selected)
            if selected['feasible']:
                candidate = record['candidates'][selected['candidate_index']]
                symbolic = candidate['symbolic']
                pred = predict_export(method, symbolic, raw[te])
                vp = predict_export(method, symbolic, raw[va])
                predictions['test_'+str(selected['budget'])] = pred
                predictions['validation_'+str(selected['budget'])] = vp
                row.update(fidelity=float(np.mean(pred == ty)), balanced_fidelity=score(ty, pred),
                    accuracy=float(np.mean(pred == labels[te])), balanced_accuracy=score(labels[te], pred),
                    total_literals=candidate['total_literals'], rules=candidate['rules'],
                    validation_balanced_fidelity=candidate['validation_balanced_fidelity'])
                if 'controller' in symbolic:
                    row.update(action=symbolic['controller']['action'],
                        capacity_witness=symbolic['controller'].get('capacity_witness', False),
                        scores_used=symbolic['controller']['scores_used'])
                elif method in ('MedianRestarts', 'MDLRestarts', 'MedianRepair'):
                    row['scores_used'] = symbolic['model']['search_work'].get('scored_neighbor_evaluations', 0)
            rows.append(row)
        np.savez_compressed(folder/f'{method}_predictions.npz', **predictions)
        record.update(rows=rows, warnings=sorted(set(str(w.message) for w in caught)))
        write_json(result_file, record)
        print(folder.name, method, 'finished', flush=True)
    return folder.name


def signature():
    sources = ['run_prospective_revision.py', 'src/lfio_diagnostic.py', 'src/lfio_research.py',
        'src/lfio_budgeted.py', 'src/lfio_followup.py', 'research_baselines/prospective.py',
        'research_baselines/brcg.py', 'research_baselines/adapters.py']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'research_baselines/vendor_brcg').glob('*.py')]
    upstream = list((ROOT/'.tools/baselines/fastsrs/src/fastsrs').glob('*.py'))
    upstream += [ROOT/'.tools/baselines/neurules'/n for n in ('neurules.py', 'configs.py', 'utils.py', 'training.py')]
    data = {name: fingerprint(*load_data(name)[:2]) for name in DATASETS}
    return dict(datasets=list(DATASETS), seeds=list(SEEDS), teachers=list(TEACHERS), noise=list(NOISE),
        budgets=list(BUDGETS), methods=list(METHODS), total_score_budget=TOTAL_SCORES,
        sources={name: digest(ROOT/name) for name in sources},
        upstream={str(p.relative_to(ROOT)): digest(p) for p in upstream}, data_sha256=data,
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'pyfim')},
        primary='clean RF teacher, budget 8: balanced fidelity DiagnosticLFIO versus MedianRestarts',
        secondary='MDLRestarts comparison; both teachers, both noise settings, budgets 16/32, runtime, native baselines',
        uncertainty='dataset-cluster bootstrap; six independent dataset units, seeds are within-dataset replicates',
        limitations='same-session prospectively frozen extension; not an external preregistration; six datasets include related rice/raisin domains')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT/'manifest.json'
    current = signature()
    if frozen.exists():
        assert json.loads(frozen.read_text())['signature'] == current, 'Frozen protocol/source mismatch'
    else:
        write_json(frozen, dict(frozen_at=datetime.now(timezone.utc).isoformat(), signature=current))
    if args.prepare_only:
        print('Protocol frozen before fitting:', frozen)
        return
    tasks = [(name, seed, teacher, noise, args.methods) for name in args.datasets
             for seed in args.seeds for teacher in TEACHERS for noise in NOISE]
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                print('COMPLETE', future.result(), flush=True)
            except Exception:
                error = dict(task=futures[future], traceback=traceback.format_exc())
                errors.append(error); print(error['traceback'], flush=True)
    write_json(OUT/('completion_'+'_'.join(args.methods)+'.json'),
        dict(expected_settings=len(tasks), completed_settings=len(tasks)-len(errors), errors=errors))
    if errors:
        raise SystemExit(f'{len(errors)} settings failed; failures retained for audit')


if __name__ == '__main__':
    main()
