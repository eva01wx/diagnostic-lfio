"""Frozen new-split study of explicit surrogate-rule generalization."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from run_prospective_revision import (DATASETS, TEACHERS, NOISE, BUDGETS, TOTAL_SCORES,
    load_data, signature as parent_signature, lfio_export, score, digest, write_json)
from src.lfio_diagnostic import AbsoluteWorkLFIO, DiagnosticLFIO
from src.lfio_budgeted import mdl_vocabulary
from src.lfio_followup import apply_vocabulary
from src.rule_generalization import (RuleGeneralizer, RuleProgram, boolean_simplify,
    verify_relation)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/rule_generalization_20261016'
SEEDS = (7001, 7002, 7003)
BASES = ('DiagnosticLFIO', 'MedianRestarts', 'MDLRestarts')
VARIANTS = ('Original', 'BooleanExact', 'SemanticExact', 'TrainGeneralize',
            'EmpiricalGeneralize', 'BoundedGeneralize')
EPSILON, DELTA = .02, .05


def signature():
    return dict(parent=parent_signature(), seeds=list(SEEDS), bases=list(BASES),
        variants=list(VARIANTS), epsilon=EPSILON, delta=DELTA,
        sources={name: digest(ROOT/name) for name in ('src/rule_generalization.py',
            'run_rule_generalization.py', 'docs/research/RULE_GENERALIZATION_PROTOCOL_2026-10-16.md')},
        split='60% fitting, 10% selection, 10% calibration, 20% test; duplicate groups disjoint',
        primary='dataset-equal relative total-condition reduction: SemanticExact versus Original',
        secondary='approximate compression, balanced fidelity, conditional harm, feature/threshold counts, short-circuit checks',
        interpretation='new splits of previously studied datasets; exploratory extension, not new external data',
        calibration_scope='one-sided binomial bounds assume original learner/path independent of iid class-conditional calibration; grouped benchmark sampling does not automatically establish population coverage')


def make_base(method, budget, raw, y, val, vy, seed):
    if method == 'DiagnosticLFIO':
        model = DiagnosticLFIO(budget, TOTAL_SCORES, seed).fit(raw, y, val, vy)
        export = lfio_export(model.model_, model.vocabulary_)
        export['controller'] = model.audit_
    else:
        vocabulary = (mdl_vocabulary(raw, y)[0] if method == 'MDLRestarts'
                      else DiagnosticLFIO.median_vocabulary(raw))
        model = AbsoluteWorkLFIO(depths=(2,), penalties=(0., .05, .2),
            polarities=(0, 1), literal_budget=budget, candidate_budget=75000,
            pure_start=False, allocate_repair=False, random_state=seed,
            cell_score_budget=TOTAL_SCORES//6).fit(
                apply_vocabulary(raw, vocabulary), y, apply_vocabulary(val, vocabulary), vy)
        export = lfio_export(model, vocabulary)
    return export


def explanation_checks(program, X):
    """Canonical rule-order short-circuit cost, not a human comprehension score."""
    remaining = np.ones(len(X), dtype=bool)
    checks = np.zeros(len(X), dtype=int)
    for rule in program.rules:
        hit = remaining.copy()
        for atom in rule:
            checks[hit] += 1
            values = X[:, atom.feature] > atom.threshold
            hit &= values if atom.greater else ~values
        remaining &= ~hit
    default = program.predict(X) != program.head
    return dict(mean_checks=float(checks.mean()),
        default_mean_checks=float(checks[default].mean()) if default.any() else None)


def fit_generalizations(original, X, y, calibration, cy):
    started, cpu = time.perf_counter(), time.process_time()
    model = RuleGeneralizer(mode='calibrated', epsilon=EPSILON, delta=DELTA).fit(original, X, y)
    empirical = copy.deepcopy(model)
    empirical.confidence = 'empirical'; empirical.audit_['confidence'] = 'empirical'
    empirical.calibrate(calibration, cy)
    model.calibrate(calibration, cy)
    boolean = boolean_simplify(original)
    proof = verify_relation(original, boolean, max_nodes=500000)
    assert proof['status'] == 'proved'
    exports = dict(Original=original.export(), BooleanExact=boolean.export(),
        SemanticExact=model.exact_.export(), TrainGeneralize=model.path_[-1].export(),
        EmpiricalGeneralize=empirical.program_.export(), BoundedGeneralize=model.program_.export())
    return dict(programs=exports, bounded=model.export(), empirical=empirical.audit_,
        boolean_proof=proof, runtime_seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu)


def one(task):
    name, seed, teacher_kind, noise = task
    folder = OUT/f'{name}_{seed}_{teacher_kind}_noise{round(noise*100):02d}'
    folder.mkdir(parents=True, exist_ok=True)
    raw, labels, names = load_data(name)
    _, groups = np.unique(raw, axis=0, return_inverse=True)
    folds = [b for _, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(raw, labels, groups)]
    te, validation, tr = folds[0], folds[1], np.concatenate(folds[2:])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        split_file = folder/'split.npz'
        if split_file.exists():
            split = np.load(split_file)
            np.testing.assert_array_equal(split['train'], tr)
            np.testing.assert_array_equal(split['test'], te)
            y, vy, ty = (split[k] for k in ('teacher_train', 'teacher_validation', 'teacher_test'))
            selection, calibration = split['selection'], split['calibration']
        else:
            observed = labels[tr].copy()
            observed[np.random.default_rng(seed+13579).random(len(tr)) < noise] ^= 1
            teacher = (RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                class_weight='balanced_subsample', random_state=seed, n_jobs=1) if teacher_kind == 'rf'
                else HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31,
                    l2_regularization=1., class_weight='balanced', random_state=seed))
            teacher.fit(raw[tr], observed)
            y, vy, ty = [teacher.predict(raw[a]).astype(int) for a in (tr, validation, te)]
            a, b = next(StratifiedGroupKFold(2, shuffle=True, random_state=seed+31)
                        .split(raw[validation], vy, groups[validation]))
            selection, calibration = validation[a], validation[b]
            np.savez_compressed(split_file, train=tr, validation=validation, selection=selection,
                calibration=calibration, test=te, observed_train=observed,
                teacher_train=y, teacher_validation=vy, teacher_selection=vy[a],
                teacher_calibration=vy[b], teacher_test=ty)
            write_json(folder/'protocol.json',dict(dataset=name, seed=seed, teacher=teacher_kind,
                noise=noise, feature_names=names, teacher_balanced_accuracy=score(labels[te], ty),
                calibration_scope='not supplied to original learner or grow-only generalization path'))
        split = np.load(split_file)
        sy, cy = split['teacher_selection'], split['teacher_calibration']
        for a, b in itertools_pairs((tr, selection, calibration, te)):
            assert not set(groups[a]) & set(groups[b])
        for method in BASES:
            for budget in BUDGETS:
                result_file = folder/f'{method}_B{budget}.json'
                if result_file.exists():
                    continue
                fit_file = folder/f'{method}_B{budget}_fit.json'
                if fit_file.exists():
                    fit = json.loads(fit_file.read_text())
                else:
                    started, cpu = time.perf_counter(), time.process_time()
                    export = make_base(method, budget, raw[tr], y, raw[selection], sy, seed)
                    original = RuleProgram.from_lfio(export, raw.shape[1])
                    fit = dict(dataset=name, seed=seed, teacher=teacher_kind, noise=noise,
                        base=method, budget=budget, source_export=export,
                        base_seconds=time.perf_counter()-started, base_cpu=time.process_time()-cpu,
                        **fit_generalizations(original, raw[tr], y, raw[calibration], cy))
                    # No test inputs are passed to either fitting stage.
                    write_json(fit_file, fit)
                # All programs and calibration choices exist on disk before test inference.
                rows, predictions = [], {}
                programs = {key: RuleProgram.from_export(value) for key, value in fit['programs'].items()}
                before = programs['Original'].predict(raw[te])
                for variant in VARIANTS:
                    program = programs[variant]; prediction = program.predict(raw[te])
                    if variant in ('BooleanExact', 'SemanticExact'):
                        np.testing.assert_array_equal(prediction, before)
                    opposite = ty == 1-program.head
                    harm = int(((before == ty)&(prediction != ty)&opposite).sum())
                    size = program.complexity
                    rows.append(dict(dataset=name, seed=seed, teacher=teacher_kind, noise=noise,
                        base=method, budget=budget, variant=variant, total_literals=size[0], rules=size[1],
                        features=size[2], thresholds=size[3], fidelity=float(np.mean(prediction == ty)),
                        macro_fidelity=score(ty,prediction), teacher_classes=int(len(np.unique(ty))),
                        harmful=harm, opposite_class_samples=int(opposite.sum()),
                        **explanation_checks(program, raw[te])))
                    predictions[variant] = prediction
                np.savez_compressed(folder/f'{method}_B{budget}_predictions.npz', **predictions)
                write_json(result_file,dict(fit_sha256=digest(fit_file), rows=rows))
        write_json(folder/'warnings.json', sorted(set(str(w.message) for w in caught)))
    print(folder.name, 'finished', flush=True)
    return folder.name


def itertools_pairs(values):
    for i, first in enumerate(values):
        for second in values[i+1:]:
            yield first, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT/'manifest.json'
    current = signature()
    if manifest.exists():
        assert json.loads(manifest.read_text())['signature'] == current, 'Frozen source/configuration mismatch'
    else:
        write_json(manifest,dict(frozen_at=datetime.now(timezone.utc).isoformat(), signature=current))
    if args.prepare_only:
        print(manifest); return
    tasks = [(n,s,t,z) for n in DATASETS for s in SEEDS for t in TEACHERS for z in NOISE]
    finished, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                finished.append(future.result())
            except Exception:
                failures.append(dict(task=list(futures[future]),traceback=traceback.format_exc()))
    records = list(OUT.glob('*/*_B[0-9]*.json'))
    records = [p for p in records if not p.stem.endswith('_fit')]
    write_json(OUT/'completion.json',dict(expected_settings=len(tasks), completed_settings=len(finished),
        expected_base_budget_fits=648, completed_base_budget_fits=len(records),
        expected_variant_rows=3888, failures=failures))
    if failures or len(records) != 648:
        raise SystemExit('Study incomplete; failures are retained in completion.json')


if __name__ == '__main__':
    main()
