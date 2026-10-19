"""Replay, audit and summarize every October independent-cohort selection."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lfio_major_revision_mpl')
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.lfio_diagnostic import DiagnosticLFIO, AbsoluteWorkLFIO, balanced_collision
from src.lfio_budgeted import mdl_vocabulary
from src.lfio_followup import apply_vocabulary
from run_prospective_revision import (ROOT, OUT as RUN, DATASETS, SEEDS, TEACHERS, NOISE,
    METHODS, BUDGETS, TOTAL_SCORES, load_data, fingerprint, digest, write_json,
    signature, predict_export, size, score)

OUT = ROOT/'docs/research/major_revision_2026-10-15'
LABELS = dict(DiagnosticLFIO='Diagnostic LFIO', MedianRestarts='Median restarts',
    MDLRestarts='MDL restarts', MedianRepair='Median repair', NativeTree='Native tree',
    **{'BRCG-light': 'BRCG-light'}, FastSRS='FastSRS', NeuRulesNative='NeuRules native')
DATA_LABELS = dict(rice='Rice', raisin='Raisin', htru2='HTRU2', qsar='QSAR',
                   climate='Climate', retinopathy='Retinopathy')


def controller_replay(folder):
    folder = Path(folder)
    protocol = json.loads((folder/'protocol.json').read_text())
    raw, _, _ = load_data(protocol['dataset'])
    split = np.load(folder/'split.npz')
    tr, va = split['train'], split['validation']
    record = json.loads((folder/'DiagnosticLFIO.json').read_text())
    for candidate in record['candidates']:
        symbolic = candidate['symbolic']
        budget = candidate['fit_budget']
        replay = DiagnosticLFIO(budget, TOTAL_SCORES, protocol['seed']).fit(
            raw[tr], split['teacher_train'], raw[va], split['teacher_validation'])
        expected, actual = symbolic['controller'].copy(), replay.audit_.copy()
        expected.pop('fit_seconds'); actual.pop('fit_seconds')
        assert json.loads(json.dumps(actual)) == expected, folder
        np.testing.assert_array_equal(replay.predict(raw[va]), predict_export('DiagnosticLFIO', symbolic, raw[va]))
    return dict(setting=folder.name, replayed_budget_decisions=len(record['candidates']),
        source_sha256=digest(folder/'DiagnosticLFIO.json'), training_only_probes_replayed=True)


def audit(replay_workers):
    manifest = json.loads((RUN/'manifest.json').read_text())
    assert manifest['signature'] == signature(), 'Protocol, data or source hashes changed'
    metrics, times, protocols, hashes = [], [], [], {}
    checked_candidates, saved_predictions = 0, 0
    folders = []
    for name in DATASETS:
        raw, labels, _ = load_data(name)
        _, groups = np.unique(raw, axis=0, return_inverse=True)
        for seed in SEEDS:
            folds = [b for _, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(raw, labels, groups)]
            expected_ids = (np.concatenate(folds[2:]), folds[1], folds[0])
            for teacher in TEACHERS:
                for noise in NOISE:
                    folder = RUN/f'{name}_{seed}_{teacher}_noise{round(noise*100):02d}'
                    folders.append(folder)
                    protocol = json.loads((folder/'protocol.json').read_text())
                    protocols.append(protocol)
                    split = np.load(folder/'split.npz')
                    tr, va, te = (split[k] for k in ('train', 'validation', 'test'))
                    for a, b in zip((tr, va, te), expected_ids):
                        np.testing.assert_array_equal(a, b)
                    for a, b in ((tr, va), (tr, te), (va, te)):
                        assert not set(groups[a]) & set(groups[b])
                    y, vy, ty = (split['teacher_'+k] for k in ('train', 'validation', 'test'))
                    assert fingerprint(raw, labels) == protocol['source_data_sha256']
                    assert fingerprint(tr, va, te) == protocol['split_sha256']
                    assert fingerprint(y, vy, ty) == protocol['teacher_predictions_sha256']
                    assert np.isclose(protocol['teacher_balanced_accuracy'], score(labels[te], ty))
                    vocabularies = dict(median=DiagnosticLFIO.median_vocabulary(raw[tr]),
                                        rich=mdl_vocabulary(raw[tr], y)[0])
                    for method in METHODS:
                        file = folder/f'{method}.json'
                        record = json.loads(file.read_text())
                        saved = np.load(folder/f'{method}_predictions.npz')
                        for candidate in record['candidates']:
                            symbolic = candidate['symbolic']
                            literals, rules = size(symbolic, method)
                            assert literals == candidate['total_literals'] and rules == candidate['rules']
                            assert np.isclose(score(vy, predict_export(method, symbolic, raw[va])),
                                              candidate['validation_balanced_fidelity'])
                            if method in ('MedianRestarts', 'MDLRestarts', 'MedianRepair'):
                                expected = vocabularies['rich' if method == 'MDLRestarts' else 'median']
                                assert [list(v) for v in expected] == symbolic['vocabulary']
                            if 'model' in symbolic:
                                model = symbolic['model']
                                assert np.all(np.diff(model['objective_path']) <= 1e-8)
                                used = model['search_work'].get('scored_neighbor_evaluations', 0)
                                assert used <= TOTAL_SCORES
                                assert sum(c['optimization']['scores_used'] for c in model['selection_trace']) == used
                                if method == 'DiagnosticLFIO':
                                    c = symbolic['controller']
                                    assert c['scores_used'] <= TOTAL_SCORES
                                    assert used+sum(p['scores_used'] for p in c['probes'].values()) == c['scores_used']
                                    expected = vocabularies['rich' if c['action'] == 'vocabulary' else 'median']
                                    assert [list(v) for v in expected] == symbolic['vocabulary']
                                else:
                                    assert all(c['optimization']['cell_score_budget'] == TOTAL_SCORES//6
                                               for c in model['selection_trace'])
                            checked_candidates += 1
                        for selected, row in zip(record['selections'], record['rows']):
                            budget = selected['budget']
                            feasible = [i for i, c in enumerate(record['candidates'])
                                if c['total_literals'] <= budget and c['fit_budget'] in (None, budget)]
                            winner = max(feasible, key=lambda i: (
                                record['candidates'][i]['validation_balanced_fidelity'],
                                -record['candidates'][i]['total_literals'],
                                -record['candidates'][i]['rules'], -i)) if feasible else None
                            assert selected['candidate_index'] == winner == row['candidate_index']
                            assert selected['feasible'] == (winner is not None)
                            if winner is not None:
                                candidate = record['candidates'][winner]
                                pred = predict_export(method, candidate['symbolic'], raw[te])
                                vp = predict_export(method, candidate['symbolic'], raw[va])
                                np.testing.assert_array_equal(pred, saved['test_'+str(budget)])
                                np.testing.assert_array_equal(vp, saved['validation_'+str(budget)])
                                assert np.isclose(row['balanced_fidelity'], score(ty, pred))
                                assert np.isclose(row['fidelity'], np.mean(ty == pred))
                                assert np.isclose(row['balanced_accuracy'], score(labels[te], pred))
                                assert candidate['total_literals'] <= budget
                                saved_predictions += 2
                            row = dict(row, observed_teacher_classes=len(np.unique(ty)))
                            row['two_class_balanced_fidelity'] = (row.get('balanced_fidelity')
                                if len(np.unique(ty)) == 2 else None)
                            metrics.append(row)
                        times.append(dict(dataset=name, seed=seed, teacher=teacher, noise=noise,
                            method=method, wall=record['runtime_seconds'], cpu=record['cpu_seconds']))
                    for f in folder.iterdir():
                        if f.suffix in ('.json', '.npz'):
                            hashes[str(f.relative_to(ROOT))] = digest(f)
    replay_file = OUT/'controller_replay.json'
    if replay_file.exists():
        replays = json.loads(replay_file.read_text())
        assert len(replays) == len(folders)
        for row in replays:
            assert row['source_sha256'] == digest(RUN/row['setting']/'DiagnosticLFIO.json')
    else:
        replays = []
        with ProcessPoolExecutor(max_workers=replay_workers) as pool:
            for future in as_completed([pool.submit(controller_replay, str(f)) for f in folders]):
                replays.append(future.result())
        write_json(replay_file, sorted(replays, key=lambda r: r['setting']))
    certificate = json.loads((RUN/'certificates/completion.json').read_text())
    assert certificate['completed'] == certificate['expected'] == 12
    cm = json.loads((RUN/'certificates/manifest.json').read_text())
    assert cm['parent_manifest_sha256'] == digest(RUN/'manifest.json')
    assert cm['sources']['src/lfio_exact.py'] == digest(ROOT/'src/lfio_exact.py')
    if cm['sources']['certify_real_revision.py'] != digest(ROOT/'certify_real_revision.py'):
        amendment = json.loads((RUN/'certificates/execution_amendment.json').read_text())
        assert amendment['old_source_sha256'] == cm['sources']['certify_real_revision.py']
        assert amendment['new_source_sha256'] == digest(ROOT/'certify_real_revision.py')
        assert amendment['mathematical_protocol_changed'] is False
    replayed_incumbents = 0
    for row in certificate['rows']:
        assert row['bounds']['collision'] <= row['bounds']['optimum_lower']+1e-6
        assert row['bounds']['optimum_lower'] <= row['bounds']['optimum_upper']+1e-6
        assert row['bounds']['optimum_upper'] <= row['bounds']['heuristic_error']+1e-6
        assert row['selected_source_sha256'] == digest(RUN/f"{row['dataset']}_6001_rf_noise00"/f"{row['method']}.json")
        folder = RUN/f"{row['dataset']}_6001_rf_noise00"
        raw, _, _ = load_data(row['dataset'])
        split = np.load(folder/'split.npz')
        result = json.loads((folder/f"{row['method']}.json").read_text())
        index = next(s['candidate_index'] for s in result['selections'] if s['budget'] == 8)
        symbolic = result['candidates'][index]['symbolic']
        X = apply_vocabulary(raw[split['train']], symbolic['vocabulary'])
        y = split['teacher_train']
        assert fingerprint(X, y) == row['training_sha256']
        assert np.isclose(balanced_collision(X, y)['error_floor'], row['bounds']['collision'])
        heuristic = predict_export(row['method'], symbolic, raw[split['train']])
        assert np.isclose(1-score(y, heuristic), row['bounds']['heuristic_error'])
        learner = AbsoluteWorkLFIO(depths=(2,), penalties=(0.,), literal_budget=8,
            candidate_budget=75000, min_support=3)
        pool = learner._candidates(X)
        assert len(pool) == row['candidates']
        for solve in row['solves']:
            assert solve['optimal'] == (solve['status'] == 0)
            if solve['objective'] is not None:
                rules = [pool[i] for i in solve['selected_indices']]
                assert sum(len(r.literals) for r in rules) <= 8 and len(rules) <= 32
                prediction = learner._predict_rules(X, rules, solve['head'])
                assert np.isclose((1-score(y, prediction))*len(y), solve['objective'])
                if solve['lower_bound'] is not None:
                    assert solve['lower_bound'] <= solve['objective']+1e-5
                replayed_incumbents += 1
    for f in (RUN/'certificates').glob('*.json'):
        hashes[str(f.relative_to(ROOT))] = digest(f)
    checks = dict(settings=len(folders), method_grids=len(times), budget_rows=len(metrics),
        candidate_exports=checked_candidates, replayed_prediction_vectors=saved_predictions,
        internal_budget_decisions_replayed=sum(r['replayed_budget_decisions'] for r in replays),
        source_and_data_hashes_verified=True, raw_groups_disjoint=True,
        fitting_only_predicates_rebuilt=True, all_validation_selections_replayed=True,
        work_caps_checked=True, real_certificate_rows=12,
        real_solver_incumbents_replayed=replayed_incumbents,
        certificate_source_hashes_and_amendment_verified=True,
        historical_cohorts_not_pooled=True, primary_endpoint_frozen_before_fitting=True)
    assert checks['settings'] == 72 and checks['method_grids'] == 576 and checks['budget_rows'] == 1728
    write_json(OUT/'audit.json', checks); write_json(OUT/'input_sha256.json', hashes)
    frame, runtime, protocol = pd.DataFrame(metrics), pd.DataFrame(times), pd.DataFrame(protocols)
    frame.to_csv(OUT/'metrics.csv', index=False)
    runtime.to_csv(OUT/'runtime.csv', index=False)
    protocol.to_csv(OUT/'teacher_quality.csv', index=False)
    return frame, runtime, protocol, certificate


def paired(frame, first, second):
    keys = ['dataset', 'seed', 'teacher', 'noise', 'budget']
    wide = frame.pivot(index=keys, columns='method', values='balanced_fidelity')
    shared = wide.reindex(columns=[first, second]).dropna()
    differences = shared[first]-shared[second]
    means = differences.groupby(level='dataset').mean()
    values = means.to_numpy()
    if not len(values):
        return dict(first=first, second=second, pairs=0, datasets=0)
    rng = np.random.default_rng(15102026)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return dict(first=first, second=second, mean=float(values.mean()),
        low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)),
        pairs=len(shared), datasets=len(values), per_dataset=means.to_dict(),
        leave_one_dataset_min=(float(min(np.delete(values, i).mean() for i in range(len(values)))) if len(values) > 1 else None),
        leave_one_dataset_max=(float(max(np.delete(values, i).mean() for i in range(len(values)))) if len(values) > 1 else None))


def latex_table(name, caption, label, headers, rows, aligns=None):
    aligns = aligns or 'l'+'r'*(len(headers)-1)
    text = ('\\begin{kbtable}\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
        '\\small\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+aligns+'}\n\\toprule\n'
        +' & '.join(headers)+r' \\'+'\n\\midrule\n')
    text += '\n'.join(' & '.join(map(str, row))+r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{kbtable}\n'
    (OUT/name).write_text(text)


def summarize(frame, runtime, protocol, certificates):
    primary = frame[(frame.teacher == 'rf') & (frame.noise == 0) & (frame.budget == 8)]
    pairs = [paired(primary, 'DiagnosticLFIO', method) for method in METHODS if method != 'DiagnosticLFIO']
    write_json(OUT/'primary_paired.json', pairs)
    two_class = primary[primary.observed_teacher_classes == 2]
    two_class_pairs = [paired(two_class, 'DiagnosticLFIO', method) for method in METHODS if method != 'DiagnosticLFIO']
    write_json(OUT/'two_class_only_paired.json', two_class_pairs)
    secondary = []
    for teacher in TEACHERS:
        for noise in NOISE:
            for budget in BUDGETS:
                sample = frame[(frame.teacher == teacher)&(frame.noise == noise)&(frame.budget == budget)]
                for method in METHODS:
                    if method != 'DiagnosticLFIO':
                        secondary.append(dict(teacher=teacher, noise=noise, budget=budget,
                            **paired(sample, 'DiagnosticLFIO', method)))
    write_json(OUT/'all_paired.json', secondary)
    summary = (frame.groupby(['method', 'teacher', 'noise', 'budget', 'dataset'])
        .agg(balanced_fidelity=('balanced_fidelity', 'mean'), fidelity=('fidelity', 'mean'),
             conditions=('total_literals', 'mean'), feasible=('feasible', 'mean'))
        .groupby(['method', 'teacher', 'noise', 'budget']).mean().reset_index())
    summary.to_csv(OUT/'summary.csv', index=False)
    teacher_rows = []
    for name in DATASETS:
        X, y, _ = load_data(name)
        p = protocol[protocol.dataset == name]
        values = [100*p[(p.teacher == teacher)&(p.noise == noise)].teacher_balanced_accuracy.mean()
                  for teacher, noise in (('rf', 0.), ('hgb', 0.), ('rf', .2), ('hgb', .2))]
        teacher_rows.append([DATA_LABELS[name], len(X), X.shape[1], f'{y.mean()*100:.1f}']+[f'{v:.2f}' for v in values])
    latex_table('datasets.tex', 'Independent numerical tasks and teacher reference-label balanced accuracy (percent). Each teacher cell averages three fixed splits. Study/Run identifiers are removed from Climate. Rice and Raisin share a morphological domain.',
        'tab:independent-data', ['Dataset', '$n$', '$d$', 'Class 1 (\\%)', 'RF clean', 'HGB clean', 'RF noisy', 'HGB noisy'], teacher_rows)
    main_rows = []
    for method in METHODS:
        sample = frame[(frame.method == method)&(frame.budget == 8)]
        cells = []
        for teacher, noise in (('rf', 0.), ('rf', .2), ('hgb', 0.), ('hgb', .2)):
            g = sample[(sample.teacher == teacher)&(sample.noise == noise)]
            mean = g.groupby('dataset').balanced_fidelity.mean().mean()
            cells.append(f'{mean*100:.2f} ({g.feasible.sum()}/18)')
        conditions = sample.groupby('dataset').total_literals.mean().mean()
        cpu = runtime[runtime.method == method].cpu.median()
        main_rows.append([LABELS[method], *cells, f'{conditions:.2f}', f'{cpu:.2f}'])
    latex_table('main.tex', 'Eight-condition test macro fidelity (percent), averaging agreement over observed teacher classes. It equals two-class balanced fidelity except for one clean-RF Climate test fold, where the frozen score is ordinary agreement; a two-class-only sensitivity excludes that fold. Parentheses give feasible settings out of 18. Means use each method\'s feasible settings and must be read with coverage; paired comparisons use the common feasible subset. Conditions average deployed atomic tests. CPU is median seconds for the complete three-budget grid, including controller probes and predicate construction, excluding the teacher.',
        'tab:independent-main', ['Method', 'RF clean', 'RF noisy', 'HGB clean', 'HGB noisy', 'Conditions', 'CPU (s)'], main_rows)
    pair_rows = []
    for row in pairs:
        if row['pairs']:
            pair_rows.append([LABELS[row['second']], f"{row['mean']*100:+.2f}",
                f"[{row['low']*100:+.2f}, {row['high']*100:+.2f}]", row['pairs'], row['datasets']])
        else:
            pair_rows.append([LABELS[row['second']], '--', '--', 0, 0])
    latex_table('paired.tex', 'Diagnostic LFIO minus each comparator: clean RF, eight conditions, observed-class macro-fidelity percentage points. Only the median-restarts row is the prespecified primary comparison. Descriptive 95\\% intervals resample the six dataset means, retaining within-dataset split dependence; secondary rows are unadjusted and exploratory. Feasible paired counts and contributing datasets are explicit.',
        'tab:independent-paired', ['Comparator', 'Difference (pp)', '95\\% interval', 'Pairs', 'Datasets'], pair_rows)
    per_data = primary.pivot_table(index='dataset', columns='method', values='balanced_fidelity').reindex(index=DATASETS, columns=METHODS)
    per_data.to_csv(OUT/'primary_per_dataset.csv')
    selected_methods = ('MedianRestarts', 'MDLRestarts', 'DiagnosticLFIO', 'FastSRS', 'BRCG-light', 'NativeTree', 'NeuRulesNative')
    coverage = primary.pivot_table(index='dataset', columns='method', values='feasible', aggfunc='sum').reindex(index=DATASETS, columns=METHODS)
    coverage.to_csv(OUT/'primary_feasibility_per_dataset.csv')
    per_data_rows = []
    for name in DATASETS:
        cells = []
        for method in selected_methods:
            value = per_data.loc[name, method]
            cell = f'{value*100:.2f}' if np.isfinite(value) else '--'
            if method == 'NeuRulesNative':
                cell += f' ({int(coverage.loc[name, method])}/3)'
            cells.append(cell)
        per_data_rows.append([DATA_LABELS[name], *cells])
    latex_table('per_dataset.tex', 'Clean-RF eight-condition observed-class macro fidelity (percent), averaging feasible splits within each dataset. NeuRules cells include the exact feasible count out of three; all other cells use all three splits. A missing value indicates no feasible native model in the fixed grid.',
        'tab:independent-per-data', ['Dataset', 'Median', 'MDL', 'Diagnostic', 'FastSRS', 'BRCG', 'Tree', 'NeuRules (count)'], per_data_rows)
    policy = frame[frame.method == 'DiagnosticLFIO']
    actions = policy.groupby(['budget', 'action']).size().unstack(fill_value=0)
    actions.to_csv(OUT/'actions.csv')
    policy_rows = []
    for budget in BUDGETS:
        selected = policy[policy.budget == budget]
        policy_rows.append([budget]+[int(actions.loc[budget].get(name, 0)) for name in ('stop', 'search', 'vocabulary')]
            +[int(selected.capacity_witness.sum()), f'{selected.scores_used.mean()/TOTAL_SCORES*100:.1f}'])
    latex_table('actions.tex', 'Fitting-only controller decisions across 72 settings at each budget. The capacity witness uses twice the allowed conditions only as an internal probe; it is never deployed. Scored work is actual mean usage divided by the common 1.2-million maximum allowance. Probe scores are included.',
        'tab:independent-actions', ['Budget', 'Stop', 'Search', 'Vocabulary', 'Capacity witness', 'Work (\\%)'], policy_rows)
    bound_rows = []
    for name in DATASETS:
        for method in ('MedianRestarts', 'MDLRestarts'):
            row = next(r for r in certificates['rows'] if r['dataset'] == name and r['method'] == method)
            b = row['bounds']
            bound_rows.append([DATA_LABELS[name], 'Median' if method == 'MedianRestarts' else 'MDL',
                row['fitting_samples'], row['predicates'], row['candidates'], f"{100*b['collision']:.2f}",
                f"[{100*b['capacity_lower']:.2f}, {100*b['capacity_upper']:.2f}]",
                f"[{100*b['search_lower']:.2f}, {100*b['search_upper']:.2f}]",
                f"{sum(s['optimal'] for s in row['solves'])}/2"])
    latex_table('certificates.tex', 'Real-task decomposition bounds for clean RF, seed 6001, eight conditions. All fitting rows and complete supported depth-two pools participate. Collision and capacity/search intervals are balanced-error percentage points; the final column counts proven-optimal head solves out of two, with a 15-second limit per head. Timeout dual bounds remain valid but do not certify an optimum. The MDL language nests all median predicates.',
        'tab:real-certificates', ['Dataset', 'Vocabulary', '$n_{fit}$', '$d_{pred}$', 'Bodies', 'Collision', 'Capacity interval', 'Search interval', 'Optimal'], bound_rows)
    primary_pair = next(r for r in pairs if r['second'] == 'MedianRestarts')
    mdl_pair = next(r for r in pairs if r['second'] == 'MDLRestarts')
    primary_means = (primary.groupby(['method', 'dataset']).balanced_fidelity.mean()
        .groupby('method').mean())
    stop_fraction = float((policy.action == 'stop').mean())
    sensitivity = next(r for r in two_class_pairs if r['second'] == 'MedianRestarts')
    findings = (f"At the prespecified independent clean-RF endpoint, Diagnostic LFIO achieves {primary_means['DiagnosticLFIO']*100:.2f}\\% observed-class macro fidelity versus {primary_means['MedianRestarts']*100:.2f}\\% for median restarts. "
        f"The paired difference is {primary_pair['mean']*100:+.2f} percentage points (dataset-cluster 95\\% interval [{primary_pair['low']*100:+.2f}, {primary_pair['high']*100:+.2f}]); "
        f"the interval {('includes' if primary_pair['low'] <= 0 <= primary_pair['high'] else 'excludes')} zero. "
        f"Against fixed MDL restarts the difference is {mdl_pair['mean']*100:+.2f} points [{mdl_pair['low']*100:+.2f}, {mdl_pair['high']*100:+.2f}]. "
        f"These results do not establish that adaptive allocation improves fidelity over a fixed enriched vocabulary. "
        f"The controller stops in {stop_fraction*100:.1f}\\% of the 216 budget decisions and uses {policy.scores_used.mean()/TOTAL_SCORES*100:.1f}\\% of its maximum scoring allowance on average. "
        f"Excluding the one single-class test-teacher fold yields a two-class-only paired sensitivity of {sensitivity['mean']*100:+.2f} points [{sensitivity['low']*100:+.2f}, {sensitivity['high']*100:+.2f}] over {sensitivity['pairs']} splits and six dataset means.\n")
    (OUT/'findings.tex').write_text(findings)
    numbers = dict(primary=primary_pair, versus_mdl=mdl_pair, two_class_sensitivity=sensitivity,
        balanced_fidelity={key: float(v) for key, v in primary_means.items()},
        controller_stop_fraction=stop_fraction, controller_work_fraction=float(policy.scores_used.mean()/TOTAL_SCORES),
        optimal_head_solves=certificates['optimal_head_solves'], total_head_solves=certificates['total_head_solves'],
        teacher_balanced_accuracy=protocol.groupby(['teacher', 'noise']).teacher_balanced_accuracy.mean().to_dict())
    numbers['teacher_balanced_accuracy'] = {f'{k[0]}_{k[1]}': float(v) for k, v in numbers['teacher_balanced_accuracy'].items()}
    write_json(OUT/'numbers.json', numbers)
    (OUT/'numbers.tex').write_text('\n'.join([
        f"\\newcommand{{\\IndependentDiagnosticBF}}{{{100*primary_means['DiagnosticLFIO']:.2f}}}",
        f"\\newcommand{{\\IndependentMedianBF}}{{{100*primary_means['MedianRestarts']:.2f}}}",
        f"\\newcommand{{\\IndependentMDLBF}}{{{100*primary_means['MDLRestarts']:.2f}}}",
        f"\\newcommand{{\\IndependentGain}}{{{100*primary_pair['mean']:+.2f}}}",
        f"\\newcommand{{\\IndependentGainLow}}{{{100*primary_pair['low']:+.2f}}}",
        f"\\newcommand{{\\IndependentGainHigh}}{{{100*primary_pair['high']:+.2f}}}",
        f"\\newcommand{{\\IndependentStop}}{{{100*stop_fraction:.1f}}}",
        f"\\newcommand{{\\IndependentWork}}{{{policy.scores_used.mean()/TOTAL_SCORES*100:.1f}}}",
        f"\\newcommand{{\\IndependentOptimal}}{{{certificates['optimal_head_solves']}}}",
    ])+'\n')
    return numbers, per_data, policy


def figures(frame, runtime, per_data, policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), layout='constrained')
    ax = axes[0]
    for i, name in enumerate(DATASETS):
        a, b = per_data.loc[name, 'MedianRestarts']*100, per_data.loc[name, 'DiagnosticLFIO']*100
        ax.plot([a, b], [i, i], color='#a2b6c5', lw=3)
        ax.scatter(a, i, color='#677a86', s=28, zorder=3)
        ax.scatter(b, i, color='#0b7285', marker='D', s=30, zorder=3)
    ax.set_yticks(range(len(DATASETS)), [DATA_LABELS[n] for n in DATASETS]); ax.invert_yaxis()
    ax.set_xlabel('Clean-RF macro fidelity (%)'); ax.set_title('(a) Median restarts → diagnostic')
    actions = policy.groupby(['budget', 'action']).size().unstack(fill_value=0).reindex(BUDGETS)
    bottom = np.zeros(3)
    for action, color in [('stop', '#677a86'), ('search', '#db9d36'), ('vocabulary', '#0b7285')]:
        values = actions.get(action, pd.Series(0, index=BUDGETS)).to_numpy()/72*100
        axes[1].bar(np.arange(3), values, bottom=bottom, color=color, label=action)
        bottom += values
    axes[1].set_xticks(range(3), BUDGETS); axes[1].set_xlabel('Condition budget')
    axes[1].set_ylabel('Decisions (%)'); axes[1].set_title('(b) Actions chosen without test data')
    axes[1].legend(frameon=False, fontsize=7, loc='lower center',
                   bbox_to_anchor=(.5, 1.02), ncols=3)
    axes[1].set_title('(b) Actions chosen without test data', pad=26)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.savefig(OUT/'independent_results.png', dpi=220)
    fig.savefig(OUT/'independent_results.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-workers', type=int, default=16)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, runtime, protocol, certificates = audit(args.replay_workers)
    numbers, per_data, policy = summarize(frame, runtime, protocol, certificates)
    figures(frame, runtime, per_data, policy)
    print(json.dumps(numbers, indent=2))


if __name__ == '__main__':
    main()
