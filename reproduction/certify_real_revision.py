"""Real-data finite-pool bounds under the prospectively specified protocol."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
from run_prospective_revision import (ROOT, OUT, DATASETS, load_data, digest,
    write_json, fingerprint, predict_export)
from src.lfio_diagnostic import AbsoluteWorkLFIO, balanced_collision
from src.lfio_followup import apply_vocabulary
from src.lfio_exact import solve_pool

CERT = OUT/'certificates'


def gap_intervals(collision, heuristic, lower, upper):
    """Bounds for E = C + (E* - C) + (E - E*) on the fitting sample."""
    lower = max(collision, lower)
    upper = min(heuristic, upper)
    if lower > upper+1e-6:
        raise AssertionError('Inconsistent certificate bounds')
    lower = min(lower, upper)  # Only absorb solver tolerance, never a real contradiction.
    return dict(collision=collision, heuristic_error=heuristic, optimum_lower=lower,
        optimum_upper=upper, capacity_lower=max(0., lower-collision),
        capacity_upper=max(0., upper-collision), search_lower=max(0., heuristic-upper),
        search_upper=max(0., heuristic-lower))


def one(task):
    name, method = task
    target = CERT/f'{name}_{method}.json'
    if target.exists():
        return json.loads(target.read_text())
    folder = OUT/f'{name}_6001_rf_noise00'
    split = np.load(folder/'split.npz')
    record = json.loads((folder/f'{method}.json').read_text())
    selected = next(r for r in record['selections'] if r['budget'] == 8)
    export = record['candidates'][selected['candidate_index']]['symbolic']
    raw, _, _ = load_data(name)
    X, y = apply_vocabulary(raw[split['train']], export['vocabulary']), split['teacher_train']
    model = AbsoluteWorkLFIO(depths=(2,), penalties=(0.,), literal_budget=8,
        candidate_budget=75000, min_support=3)
    pool = model._candidates(X)
    prediction = predict_export(method, export, raw[split['train']])
    error = sum(.5*np.mean(prediction[y == c] != c) for c in (0, 1))
    collision = balanced_collision(X, y)['error_floor']
    started, cpu = time.perf_counter(), time.process_time()
    solves = [solve_pool(X, y, pool, head=head, alpha=0., literal_budget=8,
        time_limit=15., max_candidates=75000) for head in (0, 1)]
    lower = min(s['lower_bound']/len(y) if s['lower_bound'] is not None else collision for s in solves)
    upper = min([error]+[s['objective']/len(y) for s in solves if s['objective'] is not None])
    row = dict(dataset=name, method=method, fitting_samples=len(X), predicates=X.shape[1],
        candidates=len(pool), training_sha256=fingerprint(X, y),
        selected_source_sha256=digest(folder/f'{method}.json'),
        runtime_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
        both_heads_optimal=all(s['optimal'] for s in solves),
        gap_certified=bool(abs(upper-max(collision, lower)) < 1e-7),
        bounds=gap_intervals(collision, error, lower, upper), solves=solves,
        scope='all fitting rows; complete supported depth-two pool; eight atomic tests; two heads; no length penalty')
    write_json(target, row)
    print(name, method, 'bounds', row['bounds'], flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    CERT.mkdir(parents=True, exist_ok=True)
    signature = dict(sources={n: digest(ROOT/n) for n in ('certify_real_revision.py', 'src/lfio_exact.py')},
        parent_manifest_sha256=digest(OUT/'manifest.json'), datasets=list(DATASETS),
        seed=6001, teacher='rf', noise=0., budget=8, per_head_seconds=15.,
        vocabularies=['MedianRestarts', 'MDLRestarts'], full_fitting_samples=True)
    manifest = CERT/'manifest.json'
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior != signature:
            amendment = json.loads((CERT/'execution_amendment.json').read_text())
            expected = json.loads(json.dumps(prior))
            assert amendment['old_source_sha256'] == expected['sources']['certify_real_revision.py']
            assert amendment['new_source_sha256'] == signature['sources']['certify_real_revision.py']
            expected['sources']['certify_real_revision.py'] = amendment['new_source_sha256']
            assert expected == signature, 'Only the recorded serialization amendment is permitted'
    else:
        write_json(manifest, signature)
    tasks = [(name, method) for name in DATASETS for method in ('MedianRestarts', 'MDLRestarts')]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, t) for t in tasks]):
            rows.append(future.result())
    write_json(CERT/'completion.json', dict(expected=len(tasks), completed=len(rows),
        optimal_head_solves=sum(s['optimal'] for row in rows for s in row['solves']),
        total_head_solves=2*len(tasks), rows=rows))


if __name__ == '__main__':
    main()
