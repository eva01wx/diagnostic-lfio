"""Fixed-cohort evaluation of whole-coordinate numerical feature forgetting."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from run_rule_generalization import (SEEDS, BASES, OUT as PARENT,
                                     signature as parent_signature)
from run_prospective_revision import DATASETS, TEACHERS, NOISE, BUDGETS, load_data, digest, score, write_json
from src.feature_forgetting import FeatureForgetter
from src.rule_generalization import RuleProgram

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/feature_forgetting_20261016_v2'
VARIANTS = ('Original','BoundedGeneralize')
MAX_NODES, MAX_BOXES = 50000, 4096
POLICIES = ('Fixed1','Certainty1','Fixed2','Certainty2')


def read(path):
    return json.loads(path.read_text())


def signature():
    assert read(PARENT/'manifest.json')['signature'] == parent_signature()
    inputs = list(PARENT.glob('*/*_fit.json')) + list(PARENT.glob('*/split.npz'))
    assert len(inputs) == 720
    return dict(parent=parent_signature(), parent_inputs={str(p.relative_to(ROOT)):digest(p) for p in sorted(inputs)},
        variants=list(VARIANTS),max_nodes=MAX_NODES,max_boxes=MAX_BOXES,
        sources={name:digest(ROOT/name) for name in ('src/feature_forgetting.py',
            'research_baselines/forgetting_oracle.py','run_feature_forgetting.py',
            'docs/research/FEATURE_FORGETTING_PROTOCOL_2026-10-16.md',
            'docs/research/FEATURE_FORGETTING_AMENDMENT_2026-10-16.md')},
        interpretation='post hoc reuse of known frozen source programs, not new independent training',
        domain='all finite real completions; no empirical range or cross-feature feasibility assumptions')


def size(model):
    return tuple(a+b for a,b in zip(model.certain_.complexity,model.possible_.complexity))


def selection_key(record):
    return (-record['selection_coverage'], *record['combined_complexity'][:2],
            tuple(record['features']))


def compile_choices(program, selection):
    active = sorted({a.feature for r in program.rules for a in r})
    records = {}
    def add(features):
        features = tuple(sorted(features)); key = ','.join(map(str,features)) or 'none'
        if key not in records:
            start,cpu = time.perf_counter(),time.process_time()
            model = FeatureForgetter(max_nodes=MAX_NODES,max_boxes=MAX_BOXES).fit(program,features)
            compile_cpu,compile_wall = time.process_time()-cpu,time.perf_counter()-start
            records[key] = dict(features=list(features),envelope=model.export(),
                selection_coverage=float(np.mean(model.predict(selection)>=0)),
                combined_complexity=list(size(model)),cpu_seconds=compile_cpu,seconds=compile_wall)
        return key
    singles = [add([feature]) for feature in active] or [add([])]
    first = min(singles,key=lambda k:selection_key(records[k]))
    fixed_first = singles[0]
    fixed_pair = add(active[:2])
    chosen = records[first]['features']
    candidates = [add(chosen+[feature]) for feature in active if feature not in chosen]
    second = min(candidates,key=lambda k:selection_key(records[k])) if candidates else first
    unused = next((f for f in range(program.n_features) if f not in active),None)
    unused_key = add([unused]) if unused is not None else None
    return dict(active_features=active,projections=records,unused_control=unused_key,
                policies=dict(Fixed1=fixed_first,Certainty1=first,Fixed2=fixed_pair,Certainty2=second))


def evaluate(fit, X, teacher):
    program = RuleProgram.from_export(fit['program'])
    original = program.predict(X)
    predictions, rows = {'source':original}, []
    for index,(key,record) in enumerate(fit['projections'].items()):
        model = FeatureForgetter.from_export(record['envelope'])
        bound = model.bounds(X)
        partial,upper = bound['prediction'],bound['possible_head']
        covered = partial>=0
        assert np.all(partial[covered]==original[covered])
        filled = X.copy()
        filled[:,record['features']] = np.asarray(fit['fitting_medians'])[record['features']]
        median = program.predict(filled)
        existential = np.where(upper,program.head,1-program.head).astype(np.uint8)
        size_before = program.complexity
        rows.append(dict(projection_key=key,features=record['features'],forgotten_count=len(record['features']),
            active_singleton=bool(len(record['features'])==1 and record['features'][0] in fit['active_features']),
            unused_control=key==fit['unused_control'],policies=[p for p,k in fit['policies'].items() if k==key],
            coverage=float(covered.mean()),covered_n=int(covered.sum()),test_n=len(X),
            certain_head_n=int(bound['certain_head'].sum()),certain_default_n=int((~upper).sum()),
            contradictions_source=int((partial[covered]!=original[covered]).sum()),
            certain_exact=model.audit_['certain_exact'],box_subtractions=model.audit_['box_subtractions'],
            source_tests=size_before[0],possible_tests=model.possible_.complexity[0],
            certain_tests=model.certain_.complexity[0],combined_tests=size(model)[0],
            teacher_macro_selective=score(teacher[covered],partial[covered]) if covered.any() else None,
            source_teacher_macro=score(teacher,original),
            existential_disagreement=float(np.mean(existential!=original)),
            median_disagreement=float(np.mean(median!=original)),
            existential_teacher_macro=score(teacher,existential),median_teacher_macro=score(teacher,median),
            compile_cpu=record['cpu_seconds'],compile_seconds=record['seconds']))
        for label,value in (('certain',bound['certain_head']),('possible',upper),('partial',partial),
                            ('median',median),('existential',existential)):
            predictions[f'p{index}_{label}'] = value
    return rows,predictions


def one(task):
    dataset,seed,teacher_kind,noise = task
    stem = f'{dataset}_{seed}_{teacher_kind}_noise{round(noise*100):02d}'
    source,folder = PARENT/stem,OUT/stem
    folder.mkdir(parents=True,exist_ok=True)
    raw,_,_ = load_data(dataset)
    split = np.load(source/'split.npz')
    for base in BASES:
        for budget in BUDGETS:
            parent_file = source/f'{base}_B{budget}_fit.json'; parent = read(parent_file)
            for variant in VARIANTS:
                name=f'{base}_B{budget}_{variant}'
                result_file,fit_file = folder/(name+'.json'),folder/(name+'_fit.json')
                if result_file.exists():continue
                if not fit_file.exists():
                    program = RuleProgram.from_export(parent['programs'][variant])
                    fit = dict(dataset=dataset,seed=seed,teacher=teacher_kind,noise=noise,base=base,
                        budget=budget,variant=variant,source_fit=str(parent_file.relative_to(ROOT)),
                        source_fit_sha256=digest(parent_file),program=program.export(),
                        fitting_medians=np.median(raw[split['train']],axis=0).tolist(),
                        **compile_choices(program,raw[split['selection']]))
                    # All missing-coordinate choices and formulas precede test inference.
                    write_json(fit_file,fit)
                fit=read(fit_file)
                rows,predictions = evaluate(fit,raw[split['test']],split['teacher_test'])
                np.savez_compressed(folder/(name+'_predictions.npz'),**predictions)
                write_json(result_file,dict(fit_sha256=digest(fit_file),rows=rows))
    print(stem,'finished',flush=True)
    return stem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'manifest.json';current=signature()
    if manifest.exists():assert read(manifest)['signature']==current,'Frozen source/configuration mismatch'
    else:write_json(manifest,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),signature=current))
    if args.prepare_only:print(manifest);return
    tasks=[(n,s,t,z) for n in DATASETS for s in SEEDS for t in TEACHERS for z in NOISE]
    finished,failures=[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for future in as_completed(futures):
            try:finished.append(future.result())
            except Exception:failures.append(dict(task=list(futures[future]),traceback=traceback.format_exc()))
    records=[p for p in OUT.glob('*/*_B*.json') if not p.stem.endswith('_fit')]
    write_json(OUT/'completion.json',dict(expected_settings=72,completed_settings=len(finished),
        expected_source_programs=1296,completed_source_programs=len(records),
        projection_rows=sum(len(read(p)['rows']) for p in records),failures=failures))
    if failures or len(records)!=1296:raise SystemExit('Study incomplete; see completion.json')


if __name__=='__main__':main()
