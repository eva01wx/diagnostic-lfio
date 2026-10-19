"""Frozen exploratory study of contextual and cost-aware query forgetting."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import tracemalloc

import numpy as np

from run_feature_forgetting import ROOT, OUT as OLD, PARENT, read, signature as old_signature
from run_rule_generalization import BASES, SEEDS
from run_prospective_revision import DATASETS, TEACHERS, NOISE, BUDGETS, load_data, digest, write_json
from src.contextual_forgetting import ContextualForgetter, DifferenceBound as B
from src.feature_forgetting import FeatureForgetter
from src.forgetting_selection import select_features
from src.rule_generalization import Atom, RuleProgram
from research_baselines.contextual_oracle import constrained_bounds
from research_baselines.forgetting_oracle import enumeration_bounds

OUT=ROOT/'results/contextual_forgetting_20261016'
REPORT=ROOT/'docs/research/contextual_forgetting_2026-10-16'
SOURCES=('src/contextual_forgetting.py','src/forgetting_selection.py',
         'research_baselines/contextual_oracle.py','run_contextual_forgetting.py',
         'docs/research/CONTEXTUAL_FORGETTING_PROTOCOL_2026-10-16.md')
TARGETS=(.7,.8,.9)


def signature():
    assert read(OLD/'manifest.json')['signature']==old_signature()
    inputs=sorted(OLD.glob('*/*_Original_fit.json'))
    assert len(inputs)==648
    return dict(sources={n:digest(ROOT/n) for n in SOURCES},
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        parent_manifest=digest(OLD/'manifest.json'),source_variant='Original',
        targets=list(TARGETS),max_nodes=50000,max_cache=4096,max_candidates=512,
        costs=['uniform 1','heterogeneous 1+(index modulo 5)'],
        interpretation='exploratory reuse of known cohort, no new independent datasets')


def vectors_expected(program,X,hidden):
    bounds=enumeration_bounds(program,X,hidden)
    return np.where(bounds['certain_head'],program.head,
                    np.where(~bounds['possible_head'],1-program.head,-1)).astype(np.int8)


def evaluate(fit,selection,test,old_fit):
    program=RuleProgram.from_export(fit['program']); source=program.predict(test)
    rows,comparisons,vectors=[],[],{'source':source}
    cache={}
    def predictions(features):
        key=tuple(features)
        if key not in cache:
            engine=ContextualForgetter().fit(program,key)
            selected=engine.predict(selection); predicted=engine.predict(test)
            hidden=test.copy();hidden[:,list(key)]=np.nan
            np.testing.assert_array_equal(predicted,engine.predict(hidden))
            oracle=vectors_expected(program,test[:3],key)
            actual=predicted[:3]
            assert np.all((actual < 0)|(actual == oracle))
            assigned=predicted>=0
            assert np.all(predicted[assigned]==source[assigned])
            if len(test):engine.verify_query(hidden[0],engine.query(hidden[0]))
            cache[key]=(selected,predicted,dict(engine.stats_),engine.cache_bytes())
        return cache[key]
    for profile,record in fit['profiles'].items():
        total_cost=sum(record['costs'][f] for f in record['active_features'])
        for choice in record['choices']:
            selected,predicted,stats,cache_bytes=predictions(choice['features'])
            assert np.mean(selected>=0)==choice['coverage'] and choice['coverage']>=choice['target']
            covered=predicted>=0
            class_coverage={str(c):float(np.mean(covered[source==c])) if np.any(source==c) else None for c in (0,1)}
            key=f'c{len(rows)}';vectors[key]=predicted
            rows.append(dict(profile=profile,**choice,test_coverage=float(covered.mean()),
                hidden_count=len(choice['features']),hidden_cost_fraction=choice['hidden_cost']/total_cost if total_cost else 0.,
                class0_coverage=class_coverage['0'],class1_coverage=class_coverage['1'],
                test_target_shortfall=bool(covered.mean()<choice['target']),
                contradictions=int(np.sum(predicted[covered]!=source[covered])),
                stats=stats,cache_payload_bytes=cache_bytes,prediction_key=key))
    for key,record in old_fit['projections'].items():
        selected,predicted,stats,cache_bytes=predictions(record['features'])
        envelope=record['envelope']
        # Saved formulas are the comparator; frozen compiler already independently audited.
        lower,upper=RuleProgram.from_export(envelope['certain']),RuleProgram.from_export(envelope['possible'])
        old_prediction=np.where(lower.predict(test)==program.head,program.head,
                                np.where(upper.predict(test)!=program.head,1-program.head,-1))
        if envelope['audit']['certain_exact']:
            assert np.all((predicted<0)|(predicted==old_prediction))
            if not stats['unknown_patterns']:np.testing.assert_array_equal(predicted,old_prediction)
        keyv=f'p{len(comparisons)}';vectors[keyv]=predicted
        comparisons.append(dict(projection_key=key,features=record['features'],
            old_exact=envelope['audit']['certain_exact'],old_coverage=float(np.mean(old_prediction>=0)),
            query_coverage=float(np.mean(predicted>=0)),
            gained=int(np.sum((old_prediction<0)&(predicted>=0))),
            lost=int(np.sum((old_prediction>=0)&(predicted<0))),
            stats=stats,cache_payload_bytes=cache_bytes,prediction_key=keyv))
    return dict(rows=rows,comparisons=comparisons,unique_subsets=len(cache)),vectors


def one(task):
    dataset,seed,teacher,noise=task
    stem=f'{dataset}_{seed}_{teacher}_noise{round(noise*100):02d}'
    folder=OUT/stem;folder.mkdir(parents=True,exist_ok=True)
    raw,_,_=load_data(dataset);split=np.load(PARENT/stem/'split.npz')
    selection=raw[split['selection']]
    for base in BASES:
        for budget in BUDGETS:
            name=f'{base}_B{budget}_Original';old_path=OLD/stem/(name+'_fit.json')
            old_fit=read(old_path);fit_path=folder/(name+'_fit.json');result_path=folder/(name+'.json')
            if result_path.exists():continue
            if not fit_path.exists():
                program=RuleProgram.from_export(old_fit['program']);profiles={}
                start=time.process_time()
                for profile in ('uniform','heterogeneous'):
                    costs=np.ones(program.n_features) if profile=='uniform' else 1+np.arange(program.n_features)%5
                    profiles[profile]=select_features(program,selection,costs,targets=TARGETS)
                fit=dict(dataset=dataset,seed=seed,teacher=teacher,noise=noise,base=base,budget=budget,
                    program=program.export(),source_file=str(old_path.relative_to(ROOT)),source_sha256=digest(old_path),
                    profiles=profiles,selection_cpu=time.process_time()-start)
                write_json(fit_path,fit)
            fit=read(fit_path)
            # The test matrix is first indexed only after all final choices are saved.
            result,vectors=evaluate(fit,selection,raw[split['test']],old_fit)
            np.savez_compressed(folder/(name+'_predictions.npz'),**vectors)
            write_json(result_path,dict(fit_sha256=digest(fit_path),**result))
    print(stem,'finished',flush=True)
    return stem


def controlled():
    source=RuleProgram(((Atom(0,0.,True),),),1,3)
    cases=dict(omitted=[],equality=[B(0,1),B(1,0)],order=[B(1,0)],
        band=[B(1,0,'1/4',True),B(0,1,'1/4')],
        wrong_equality=[B(0,1,-2),B(1,0,2)],
        infeasible=[B(0,1),B(1,0),B(1,None,1)])
    records=[]
    for name,constraints in cases.items():
        engine=ContextualForgetter().fit(source,[0],constraints)
        for y in (-2.,-1.,0.,1.,2.):
            row=[y,y,0.];result=engine.query(row)
            oracle=constrained_bounds(source,row,[0],constraints)
            assert result['prediction']==oracle['prediction']
            records.append(dict(case=name,y=y,constraints=[b.export() for b in constraints],
                                **result,true_domain_class=int(y>0)))
    rng=np.random.default_rng(20261016);random_records=[]
    for index in range(100):
        source=RuleProgram(tuple(tuple(Atom(int(rng.integers(3)),float(rng.integers(-1,2)),bool(rng.integers(2)))
                                      for _ in range(int(rng.integers(1,4)))) for _ in range(int(rng.integers(5)))),index%2,3)
        constraints=[B(int(rng.integers(3)),int(rng.integers(3)),int(rng.integers(-1,2)),bool(rng.integers(2))) for _ in range(3)]
        for hidden in ((0,),(0,1),(0,1,2)):
            model=ContextualForgetter().fit(source,hidden,constraints)
            for z in (-1.,0.,1.):
                row=[0.,0.,z];result=model.query(row);oracle=constrained_bounds(source,row,hidden,constraints)
                assert result['prediction']==oracle['prediction']
                random_records.append(dict(index=index,program=source.export(),features=list(hidden),
                    constraints=[b.export() for b in constraints],row=row,query=result,oracle=oracle))
    write_json(OUT/'controlled.json',dict(cases=records,random=random_records))


def benchmark_one(task):
    path,key=task
    fit=read(ROOT/path);program=RuleProgram.from_export(fit['program']);record=fit['projections'][key]
    raw,_,_=load_data(fit['dataset']);split=np.load(PARENT/Path(path).parent.name/'split.npz')
    X=raw[split['test']][:64];features=record['features'];results={}
    for kind in ('global','query'):
        tracemalloc.start();start=time.process_time()
        engine=(FeatureForgetter().fit(program,features) if kind=='global' else ContextualForgetter().fit(program,features))
        setup=time.process_time()-start;start=time.process_time();pred=engine.predict(X)
        query=time.process_time()-start;start=time.process_time();warm=engine.predict(X)
        warm_cpu=time.process_time()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        np.testing.assert_array_equal(pred,warm)
        results[kind]=dict(setup_cpu=setup,batch_cpu=query,warm_cpu=warm_cpu,total_cpu=setup+query,
                           python_peak_bytes=peak,coverage=float(np.mean(pred>=0)))
        if kind=='query':results[kind].update(stats=engine.stats_,cache_payload_bytes=engine.cache_bytes())
    return dict(source_file=path,projection_key=key,features=features,n_rows=len(X),
                expanded_tests=sum(len(r) for part in ('certain','possible') for r in record['envelope'][part]['rules']),
                **results)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);manifest=OUT/'manifest.json';current=signature()
    if manifest.exists():assert read(manifest)['signature']==current,'Frozen source/configuration mismatch'
    else:write_json(manifest,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),signature=current))
    if args.prepare_only:print(manifest);return
    controlled();tasks=[(d,s,t,z) for d in DATASETS for s in SEEDS for t in TEACHERS for z in NOISE]
    failures=[];finished=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for future in as_completed(futures):
            try:finished.append(future.result())
            except Exception:failures.append(dict(task=list(futures[future]),traceback=traceback.format_exc()))
    write_json(OUT/'completion.json',dict(settings=len(finished),programs=len(list(OUT.glob('*/*_Original.json'))),failures=failures))
    if failures or len(finished)!=72:raise SystemExit('Incomplete study')
    candidates=[]
    for path in sorted(OLD.glob('*/*_Original_fit.json')):
        for key,r in read(path)['projections'].items():
            size=sum(len(rule) for part in ('certain','possible') for rule in r['envelope'][part]['rules'])
            candidates.append((-size,str(path.relative_to(ROOT)),key))
    selected=sorted(candidates)[:32]
    with ProcessPoolExecutor(max_workers=min(4,args.workers)) as pool:
        benchmarks=list(pool.map(benchmark_one,[(p,k) for _,p,k in selected]))
    write_json(OUT/'benchmark.json',benchmarks)
    print('Completed',len(finished),'settings and',len(benchmarks),'representation benchmarks')


if __name__=='__main__':main()
