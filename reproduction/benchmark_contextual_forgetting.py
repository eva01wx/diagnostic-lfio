"""Untraced timing refinement on the existing 32 representation stress cases."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import time

import numpy as np

from run_contextual_forgetting import ROOT,OUT,PARENT,read
from run_prospective_revision import load_data,digest,write_json
from src.contextual_forgetting import ContextualForgetter
from src.feature_forgetting import FeatureForgetter
from src.rule_generalization import RuleProgram


def one(task):
    index,case=task;fit=read(ROOT/case['source_file'])
    program=RuleProgram.from_export(fit['program']);features=case['features']
    raw,_,_=load_data(fit['dataset']);split=np.load(PARENT/Path(case['source_file']).parent.name/'split.npz')
    X=raw[split['test']][:64];records={}
    for kind in (('global','query') if index%2==0 else ('query','global')):
        samples=[]
        for _ in range(3):
            start=time.process_time()
            model=(FeatureForgetter().fit(program,features) if kind=='global' else ContextualForgetter().fit(program,features))
            setup=time.process_time()-start;start=time.process_time();pred=model.predict(X)
            first=time.process_time()-start;start=time.process_time();warm=model.predict(X)
            warm_cpu=time.process_time()-start
            np.testing.assert_array_equal(pred,warm)
            covered=pred>=0;assert np.all(pred[covered]==program.predict(X)[covered])
            samples.append(dict(setup_cpu=setup,batch_cpu=first,warm_cpu=warm_cpu,total_cpu=setup+first))
        records[kind]=dict(samples=samples,**{k:float(np.median([r[k] for r in samples])) for k in samples[0]})
    return dict(source_file=case['source_file'],projection_key=case['projection_key'],**records)


def main():
    sources=['benchmark_contextual_forgetting.py','docs/research/CONTEXTUAL_BENCHMARK_NOTE_2026-10-16.md',
             'src/contextual_forgetting.py','src/feature_forgetting.py']
    signature=dict(sources={n:digest(ROOT/n) for n in sources},input=digest(OUT/'benchmark.json'),
                   repeats=3,tracing=False)
    manifest=OUT/'benchmark_untraced_manifest.json'
    if manifest.exists():assert read(manifest)['signature']==signature
    else:write_json(manifest,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),signature=signature))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(one,enumerate(read(OUT/'benchmark.json'))))
    write_json(OUT/'benchmark_untraced.json',rows);print('Timed',len(rows),'fixed cases without tracing')


if __name__=='__main__':main()
