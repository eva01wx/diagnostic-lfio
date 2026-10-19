"""Independently audit and summarize numerical feature-forgetting records."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/lfio_feature_forgetting_mpl')
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from run_feature_forgetting import (ROOT,OUT as RUN,PARENT,VARIANTS,BASES,POLICIES,
    SEEDS,DATASETS,TEACHERS,NOISE,BUDGETS,read,signature,evaluate)
from run_prospective_revision import load_data,digest,write_json
from research_baselines.forgetting_oracle import enumeration_bounds
from src.feature_forgetting import FeatureForgetter
from src.rule_generalization import RuleProgram,verify_relation

OUT=ROOT/'docs/research/feature_forgetting_2026-10-16'
KEY=['dataset','seed','teacher','noise','base','budget','variant']


def feature_key(features):
    return ','.join(map(str,sorted(features))) or 'none'


def audit_ready(task):
    dataset,seed,teacher,noise=task
    folder=RUN/f'{dataset}_{seed}_{teacher}_noise{round(noise*100):02d}'
    expected=[folder/f'{b}_B{k}_{v}.json' for b in BASES for k in BUDGETS for v in VARIANTS]
    deadline=time.monotonic()+3600
    while True:
        if (RUN/'completion.json').exists() and read(RUN/'completion.json')['failures']:
            raise RuntimeError('Source execution failed; do not wait for missing records')
        try:
            if all(p.is_file() for p in expected):
                for p in expected:read(p)
                return audit_folder(task)
        except json.JSONDecodeError:
            pass  # The final small result file may still be being written.
        if time.monotonic()>deadline:raise RuntimeError('Timed out awaiting completed folder: '+str(folder))
        time.sleep(1)


def audit_folder(task):
    dataset,seed,teacher_kind,noise=task
    stem=f'{dataset}_{seed}_{teacher_kind}_noise{round(noise*100):02d}'
    folder,source=RUN/stem,PARENT/stem
    raw,_,_=load_data(dataset);split=np.load(source/'split.npz')
    selection,test=raw[split['selection']],raw[split['test']]
    rows,hashes,queries,cells,assigned= [],{},0,0,0
    for base in BASES:
        for budget in BUDGETS:
            parent_file=source/f'{base}_B{budget}_fit.json';parent=read(parent_file)
            for variant in VARIANTS:
                name=f'{base}_B{budget}_{variant}'
                fit_file,result_file=folder/(name+'_fit.json'),folder/(name+'.json')
                fit,result=read(fit_file),read(result_file)
                assert fit['source_fit_sha256']==digest(parent_file)
                assert result['fit_sha256']==digest(fit_file)
                program=RuleProgram.from_export(fit['program'])
                assert fit['program']==parent['programs'][variant]
                np.testing.assert_array_equal(fit['fitting_medians'],np.median(raw[split['train']],axis=0))
                active=sorted({a.feature for r in program.rules for a in r})
                assert fit['active_features']==active
                assert fit['unused_control']==(feature_key([f]) if (f:=next((i for i in range(raw.shape[1]) if i not in active),None)) is not None else None)
                selection_scores={}
                for key,record in fit['projections'].items():
                    assert key==feature_key(record['features'])
                    model=FeatureForgetter.from_export(record['envelope'])
                    assert model.original_==program
                    assert list(model.features_)==record['features']
                    assert verify_relation(model.certain_,program,equivalent=False,max_nodes=1000000)['status']=='proved'
                    assert verify_relation(program,model.possible_,equivalent=False,max_nodes=1000000)['status']=='proved'
                    for X in (selection,test):
                        bound=model.bounds(X);expected=enumeration_bounds(program,X,model.features_)
                        np.testing.assert_array_equal(bound['possible_head'],expected['possible_head'])
                        if model.audit_['certain_exact']:
                            np.testing.assert_array_equal(bound['certain_head'],expected['certain_head'])
                        else:
                            assert np.all(~bound['certain_head']|expected['certain_head'])
                        retained=model.bounds(X[:,model.retained_features_],retained_only=True)
                        np.testing.assert_array_equal(bound['prediction'],retained['prediction'])
                        queries+=len(X);cells+=expected['cells']
                    coverage=float(np.mean(model.predict(selection)>=0))
                    assert coverage==record['selection_coverage']
                    combined=tuple(a+b for a,b in zip(model.certain_.complexity,model.possible_.complexity))
                    assert list(combined)==record['combined_complexity']
                    selection_scores[key]=(-coverage,*combined[:2],tuple(record['features']))
                    if key==fit['unused_control']:
                        assert verify_relation(program,model.possible_,max_nodes=1000000)['status']=='proved'
                        assert verify_relation(program,model.certain_,max_nodes=1000000)['status']=='proved'
                singles=[feature_key([f]) for f in active] or ['none']
                first=min(singles,key=lambda k:selection_scores[k])
                first_features=fit['projections'][first]['features']
                pairs=[feature_key(first_features+[f]) for f in active if f not in first_features]
                second=min(pairs,key=lambda k:selection_scores[k]) if pairs else first
                expected_policies=dict(Fixed1=singles[0],Certainty1=first,
                    Fixed2=feature_key(active[:2]),Certainty2=second)
                assert fit['policies']==expected_policies
                expected_keys=set(singles+pairs+[expected_policies['Fixed2']])
                if fit['unused_control'] is not None:expected_keys.add(fit['unused_control'])
                assert set(fit['projections'])==expected_keys
                replay,predictions=evaluate(fit,test,split['teacher_test'])
                assert replay==result['rows']
                saved=np.load(folder/(name+'_predictions.npz'))
                assert set(saved.files)==set(predictions)
                for key,value in predictions.items():np.testing.assert_array_equal(value,saved[key])
                for row in replay:
                    assigned+=row['covered_n']
                    head_n=int((predictions['source']==program.head).sum())
                    default_n=len(test)-head_n
                    rows.append(dict(dataset=dataset,seed=seed,teacher=teacher_kind,noise=noise,
                        base=base,budget=budget,variant=variant,active_features=len(active),
                        source_head_n=head_n,source_default_n=default_n,
                        head_class_coverage=row['certain_head_n']/head_n if head_n else None,
                        default_class_coverage=row['certain_default_n']/default_n if default_n else None,**row))
                for p in (fit_file,result_file,folder/(name+'_predictions.npz')):
                    hashes[str(p.relative_to(ROOT))]=digest(p)
    print(stem,'audited',flush=True)
    return rows,hashes,queries,cells,assigned


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261016)
    means=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)))


def dataset_interval(frame,column):
    return interval(frame.groupby('dataset')[column].mean())


def table(name,caption,label,headers,rows):
    content=['\\begin{kbtable}','\\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
             '\\small','\\begin{adjustbox}{max width=\\linewidth}',
             '\\begin{tabular}{'+'l'*3+'r'*(len(headers)-3)+'}',
             '\\toprule',' & '.join(headers)+' \\\\','\\midrule']
    content+=[' & '.join(map(str,row))+' \\\\' for row in rows]
    content+=['\\bottomrule','\\end{tabular}','\\end{adjustbox}','\\end{kbtable}','']
    (OUT/name).write_text('\n'.join(content))


def summarize(frame,audit):
    policy=[]
    for _,row in frame.iterrows():
        for name in row.policies:
            count=1 if name.endswith('1') else 2
            if row.active_features>=count:
                policy.append(dict(row.drop(labels=['policies']),policy=name,requested=count))
    policies=pd.DataFrame(policy)
    policies.to_csv(OUT/'policies.csv',index=False)
    details,table_rows={},[]
    for variant in VARIANTS:
        details[variant]={}
        for name in POLICIES:
            sample=policies[(policies.variant==variant)&(policies.policy==name)].copy()
            sample['size_increased']=(sample.combined_tests>sample.source_tests).astype(float)
            coverage=dataset_interval(sample,'coverage')
            summary=dict(fits=len(sample),certified_coverage=coverage,
                exact_fraction=float(sample.certain_exact.mean()),
                combined_tests=float(sample.combined_tests.mean()),
                median_combined_tests=float(sample.combined_tests.median()),
                maximum_combined_tests=int(sample.combined_tests.max()),
                size_increased_fraction=float(sample.size_increased.mean()),
                median_disagreement=dataset_interval(sample,'median_disagreement'),
                existential_disagreement=dataset_interval(sample,'existential_disagreement'),
                head_class_coverage=dataset_interval(sample,'head_class_coverage'),
                default_class_coverage=dataset_interval(sample,'default_class_coverage'),
                median_compile_cpu=float(sample.compile_cpu.median()),
                maximum_compile_cpu=float(sample.compile_cpu.max()))
            details[variant][name]=summary
            table_rows.append(['Original' if variant=='Original' else 'Bounded',
                1 if name.endswith('1') else 2,'Fixed' if name.startswith('Fixed') else 'Selection',
                len(sample),f'{100*coverage["mean"]:.1f}',f'{100*summary["exact_fraction"]:.1f}',
                f'{summary["combined_tests"]:.2f}',
                f'{100*summary["size_increased_fraction"]:.1f}',
                f'{100*summary["median_disagreement"]["mean"]:.2f}'])
        for count in (1,2):
            selected=policies[(policies.variant==variant)&(policies.policy==f'Certainty{count}')].set_index(KEY)
            fixed=policies[(policies.variant==variant)&(policies.policy==f'Fixed{count}')].set_index(KEY)
            delta=(selected.coverage-fixed.coverage).groupby('dataset').mean()
            details[variant][f'coverage_gain_{count}']=interval(delta)
    table('forgetting.tex','Feature forgetting on the existing 648 fits, applied to Original and '
        'BoundedGeneralize separately. Fixed uses the lowest active indices; Selection maximizes '
        'certified coverage on selection inputs only. Two-feature rows require at least two active '
        'coordinates. Coverage is averaged equally over six datasets. Exact reports completed universal '
        'compilations; other cases use a conservative lower bound. L/U tests average both explicit '
        'formulas; Grow is the fraction whose combined count exceeds the source count. Fill disagrees '
        'with the fully observed source after fitting-median imputation; it is a forced prediction '
        'reference with a different coverage denominator. All certificates concern the source surrogate.',
        'tab:feature-forgetting',['Source','Missing','Policy','Fits','Certified (\\%)',
            'Exact (\\%)','L/U tests','Grow (\\%)','Fill disagree (\\%)'],table_rows)
    singles=frame[frame.active_singleton].groupby(KEY+['active_features'])[
        ['coverage','certain_exact','combined_tests','compile_cpu']].mean().reset_index()
    singles.to_csv(OUT/'all_singletons.csv',index=False)
    details['all_singletons']={variant:dict(fits=int((singles.variant==variant).sum()),
        coverage=dataset_interval(singles[singles.variant==variant],'coverage')) for variant in VARIANTS}
    details.update(audit=audit,primary='Original/Certainty1 certified coverage',
        projection_exact_fraction=float(frame.certain_exact.mean()),
        conservative_projections=int((~frame.certain_exact).sum()),
        assigned_test_queries=int(frame.covered_n.sum()),
        source_contradictions=int(frame.contradictions_source.sum()),
        median_projection_cpu=float(frame.compile_cpu.median()),
        maximum_combined_tests=int(frame.combined_tests.max()),
        unused_identity_controls=int(frame.unused_control.sum()))
    write_json(OUT/'numbers.json',details)
    original=details['Original'];single=original['Certainty1']['certified_coverage'];pair=original['Certainty2']['certified_coverage']
    gain=original['coverage_gain_1']
    (OUT/'numbers.tex').write_text(
        f'\\newcommand{{\\ForgettingCoverage}}{{{100*single["mean"]:.1f}}}\n'
        f'\\newcommand{{\\ForgettingGain}}{{{100*gain["mean"]:+.1f}}}\n'
        f'\\newcommand{{\\ForgettingPairCoverage}}{{{100*pair["mean"]:.1f}}}\n')
    (OUT/'findings.tex').write_text(
        f'For Original, selection-based single-feature forgetting certifies {100*single["mean"]:.1f}\\% '
        f'of test inputs, with a descriptive dataset-cluster interval of '
        f'[{100*single["low"]:.1f}, {100*single["high"]:.1f}], and a paired coverage gain of '
        f'{100*gain["mean"]:+.1f} points [{100*gain["low"]:+.1f}, {100*gain["high"]:+.1f}] over the fixed-index query. '
        f'The nested two-feature policy certifies {100*pair["mean"]:.1f}\\% on '
        f'{original["Certainty2"]["fits"]} eligible programs. '
        f'For the singleton selection policy, coverage conditional on the source head and default '
        f'classes is {100*original["Certainty1"]["head_class_coverage"]["mean"]:.1f}\\% and '
        f'{100*original["Certainty1"]["default_class_coverage"]["mean"]:.1f}\\%, respectively '
        '(using only nonempty class subsets). '
        f'All {details["assigned_test_queries"]:,} recorded assigned test queries agree with their '
        f'respective source programs. The independent checker replays {audit["retained_row_queries_checked"]:,} '
        f'selection/test retained-row queries across {audit["projection_rows"]:,} compiled queries. '
        f'{details["conservative_projections"]} compilations exhaust a fixed resource limit and are marked conservative. '
        f'The singleton selection policy has a larger combined representation than its source in '
        f'{100*original["Certainty1"]["size_increased_fraction"]:.1f}\\% of fits; '
        f'the median and maximum combined condition counts are '
        f'{original["Certainty1"]["median_combined_tests"]:.0f} and '
        f'{original["Certainty1"]["maximum_combined_tests"]:,}. '
        'The extension therefore establishes partial-input certificates, not uniformly shorter descriptions. '
        'These repeated uses of the same rows are implementation checks, not independent statistical observations.\n')
    return policies,details


def figure(policies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=policies[policies.variant=='Original']
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    for index,name in enumerate(('Fixed1','Certainty1','Certainty2')):
        ci=dataset_interval(values[values.policy==name],'coverage')
        axes[0].errorbar(100*ci['mean'],index,
            xerr=[[100*(ci['mean']-ci['low'])],[100*(ci['high']-ci['mean'])]],
            fmt='o',capsize=3,color='#087f8c')
    axes[0].set_yticks(range(3),['Fixed: one','Selection: one','Selection: two'])
    axes[0].set_xlabel('Certified test coverage (%)');axes[0].set_xlim(0,100);axes[0].invert_yaxis()
    sample=values[values.policy=='Certainty1']
    for exact,color,marker,label in ((True,'#087f8c','o','Exact'),(False,'#c47b11','x','Conservative')):
        subset=sample[sample.certain_exact==exact]
        axes[1].scatter(subset.source_tests,subset.combined_tests,s=10,alpha=.35,color=color,marker=marker,label=label)
    limit=sample.source_tests.max()+1
    axes[1].plot([0,limit],[0,limit],color='#888888',linewidth=.8)
    axes[1].set_xlim(0,limit);axes[1].set_yscale('symlog',linthresh=1)
    axes[1].set_ylim(0,max(2,float(sample.combined_tests.max())*1.4));axes[1].legend(fontsize=7,frameon=False)
    axes[1].set_xlabel('Source conditions');axes[1].set_ylabel('Combined L/U conditions')
    axes[0].set_title('(a) Missing-feature decisions');axes[1].set_title('(b) Representation cost')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.savefig(OUT/'forgetting.png',dpi=220);fig.savefig(OUT/'forgetting.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--wait-for-records',action='store_true',help='Audit completed folders while execution finishes.')
    parser.add_argument('--report-only',action='store_true',help='Regenerate reporting from audited rows after checking input hashes.')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    assert read(RUN/'manifest.json')['signature']==signature()
    if args.report_only:
        audit=read(OUT/'audit.json')
        for name,expected in read(OUT/'input_sha256.json').items():assert digest(ROOT/name)==expected
        frame=pd.read_csv(OUT/'metrics.csv',float_precision='round_trip')
        frame['policies']=frame.policies.map(ast.literal_eval)
        assert len(frame)==audit['projection_rows'] and not frame.contradictions_source.any()
        policies,details=summarize(frame,audit);figure(policies)
        print(json.dumps(details,indent=2));return
    if not args.wait_for_records:
        completion=read(RUN/'completion.json')
        assert completion['completed_source_programs']==1296 and completion['completed_settings']==72
        assert not completion['failures']
    tasks=[(n,s,t,z) for n in DATASETS for s in SEEDS for t in TEACHERS for z in NOISE]
    records,hashes,queries,cells,assigned=[],{},0,0,0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows,inputs,q,c,a in pool.map(audit_ready if args.wait_for_records else audit_folder,tasks):
            records.extend(rows);hashes.update(inputs);queries+=q;cells+=c;assigned+=a
    completion=read(RUN/'completion.json')
    assert completion['completed_source_programs']==1296 and completion['completed_settings']==72
    assert not completion['failures'] and len(records)==completion['projection_rows']
    frame=pd.DataFrame(records);frame.to_csv(OUT/'metrics.csv',index=False)
    assert not frame.contradictions_source.any()
    audit=dict(settings=72,source_programs=1296,projection_rows=len(frame),
        retained_row_queries_checked=queries,threshold_cell_families_enumerated=cells,
        assigned_test_queries=assigned,source_contradictions=0,
        source_data_hashes_verified=True,certificates_recomputed=True,
        global_sandwich_relations_reverified=True,independent_cell_oracle_checked=True,
        selection_policies_replayed=True,test_vectors_replayed=True,
        hidden_values_not_consulted=True,teacher_level_guarantee_claim=False,
        new_independent_dataset_claim=False)
    write_json(OUT/'audit.json',audit);write_json(OUT/'input_sha256.json',hashes)
    policies,details=summarize(frame,audit);figure(policies)
    print(json.dumps(details,indent=2))


if __name__=='__main__':main()
