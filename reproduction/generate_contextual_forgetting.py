"""Replay and report the contextual/cost-aware forgetting extension."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/lfio_contextual_mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from fractions import Fraction

import numpy as np
import pandas as pd

from run_contextual_forgetting import (ROOT,OUT as RUN,REPORT as OUT,OLD,PARENT,
    read,signature,evaluate,vectors_expected,controlled)
from run_prospective_revision import load_data,digest,write_json
from src.contextual_forgetting import ContextualForgetter,DifferenceBound
from src.rule_generalization import RuleProgram
from research_baselines.contextual_oracle import constrained_bounds


def check_witnesses(program,row,constraints,record):
    for label in ('head','default'):
        witness=record[label+'_witness']
        assert (witness is not None)==(record[label+'_status']=='SAT')
        if witness is None:continue
        values=[Fraction(float(v)) for v in row]
        for feature,value in witness.items():values[int(feature)]=Fraction(value)
        for bound in constraints:
            left=0 if bound.left is None else values[bound.left]
            right=0 if bound.right is None else values[bound.right]
            assert left-right < bound.bound if bound.strict else left-right <= bound.bound
        hit=any(all((values[a.feature]>Fraction(a.threshold))==a.greater for a in r) for r in program.rules)
        assert hit==(label=='head')


def replay_path(profile):
    """Reconstruct decisions from the audited candidate ledger, not the selector."""
    bank={tuple(r['features']):r for r in profile['candidates']}
    active=profile['active_features'];cost=profile['costs']
    targets=sorted({r['target'] for r in profile['choices']})
    outputs={}
    for policy in ('Fixed','Frequency','Impact','Coverage','CostAware'):
        for q in targets:
            current=()
            while True:
                options=[]
                for f in active:
                    if f in current:continue
                    proposal=tuple(sorted((*current,f)))
                    if proposal not in bank:continue
                    record=bank[proposal]
                    if record['coverage'] < q:continue
                    if policy=='Fixed':rank=(f,)
                    elif policy=='Frequency':rank=(profile['frequency'][str(f)],f)
                    elif policy=='Impact':rank=(profile['singleton_impact'][str(f)],f)
                    elif policy=='Coverage':rank=(-record['coverage'],f)
                    else:rank=(max(0.,bank[current]['coverage']-record['coverage'])/cost[f],
                               -cost[f],-record['coverage'],f)
                    options.append((rank,proposal))
                if not options:break
                current=min(options)[1]
            outputs[policy,q]=current
    return outputs


def audit_fit(path):
    fit=read(path);name=path.name.removesuffix('_fit.json');result_path=path.with_name(name+'.json')
    result=read(result_path);assert result['fit_sha256']==digest(path)
    old_path=ROOT/fit['source_file'];assert digest(old_path)==fit['source_sha256']
    old_fit=read(old_path);assert fit['program']==old_fit['program']
    source=RuleProgram.from_export(fit['program'])
    raw,_,_=load_data(fit['dataset']);split=np.load(PARENT/path.parent.name/'split.npz')
    selection,test=raw[split['selection']],raw[split['test']]
    audited={};oracle_count=0;oracle_skipped=0
    for profile_name,profile in fit['profiles'].items():
        for candidate in profile['candidates']:
            hidden=tuple(candidate['features'])
            if hidden not in audited:
                model=ContextualForgetter().fit(source,hidden);pred=model.predict(selection)
                check_witnesses(source,selection[0],[],model.query(selection[0]))
                audited[hidden]=float(np.mean(pred>=0))
                cells=math.prod(1+len({a.threshold for r in source.rules for a in r if a.feature==f}) for f in hidden)
                if cells<=4096:
                    expected=vectors_expected(source,selection[:1],hidden)
                    assert pred[0]<0 or pred[0]==expected[0]
                    oracle_count+=1
                else:oracle_skipped+=1
            assert audited[hidden]==candidate['coverage']
            assert candidate['hidden_cost']==sum(profile['costs'][f] for f in hidden)
        # If the common candidate budget binds, order-dependent allocation can
        # prevent later paths from using the final ledger. Fresh replay checks
        # those fits with the exact original budget and candidate sequence.
        if any(r['candidate_budget_exhausted'] for r in profile['choices']):
            from src.forgetting_selection import select_features
            replay=select_features(source,selection,profile['costs'])
            assert json.loads(json.dumps(replay))==profile
        else:
            replay=replay_path(profile)
            for choice in profile['choices']:
                assert tuple(choice['features'])==replay[choice['policy'],choice['target']]
    replay,vectors=evaluate(fit,selection,test,old_fit)
    assert replay=={k:result[k] for k in replay}
    saved=np.load(path.with_name(name+'_predictions.npz'))
    assert set(saved.files)==set(vectors)
    for key,value in vectors.items():np.testing.assert_array_equal(saved[key],value)
    common={k:fit[k] for k in ('dataset','seed','teacher','noise','base','budget')}
    rows=[dict(**common,**r) for r in result['rows']]
    comparisons=[dict(**common,**r) for r in result['comparisons']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in (path,result_path,path.with_name(name+'_predictions.npz'))}
    return rows,comparisons,hashes,dict(candidate_subsets=len(audited),candidate_oracles=oracle_count,
        candidate_oracle_caps_skipped=oracle_skipped,selection_rows=len(selection),test_rows=len(test),
        selection_cpu=fit['selection_cpu'])


def cluster(frame,column):
    values=frame.groupby('dataset')[column].mean().to_numpy()
    rng=np.random.default_rng(20261016)
    means=rng.choice(values,(10000,len(values)),replace=True).mean(axis=1)
    return dict(mean=float(values.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)))


def report():
    frame=pd.read_csv(OUT/'policies.csv',float_precision='round_trip')
    queries=pd.read_csv(OUT/'queries.csv',float_precision='round_trip')
    groups=[]
    for (profile,target,policy),data in frame.groupby(['profile','target','policy']):
        groups.append(dict(profile=profile,target=float(target),policy=policy,n=len(data),
            coverage=cluster(data,'test_coverage'),hidden_cost_fraction=cluster(data,'hidden_cost_fraction'),
            hidden_count=cluster(data,'hidden_count'),zero_hiding_fraction=float((data.hidden_count==0).mean()),
            test_shortfall_fraction=float(data.test_target_shortfall.mean()),
            class0_coverage=cluster(data,'class0_coverage'),class1_coverage=cluster(data,'class1_coverage'),
            budget_exhausted_fraction=float(data.candidate_budget_exhausted.mean())))
    primary=frame[(frame.profile=='heterogeneous')&(frame.target==.8)]
    paired=primary.pivot(index=['dataset','seed','teacher','noise','base','budget'],columns='policy',values='hidden_cost_fraction').reset_index()
    paired['gain']=paired.CostAware-paired.Coverage
    bench=read(RUN/'benchmark.json')
    untraced=read(RUN/'benchmark_untraced.json')
    assert [(r['source_file'],r['projection_key']) for r in untraced]==[(r['source_file'],r['projection_key']) for r in bench]
    benchmark=dict(n=len(bench),expanded_tests_min=min(r['expanded_tests'] for r in bench),
        expanded_tests_max=max(r['expanded_tests'] for r in bench),
        global_median_peak=float(np.median([r['global']['python_peak_bytes'] for r in bench])),
        query_median_peak=float(np.median([r['query']['python_peak_bytes'] for r in bench])),
        median_peak_ratio=float(np.median([r['query']['python_peak_bytes']/r['global']['python_peak_bytes'] for r in bench])),
        global_median_total_cpu=float(np.median([r['global']['total_cpu'] for r in untraced])),
        query_median_total_cpu=float(np.median([r['query']['total_cpu'] for r in untraced])),
        global_median_warm_cpu=float(np.median([r['global']['warm_cpu'] for r in untraced])),
        query_median_warm_cpu=float(np.median([r['query']['warm_cpu'] for r in untraced])),
        cpu_tracing=False,cpu_repetitions=3,
        query_median_patterns=float(np.median([r['query']['stats']['solved_patterns'] for r in bench])))
    numbers=dict(policies=groups,primary_cost_gain=cluster(paired,'gain'),benchmark=benchmark,
        queries=dict(n=len(queries),old_conservative=int((~queries.old_exact).sum()),
            unknown=int((queries.unknown_patterns>0).sum()),gained_row_queries=int(queries.gained.sum()),
            lost_row_queries=int(queries.lost.sum()),
            median_patterns=float(queries.solved_patterns.median())),audit=read(OUT/'audit.json'))
    write_json(OUT/'numbers.json',numbers)
    lookup={(r['profile'],r['target'],r['policy']):r for r in groups}
    cost=lookup['heterogeneous',.8,'CostAware'];cover=lookup['heterogeneous',.8,'Coverage'];gain=numbers['primary_cost_gain']
    macros=dict(ContextCost=f"{cost['hidden_cost_fraction']['mean']*100:.1f}",
        ContextCoverage=f"{cost['coverage']['mean']*100:.1f}",ContextGain=f"{gain['mean']*100:.1f}",
        ContextGainLow=f"{gain['low']*100:.1f}",ContextGainHigh=f"{gain['high']*100:.1f}",
        ContextQueryGained=str(numbers['queries']['gained_row_queries']))
    (OUT/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    lines=[r'\begin{kbtable}',r'\caption{Cost-constrained hiding on 648 Original programs. Heterogeneous controlled costs, selection coverage requirement $q=0.8$; means weight the six datasets equally and include empty-subset fallbacks. Hidden cost is a fraction of total active-feature cost.}',
        r'\label{tab:contextual-cost}',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Policy & Hidden cost (\%) & Features & Test coverage (\%)\\',r'\midrule']
    for policy in ('Fixed','Frequency','Impact','Coverage','CostAware'):
        r=lookup['heterogeneous',.8,policy]
        lines.append(f"{policy} & {r['hidden_cost_fraction']['mean']*100:.1f} & {r['hidden_count']['mean']:.2f} & {r['coverage']['mean']*100:.1f}\\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{kbtable}'])
    (OUT/'costs.tex').write_text('\n'.join(lines)+'\n')
    query=numbers['queries']
    findings=(f"CostAware hides {cost['hidden_cost_fraction']['mean']*100:.1f}\\% of active-coordinate cost versus "
        f"{cover['hidden_cost_fraction']['mean']*100:.1f}\\% for Coverage, a paired difference of "
        f"{gain['mean']*100:.1f} points (descriptive dataset-cluster interval "
        f"[{gain['low']*100:.1f},{gain['high']*100:.1f}]). Test certified coverage is "
        f"{cost['coverage']['mean']*100:.1f}\\%; the selection constraint does not guarantee the test target, "
        f"which is missed by {cost['test_shortfall_fraction']*100:.1f}\\% of fits. "
        f"Empty hiding sets occur in {cost['zero_hiding_fraction']*100:.1f}\\% of fits. "
        "These controlled coordinate costs are not measured acquisition or privacy costs.\n\n"
        f"Across {query['n']:,} previous Original projections, residual queries encounter "
        f"{query['unknown']} unresolved projections and add {query['gained_row_queries']:,} assigned test-row queries "
        f"relative to the bounded compiler, with {query['lost_row_queries']:,} lost assignments. "
        "These are repeated evaluation events, not independent observations. "
        f"On the 32 deliberately large compiled representations ({benchmark['expanded_tests_min']}--"
        f"{benchmark['expanded_tests_max']} combined tests), median Python allocation peaks are "
        f"{benchmark['global_median_peak']/1024:.1f} KiB for compilation and {benchmark['query_median_peak']/1024:.1f} KiB for queries. "
        f"Median setup-plus-first-batch CPU is {benchmark['global_median_total_cpu']:.4f} versus "
        f"{benchmark['query_median_total_cpu']:.4f} seconds; warm-batch CPU is "
        f"{benchmark['global_median_warm_cpu']:.5f} versus {benchmark['query_median_warm_cpu']:.5f} seconds. "
        "CPU uses three fresh repetitions without tracing, separately from allocation measurement. "
        "The batch contains at most 64 test rows. Python peaks omit native allocations, "
        "and this stress subset does not establish a general speed advantage.\n")
    (OUT/'findings.tex').write_text(findings)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    for policy,marker in (('Coverage','o'),('CostAware','s')):
        part=[lookup['heterogeneous',q,policy] for q in (.7,.8,.9)]
        axes[0].plot([r['coverage']['mean']*100 for r in part],[r['hidden_cost_fraction']['mean']*100 for r in part],marker=marker,label=policy)
        for q,r in zip((.7,.8,.9),part):axes[0].annotate(f'{q:.1f}',(r['coverage']['mean']*100,r['hidden_cost_fraction']['mean']*100),xytext=(3,4),textcoords='offset points',fontsize=7)
    axes[0].set(xlabel='Test certified coverage (%)',ylabel='Hidden active cost (%)',title='Controlled heterogeneous costs')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].boxplot([[r[k]['python_peak_bytes']/1024 for r in bench] for k in ('global','query')],tick_labels=['Global envelope','Residual query'],showfliers=False)
    axes[1].set(yscale='log',ylabel='Peak Python allocation (KiB)',title='32 large representations');axes[1].grid(axis='y',alpha=.2)
    fig.savefig(OUT/'contextual.pdf');fig.savefig(OUT/'contextual.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--report-only',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not args.report_only:
        assert read(RUN/'manifest.json')['signature']==signature()
        assert read(RUN/'completion.json')==dict(settings=72,programs=648,failures=[])
        paths=sorted(RUN.glob('*/*_Original_fit.json'));assert len(paths)==648
        rows=[];queries=[];hashes={};audits=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index,(r,q,h,a) in enumerate(pool.map(audit_fit,paths)):
                rows.extend(r);queries.extend(q);hashes.update(h);audits.append(a)
                if index%54==0:print('Audited',index+1,'/ 648',flush=True)
        controlled_records=read(RUN/'controlled.json')
        for record in controlled_records['random']:
            program=RuleProgram.from_export(record['program']);bounds=[DifferenceBound(*b) for b in record['constraints']]
            check_witnesses(program,record['row'],bounds,record['query'])
            oracle=constrained_bounds(program,record['row'],record['features'],bounds)
            assert oracle==record['oracle'] and record['query']['prediction']==oracle['prediction']
        from src.rule_generalization import Atom
        controlled_source=RuleProgram(((Atom(0,0.,True),),),1,3)
        for record in controlled_records['cases']:
            check_witnesses(controlled_source,[record['y'],record['y'],0.],
                [DifferenceBound(*b) for b in record['constraints']],record)
        for q in queries:
            q.update(q.pop('stats'))
        frame=pd.DataFrame(rows);pd.DataFrame(queries).to_csv(OUT/'queries.csv',index=False)
        frame.drop(columns=['stats']).to_csv(OUT/'policies.csv',index=False)
        write_json(OUT/'fit_audits.json',audits);write_json(OUT/'input_sha256.json',hashes)
        write_json(OUT/'audit.json',dict(status='passed',settings=72,programs=len(paths),policy_rows=len(rows),
            projection_rows=len(queries),candidate_subsets=sum(a['candidate_subsets'] for a in audits),
            candidate_cell_oracles=sum(a['candidate_oracles'] for a in audits),
            candidate_oracle_caps_skipped=sum(a['candidate_oracle_caps_skipped'] for a in audits),
            random_constrained_oracles=len(controlled_records['random']),controlled_rows=len(controlled_records['cases']),
            contradictions=int(frame.contradictions.sum()),source_hashes_verified=True,selection_choices_replayed=True,
            saved_vectors_replayed=True,hidden_values_not_consulted=True,teacher_guarantee=False,
            rational_witnesses_independently_checked=True,
            new_independent_data=False))
    report();print(json.dumps(read(OUT/'numbers.json')['primary_cost_gain']))


if __name__=='__main__':main()
