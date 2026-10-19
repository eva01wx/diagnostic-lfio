"""Audit and report the explicit-rule-generalization extension."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lfio_rule_generalization_mpl')
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import StratifiedGroupKFold

from run_rule_generalization import (ROOT, OUT as RUN, BASES, VARIANTS, EPSILON,
    DELTA, SEEDS, signature, explanation_checks)
from run_prospective_revision import DATASETS, TEACHERS, NOISE, BUDGETS, load_data, digest, score, write_json
from src.rule_generalization import RuleProgram, verify_relation

OUT = ROOT/'docs/research/rule_generalization_2026-10-16'
KEY = ['dataset','seed','teacher','noise','base','budget']
LABELS = dict(Original='Original', BooleanExact='Boolean exact', SemanticExact='Semantic exact',
              TrainGeneralize='Grow constrained', EmpiricalGeneralize='Empirical gate',
              BoundedGeneralize='Simultaneous bound')


def read(path):
    return json.loads(path.read_text())


def proof_trace(audit, start):
    current = start
    count = 0
    for row in audit['trace']:
        before, after = (RuleProgram.from_export(row[k]) for k in ('before','after'))
        assert before == current
        assert after.complexity <= before.complexity
        proof = verify_relation(before, after, max_nodes=1000000)
        assert proof['status'] == 'proved'
        current = after; count += 1
    return current, count


def audit():
    assert read(RUN/'manifest.json')['signature'] == signature()
    completion = read(RUN/'completion.json')
    assert completion['completed_settings'] == completion['expected_settings'] == 72
    assert completion['completed_base_budget_fits'] == 648 and not completion['failures']
    rows, cost, cells, hashes = [], [], [], {}
    proof_count, path_count, prediction_count = 0, 0, 0
    for dataset in DATASETS:
        raw, labels, _ = load_data(dataset)
        _, groups = np.unique(raw, axis=0, return_inverse=True)
        for seed in SEEDS:
            folds = [b for _, b in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(raw,labels,groups)]
            expected_tr, expected_va, expected_te = np.concatenate(folds[2:]),folds[1],folds[0]
            for teacher in TEACHERS:
                for noise in NOISE:
                    folder = RUN/f'{dataset}_{seed}_{teacher}_noise{round(noise*100):02d}'
                    split = np.load(folder/'split.npz')
                    tr, selection, calibration, te = (split[k] for k in ('train','selection','calibration','test'))
                    for a,b in ((tr,expected_tr),(split['validation'],expected_va),(te,expected_te)):
                        np.testing.assert_array_equal(a,b)
                    a,b = next(StratifiedGroupKFold(2,shuffle=True,random_state=seed+31)
                        .split(raw[expected_va],split['teacher_validation'],groups[expected_va]))
                    np.testing.assert_array_equal(selection,expected_va[a])
                    np.testing.assert_array_equal(calibration,expected_va[b])
                    for first, second in ((tr,selection),(tr,calibration),(tr,te),
                                          (selection,calibration),(selection,te),(calibration,te)):
                        assert not set(groups[first]) & set(groups[second])
                    for method in BASES:
                        for budget in BUDGETS:
                            stem=f'{method}_B{budget}'
                            fit_file=folder/(stem+'_fit.json'); result_file=folder/(stem+'.json')
                            fit, result = read(fit_file), read(result_file)
                            assert result['fit_sha256'] == digest(fit_file)
                            original=RuleProgram.from_lfio(fit['source_export'],raw.shape[1])
                            programs={name:RuleProgram.from_export(value) for name,value in fit['programs'].items()}
                            assert programs['Original'] == original
                            assert original.complexity[0] <= budget
                            bounded=fit['bounded']; report=bounded['audit']
                            exact, count = proof_trace(report['exact'],original); proof_count += count
                            assert exact == programs['SemanticExact']
                            path=[RuleProgram.from_export(p) for p in bounded['path']]
                            assert path[0] == exact and [p.fingerprint for p in path] == report['path_sha256']
                            base_grow=original.predict(raw[tr]); gy=split['teacher_train']
                            for index, step in enumerate(report['path'],1):
                                assert RuleProgram.from_export(step['before']) == path[index-1]
                                expanded=RuleProgram.from_export(step['expanded'])
                                assert verify_relation(path[index-1],expanded,equivalent=False,
                                    max_nodes=1000000)['status']=='proved'
                                endpoint,count=proof_trace(step['exact'],expanded); proof_count+=count
                                assert endpoint == path[index]
                                risk=gy==1-original.head
                                assert risk.any()
                                harm=((base_grow==gy)&(endpoint.predict(raw[tr])!=gy)&risk).sum()/risk.sum()
                                assert harm <= 2*EPSILON+1e-12
                                path_count += 1
                            assert programs['TrainGeneralize'] == path[-1]
                            for variant, gate in (('EmpiricalGeneralize',fit['empirical']),('BoundedGeneralize',report)):
                                assert gate['calibration_used']
                                assert gate['path_sha256'] == report['path_sha256']
                                cp=original.predict(raw[calibration]); cy=split['teacher_calibration']
                                valid=[0]; family=max(1,len(path)-1)
                                assert gate['family_size']==family
                                for row in gate['calibration']:
                                    i=row['index']; prediction=path[i].predict(raw[calibration])
                                    risk=cy==1-original.head; n=int(risk.sum())
                                    harmful=int(((cp==cy)&(prediction!=cy)&risk).sum())
                                    upper=(float(beta.ppf(1-DELTA/family,harmful+1,n-harmful))
                                           if n and harmful<n else 1.)
                                    bound=(upper/2 if variant=='BoundedGeneralize' else harmful/(2*n) if n else .5)
                                    assert row['harmful']==harmful and row['opposite_class_samples']==n
                                    assert abs(row['balanced_error_increase_bound']-bound)<1e-12
                                    accept=bool(n and bound<=EPSILON+1e-12)
                                    assert row['accepted']==accept
                                    if accept: valid.append(i)
                                selected=min(valid,key=lambda i:(path[i].complexity,i))
                                assert gate['selected_index']==selected
                                assert programs[variant]==path[selected]
                            predictions=np.load(folder/(stem+'_predictions.npz'))
                            base_test=original.predict(raw[te]); ty=split['teacher_test']
                            assert {r['variant'] for r in result['rows']}==set(VARIANTS)
                            for row in result['rows']:
                                p=programs[row['variant']]; prediction=p.predict(raw[te])
                                np.testing.assert_array_equal(predictions[row['variant']],prediction)
                                assert abs(row['macro_fidelity']-score(ty,prediction))<1e-12
                                assert abs(row['fidelity']-np.mean(ty==prediction))<1e-12
                                assert tuple(row[k] for k in ('total_literals','rules','features','thresholds'))==p.complexity
                                for key,value in explanation_checks(p,raw[te]).items():
                                    assert row[key]==value
                                risk=ty==1-p.head
                                assert row['harmful']==int(((base_test==ty)&(prediction!=ty)&risk).sum())
                                if row['variant'] in ('BooleanExact','SemanticExact'):
                                    assert verify_relation(original,p,max_nodes=1000000)['status']=='proved'
                                    np.testing.assert_array_equal(prediction,base_test)
                                assert p.complexity <= original.complexity
                                rows.append(row); prediction_count+=1
                            cost.append(dict(dataset=dataset,seed=seed,teacher=teacher,noise=noise,
                                base=method,budget=budget,base_cpu=fit['base_cpu'],compression_cpu=fit['cpu_seconds']))
                            cells.append(dict(dataset=dataset,seed=seed,teacher=teacher,noise=noise,
                                base=method,budget=budget,path_candidates=len(path)-1,
                                empirical_index=fit['empirical']['selected_index'],
                                bounded_index=report['selected_index'],
                                proof_nodes=report['exact']['nodes'],unknown_proofs=report['exact']['unknown_proofs']))
                            for p in (fit_file,result_file,folder/(stem+'_predictions.npz')):
                                hashes[str(p.relative_to(ROOT))]=digest(p)
                    hashes[str((folder/'split.npz').relative_to(ROOT))]=digest(folder/'split.npz')
    checks=dict(settings=72,base_budget_fits=648,variant_rows=len(rows),prediction_vectors_replayed=prediction_count,
        exact_proof_steps_replayed=proof_count,approximate_steps_replayed=path_count,
        global_equivalence_reverified=True,cumulative_grow_harm_checked=True,calibration_bounds_recomputed=True,
        grouped_partitions_disjoint=True,source_and_data_hashes_verified=True,
        test_predictions_only_after_selection=True,population_sampling_assumptions_not_inferred=True)
    write_json(OUT/'audit.json',checks); write_json(OUT/'input_sha256.json',hashes)
    frame, times, policies=map(pd.DataFrame,(rows,cost,cells))
    frame.to_csv(OUT/'metrics.csv',index=False);times.to_csv(OUT/'runtime.csv',index=False)
    policies.to_csv(OUT/'decisions.csv',index=False)
    return frame,times,policies,checks


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(16102026)
    means=rng.choice(values,(10000,len(values)),replace=True).mean(axis=1)
    return dict(mean=float(values.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)))


def table(name,caption,label,headers,rows):
    content=('\\begin{kbtable}\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
        '\\small\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n\\toprule\n'
        +' & '.join(headers)+r' \\'+'\n\\midrule\n'
        +'\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
        +'\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{kbtable}\n')
    (OUT/name).write_text(content)


def summarize(frame,times,policies,checks):
    baseline=frame[frame.variant=='Original'].set_index(KEY)
    rows,contrasts,details=[],[],{}
    for variant in VARIANTS:
        sample=frame[frame.variant==variant].set_index(KEY)
        joined=sample.join(baseline,rsuffix='_original')
        joined['condition_reduction']=1-joined.total_literals/joined.total_literals_original.replace(0,np.nan)
        joined['condition_reduction']=joined.condition_reduction.fillna(0.)
        joined['fidelity_change']=joined.macro_fidelity-joined.macro_fidelity_original
        reduction=interval(joined.groupby('dataset').condition_reduction.mean())
        fidelity=interval(joined.groupby('dataset').fidelity_change.mean())
        details[variant]=dict(condition_reduction=reduction,macro_fidelity_change=fidelity,
            compressed_fits=int((joined.total_literals<joined.total_literals_original).sum()),
            nonzero_macro_fidelity_changes=int((joined.fidelity_change.abs()>1e-12).sum()),
            test_losses_above_two_points=int(((joined.teacher_classes==2)&(joined.fidelity_change < -EPSILON-1e-12)).sum()))
        contrasts.append(dict(variant=variant,**{f'reduction_{k}':v for k,v in reduction.items()},
            **{f'fidelity_{k}':v for k,v in fidelity.items()}))
        rows.append([LABELS[variant],f'{sample.total_literals.mean():.2f}',f'{sample.rules.mean():.2f}',
            f'{sample.features.mean():.2f}',f'{sample.thresholds.mean():.2f}',
            f'{100*reduction["mean"]:.1f}',f'{100*fidelity["mean"]:+.2f}',
            f'[{100*fidelity["low"]:+.2f}, {100*fidelity["high"]:+.2f}]',
            details[variant]['compressed_fits']])
    pd.DataFrame(contrasts).to_csv(OUT/'paired.csv',index=False)
    summary=(frame.groupby(['base','variant','dataset'])
        [['total_literals','rules','features','thresholds','macro_fidelity','mean_checks']].mean()
        .groupby(['base','variant']).mean().reset_index())
    summary.to_csv(OUT/'summary.csv',index=False)
    table('generalization.tex',
        'Explicit rule generalization on the six previously studied datasets with new seeds 7001--7003. '
        'Each row uses the same 648 base/budget fits (three learners, 72 teacher settings, three budgets). '
        'Reduction averages per-fit relative condition savings, counting zero-size originals as zero savings. '
        'Macro fidelity changes are paired with each original; descriptive intervals resample six dataset means. '
        'Boolean and semantic exact stages preserve global predictions. The two-point tolerance applies to '
        'two-class balanced error under the stated calibration assumptions, not a deterministic test-set guarantee.',
        'tab:rule-generalization',['Variant','Tests','Bodies','Features','Cuts','Saved (\\%)',
            '$\\Delta$ fidelity (pp)','95\\% interval','Smaller /648'],rows)
    per_base=[]
    for base in BASES:
        base_frame=frame[frame.base==base]
        before=base_frame[base_frame.variant=='Original'].set_index(KEY)
        for variant in ('BooleanExact','SemanticExact','BoundedGeneralize'):
            after=base_frame[base_frame.variant==variant].set_index(KEY)
            change=after.macro_fidelity-before.macro_fidelity
            savings=(1-after.total_literals/before.total_literals.replace(0,np.nan)).fillna(0.)
            per_base.append([base,LABELS[variant],f'{before.total_literals.mean():.2f}',
                f'{after.total_literals.mean():.2f}',f'{savings.mean()*100:.1f}',f'{change.mean()*100:+.2f}'])
    table('per_base.tex','Generalization effects by base learner, across 216 fits each. '
        'The numerical semantic stage uses threshold implications that a Boolean-only simplifier cannot exploit.',
        'tab:generalization-base',['Base','Stage','Original tests','Final tests','Saved (\\%)','$\\Delta$ fidelity (pp)'],per_base)
    details.update(audit=checks,
        approximate_paths_with_candidates=int((policies.path_candidates>0).sum()),
        empirical_nonexact_selections=int((policies.empirical_index>0).sum()),
        bounded_nonexact_selections=int((policies.bounded_index>0).sum()),
        median_compression_cpu=float(times.compression_cpu.median()),
        median_base_cpu=float(times.base_cpu.median()),
        zero_size_originals=int((baseline.total_literals==0).sum()),
        epsilon=EPSILON,delta=DELTA)
    write_json(OUT/'numbers.json',details)
    exact=details['SemanticExact']; bounded=details['BoundedGeneralize']
    findings=(f'The numerical semantic exact stage reduces total conditions in {exact["compressed_fits"]} of 648 fits, '
        f'with mean per-fit savings of {100*exact["condition_reduction"]["mean"]:.1f}\\% '
        f'[{100*exact["condition_reduction"]["low"]:.1f}, {100*exact["condition_reduction"]["high"]:.1f}]. '
        f'All exact-stage test predictions are unchanged, and global interval proofs are replayed. '
        f'The grow-only path admits approximate candidates for {details["approximate_paths_with_candidates"]} fits; '
        f'the empirical and simultaneous-bound gates select a nonexact model in '
        f'{details["empirical_nonexact_selections"]} and {details["bounded_nonexact_selections"]} fits, respectively. '
        f'The simultaneous-bound variant saves {100*bounded["condition_reduction"]["mean"]:.1f}\\% of conditions '
        f'with a paired test macro-fidelity change of {100*bounded["macro_fidelity_change"]["mean"]:+.2f} points '
        f'[{100*bounded["macro_fidelity_change"]["low"]:+.2f}, {100*bounded["macro_fidelity_change"]["high"]:+.2f}]. '
        f'These are repeated-split results on known datasets, not human-comprehension evidence or new external validation.\n')
    (OUT/'findings.tex').write_text(findings)
    (OUT/'numbers.tex').write_text(
        f'\\newcommand{{\\GeneralizationSaved}}{{{100*bounded["condition_reduction"]["mean"]:.2f}}}\n'
        f'\\newcommand{{\\GeneralizationChange}}{{{100*bounded["macro_fidelity_change"]["mean"]:+.3f}}}\n')
    return details


def figures(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    variants=VARIANTS[1:]
    baseline=frame[frame.variant=='Original'].set_index(KEY)
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
    for j,variant in enumerate(variants):
        sample=frame[frame.variant==variant].set_index(KEY)
        savings=(1-sample.total_literals/baseline.total_literals.replace(0,np.nan)).fillna(0.)
        delta=sample.macro_fidelity-baseline.macro_fidelity
        for ax,values in zip(axes,(savings,delta)):
            ci=interval(values.groupby('dataset').mean())
            ax.errorbar(ci['mean']*100,j,xerr=[[100*(ci['mean']-ci['low'])],[100*(ci['high']-ci['mean'])]],
                fmt='o',color='#0b7285',capsize=3,markersize=4)
    axes[0].set_yticks(range(len(variants)),[LABELS[v] for v in variants])
    axes[1].set_yticks(range(len(variants)),[])
    for ax in axes:
        ax.invert_yaxis();ax.axvline(0,color='#899199',linewidth=.8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_xlabel('Conditions saved (%)');axes[1].set_xlabel('Macro fidelity change (pp)')
    axes[0].set_title('(a) Structural compression');axes[1].set_title('(b) Paired test effect')
    fig.savefig(OUT/'generalization.png',dpi=220);fig.savefig(OUT/'generalization.pdf')
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    frame,times,policies,checks=audit()
    details=summarize(frame,times,policies,checks);figures(frame)
    print(json.dumps(details,indent=2))


if __name__=='__main__':
    main()
