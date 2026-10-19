"""Package the contextual extension without rewriting preceding releases."""
import json

import package_major_revision as legacy
import package_feature_forgetting as previous

ROOT=legacy.ROOT
OUT=ROOT/'docs/research/contextual_forgetting_2026-10-16'


def archive(name,files):
    old_out=previous.OUT
    try:
        previous.OUT=OUT
        return previous.archive_original(name,files)
    finally:
        previous.OUT=old_out


def emit(name,files):
    name=name.replace('KBS_Feature_Forgetting_','KBS_Contextual_Forgetting_')
    files=dict(files)
    if 'LaTeX_sources' in name:
        files['CURRENT_RELEASE.md']=(ROOT/'docs/research/CURRENT_RELEASE.md').read_bytes()
        # Provide the requested submission filename as a self-contained flat
        # document, while retaining the established manuscript filename.
        files['elsarticle-template-num.tex']=files['LFIO_Negation_Paper.tex']
        bbl=ROOT/'elsarticle-template-num.bbl'
        if bbl.exists():files[bbl.name]=bbl.read_bytes()
        for vendor_name in ('README','manifest.txt','PROVENANCE.md','elsarticle.dtx','elsarticle.ins'):
            files['ELSARTICLE_'+vendor_name]=(ROOT/'docs/vendor/elsarticle'/vendor_name).read_bytes()
        files['BUILD.txt']=b'''Current release: contextual-forgetting-2026-10-16.
Title: Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting.
Release directory: docs/research/contextual_forgetting_2026-10-16/.
Submission: elsarticle-template-num.tex, elsarticle review,12pt, single column.
Numeric references: bibliographystyle elsarticle-num.
LFIO_Negation_Paper.tex is the same manuscript under its compatibility name;
LFIO_Negation_Paper_review.tex is a compatibility wrapper for that review layout.
The historical development supplement retains its separate CAS layout.
Compile the chosen flat TeX document using Tectonic.
CURRENT_RELEASE.md lists workspace paths for the separate code/record packages.
Classical logical foundations are attributed; numerical difference bounds,
infeasibility, residual queries and controlled input costs define the extension.
'''
    elif 'Reproducibility_Code' in name:
        names=['README.md','elsarticle-template-num.tex','docs/research/CURRENT_RELEASE.md',
            'docs/research/SUBMISSION_CHECKLIST.md','docs/research/KBS_FORMAT.md',
            'docs/research/KBS_COVER_LETTER_DRAFT.md','docs/research/AUTHOR_DECLARATIONS_DRAFT.md',
            'run_contextual_forgetting.py','generate_contextual_forgetting.py',
            'benchmark_contextual_forgetting.py','package_contextual_forgetting.py',
            'research_baselines/contextual_oracle.py','tests/test_contextual_forgetting.py',
            'docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md',
            'docs/research/CONTEXTUAL_FORGETTING_PROTOCOL_2026-10-16.md',
            'docs/research/CONTEXTUAL_BENCHMARK_NOTE_2026-10-16.md',
            'docs/research/FORGETTING_LITERATURE_2026-10-16.md',
            'docs/research/FORGETTING_LITERATURE_2026-10-16.bib']
        files.update({n:(ROOT/n).read_bytes() for n in names})
        files.update({str(p.relative_to(ROOT)):p.read_bytes()
                      for p in (ROOT/'docs/vendor/elsarticle').iterdir() if p.is_file()})
        files.update(previous.reports(OUT))
        files['README_MAJOR_REVISION.md']=b'''# Contextual and cost-aware feature forgetting

Current release: contextual-forgetting-2026-10-16.
Title: Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting.
Package index: docs/research/CURRENT_RELEASE.md.
Release directory: docs/research/contextual_forgetting_2026-10-16/.

Extract this code ZIP and KBS_Contextual_Forgetting_Experiment_Records_2026-10-16.zip
together. Python 3.12 and requirements-major-revision.txt define the environment.
Pinned baseline sources/data and all earlier extension provenance are retained.
NeuRules is not bundled; use fetch_prospective_dependencies.py if needed.

python generate_contextual_forgetting.py --workers 12
python -m pytest -q tests/test_contextual_forgetting.py tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
python build_paper.py --all

The four core extension modules contain 47 tests; all five bundled modules
contain 54 tests and require pinned NeuRules sources for the diagnostic module.
The historical full workspace report covers 167 tests;
historical application tests/datasets outside the research release are omitted.
The submission entry is elsarticle-template-num.tex (elsarticle review,12pt)
with numeric references. Build output elsarticle-template-num.pdf is preferred;
the development supplement retains its separate historical CAS layout.
See docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md for guarantees, limits,
fresh execution and the separation of untraced CPU from Python allocation peaks.
This is reuse of known datasets, not additional independent validation.
'''
    else:
        files['docs/research/CURRENT_RELEASE.md']=(ROOT/'docs/research/CURRENT_RELEASE.md').read_bytes()
        files.update({str(p.relative_to(ROOT)):p.read_bytes()
                      for p in (ROOT/'results/contextual_forgetting_20261016').rglob('*')
                      if p.is_file() and p.suffix in ('.json','.npz','.log')})
        files.update(previous.reports(OUT))
        files['README_EXPERIMENT_RECORDS.md']=b'''# Frozen allocation, generalization and forgetting records

Current release: contextual-forgetting-2026-10-16.
Title: Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting.
Package index: docs/research/CURRENT_RELEASE.md.
Release directory: docs/research/contextual_forgetting_2026-10-16/.

The new extension is results/contextual_forgetting_20261016. Earlier cohorts
and the initial/amended global feature-forgetting records retain their names.
The extension uses 648 existing Original source programs; choices/candidates
were saved before test inference. Controlled cost scenarios and known synthetic
background constraints are distinct from verified real-world acquisition costs
or domain assumptions. All saved vectors and choices are replayable.
benchmark.json contains traced allocation measurements; benchmark_untraced.json
contains separate CPU repetitions on exactly the same 32 stress cases.
'''
    return archive(name,files)


def main():
    assert json.loads((OUT/'audit.json').read_text())['status']=='passed'
    original_archive=previous.archive
    major_archive=legacy.archive
    original_template=legacy.TEMPLATE
    try:
        legacy.TEMPLATE=(*original_template,'elsarticle.cls')
        previous.archive_original=original_archive
        previous.archive=emit
        legacy.archive=previous.emit
        packages=[legacy.source_package(),legacy.code_package(),legacy.record_package()]
    finally:
        previous.archive=original_archive
        legacy.archive=major_archive
        legacy.TEMPLATE=original_template
        del previous.archive_original
    (OUT/'package_manifest.json').write_text(json.dumps(packages,indent=2)+'\n')
    print(json.dumps(packages,indent=2))


if __name__=='__main__':main()
