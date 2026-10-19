"""Package feature forgetting while preserving every preceding dated release."""
import hashlib
import json
from pathlib import Path
import zipfile

import package_major_revision as legacy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'docs/research/feature_forgetting_2026-10-16'
PREVIOUS=ROOT/'docs/research/rule_generalization_2026-10-16'
SKIP={'package_manifest.json','validation.json','delivery_manifest.json','relocation_verification.json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reports(folder):
    return {str(p.relative_to(ROOT)):p.read_bytes() for p in folder.iterdir()
            if p.is_file() and p.suffix!='.zip' and p.name not in SKIP}


def archive(name,files):
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files)
    manifest={n:sha(data) for n,data in sorted(files.items())}
    files=dict(files,**{'BUNDLE_CONTENTS_SHA256.json':(json.dumps(manifest,indent=2)+'\n').encode()})
    target=OUT/name
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for filename,data in sorted(files.items()):bundle.writestr(filename,data)
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
        for filename,expected in manifest.items():assert sha(bundle.read(filename))==expected
    return dict(file=name,sha256=sha(target.read_bytes()),bytes=target.stat().st_size,
                entries=len(files),entries_verified=True)


def emit(name,files):
    name=name.replace('KBS_','KBS_Feature_Forgetting_').replace('2026-10-15','2026-10-16')
    files=dict(files)
    if 'LaTeX_sources' in name:
        files['BUILD.txt']=b'''Feature-forgetting revision, 16 October 2026.
All scientific inputs are flattened; compile each document with Tectonic or
pdflatex/bibtex. LFIO_Negation_Paper.tex is the main CAS double-column manuscript,
LFIO_Negation_Paper_review.tex is the single-column review, and
LFIO_Development_Supplement.tex preserves historical development evidence.
Forgetting/necessary-sufficient duality are classical, not new theories here.
'''
    elif 'Reproducibility_Code' in name:
        names=['run_rule_generalization.py','generate_rule_generalization.py','package_rule_generalization.py',
            'tests/test_rule_generalization.py','tests/test_lfio_generalized.py',
            'docs/research/RULE_GENERALIZATION_PROTOCOL_2026-10-16.md',
            'docs/research/RULE_GENERALIZATION_2026-10-16.md',
            'run_feature_forgetting.py','generate_feature_forgetting.py','package_feature_forgetting.py',
            'research_baselines/forgetting_oracle.py','tests/test_feature_forgetting.py',
            'docs/research/FEATURE_FORGETTING_PROTOCOL_2026-10-16.md',
            'docs/research/FEATURE_FORGETTING_AMENDMENT_2026-10-16.md',
            'docs/research/FEATURE_FORGETTING_2026-10-16.md']
        files.update({n:(ROOT/n).read_bytes() for n in names})
        files.update(reports(PREVIOUS));files.update(reports(OUT))
        initial=ROOT/'docs/research/archive/feature_forgetting_initial_2026-10-16'
        files.update({str(p.relative_to(ROOT)):p.read_bytes() for p in initial.iterdir() if p.is_file()})
        files['README_MAJOR_REVISION.md']=b'''# Numerical feature-forgetting release

Extract this code ZIP and KBS_Feature_Forgetting_Experiment_Records_2026-10-16.zip
together. Python 3.12 and requirements-major-revision.txt define the dependencies.
FastSRS, six dataset archives, and their licenses/provenance are included.
Fetch pinned NeuRules sources with fetch_prospective_dependencies.py; those
unlicensed files are not redistributed. Install Tectonic separately.

python fetch_prospective_dependencies.py --verify-only
python generate_feature_forgetting.py --workers 16
python generate_rule_generalization.py
python generate_major_revision.py --replay-workers 16
python -m pytest -q tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
python build_paper.py --all

See docs/research/FEATURE_FORGETTING_2026-10-16.md for the partial-input API,
source-relative guarantees, amended execution, and fresh-run instructions.
This reuses known source programs; it is not new external validation.
The initial identity-control limitation and its narrow implementation amendment
are preserved separately. Current outputs use feature_forgetting_20261016_v2.
The complete workspace test report is retained; only relevant tests are bundled
because historical application datasets are outside this release.
'''
    else:
        for folder in ('rule_generalization_20261016','feature_forgetting_20261016','feature_forgetting_20261016_v2'):
            files.update({str(p.relative_to(ROOT)):p.read_bytes() for p in (ROOT/'results'/folder).rglob('*')
                          if p.is_file() and p.suffix in ('.json','.npz','.log')})
        files.update(reports(PREVIOUS));files.update(reports(OUT))
        files['README_EXPERIMENT_RECORDS.md']=b'''# Allocation, generalization and forgetting records

The old frozen cohorts retain their identities under prospective_20261015 and
rule_generalization_20261016. Current feature-forgetting certificates and partial
prediction records are under feature_forgetting_20261016_v2. The initial run
remains under feature_forgetting_20261016, with the unused-feature identity
failure and its amendment explained in the code/documentation package.

Current execution uses 1,296 source programs, not 1,296 new fits. The source-
relative lower/upper formulas, selection choices and fitting medians were saved
before test inference. Independent threshold-cell checks and all saved vectors
are reproducible. Repeated row queries are checks, not independent samples.
'''
    return archive(name,files)


def main():
    assert json.loads((OUT/'audit.json').read_text())['source_programs']==1296
    previous=legacy.archive
    try:
        legacy.archive=emit
        packages=[legacy.source_package(),legacy.code_package(),legacy.record_package()]
    finally:legacy.archive=previous
    (OUT/'package_manifest.json').write_text(json.dumps(packages,indent=2)+'\n')
    print(json.dumps(packages,indent=2))


if __name__=='__main__':main()
