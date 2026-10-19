"""Package the current explicit-generalization revision without rewriting old ZIPs."""
import hashlib
import json
from pathlib import Path
import zipfile

import package_major_revision as legacy

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'docs/research/rule_generalization_2026-10-16'
RUN = ROOT/'results/rule_generalization_20261016'
SKIP = {'package_manifest.json','validation.json','delivery_manifest.json','relocation_verification.json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reports():
    return {str(p.relative_to(ROOT)):p.read_bytes() for p in OUT.iterdir()
            if p.is_file() and p.suffix!='.zip' and p.name not in SKIP}


def archive(name, files):
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files)
    manifest={n:sha(data) for n,data in sorted(files.items())}
    files=dict(files,**{'BUNDLE_CONTENTS_SHA256.json':(json.dumps(manifest,indent=2)+'\n').encode()})
    path=OUT/name
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for filename,data in sorted(files.items()): bundle.writestr(filename,data)
    with zipfile.ZipFile(path) as bundle:
        assert bundle.testzip() is None
        for filename,expected in manifest.items(): assert sha(bundle.read(filename))==expected
    return dict(file=name,sha256=sha(path.read_bytes()),bytes=path.stat().st_size,
                entries=len(files),entries_verified=True)


def emit(name, files):
    files=dict(files)
    name=name.replace('2026-10-15','2026-10-16')
    if 'LaTeX_sources' in name:
        files['BUILD.txt']=b'''Current revision: explicit rule generalization, 16 October 2026.
Main: LFIO_Negation_Paper.tex. Review: LFIO_Negation_Paper_review.tex.
Historical development: LFIO_Development_Supplement.tex.
All scientific inputs are flattened. Compile with Tectonic or pdflatex/bibtex.
The new generalization extension reuses six datasets with new seeds, and is
separate from the earlier frozen allocation experiment. Classical logical
operations and binomial limits are not claimed as newly invented theories.
'''
    elif 'Reproducibility_Code' in name:
        extra=['src/rule_generalization.py','src/lfio_generalized.py',
               'run_rule_generalization.py','generate_rule_generalization.py','package_rule_generalization.py',
               'tests/test_rule_generalization.py','tests/test_lfio_generalized.py',
               'docs/research/RULE_GENERALIZATION_2026-10-16.md',
               'docs/research/RULE_GENERALIZATION_PROTOCOL_2026-10-16.md']
        files.update({n:(ROOT/n).read_bytes() for n in extra})
        files.update(reports())
        files['README_MAJOR_REVISION.md']=b'''# Explicit rule-generalization revision

Extract this code ZIP and KBS_Experiment_Records_2026-10-16.zip together.
Use Python 3.12 and requirements-major-revision.txt. Fetch pinned NeuRules
sources using fetch_prospective_dependencies.py (unlicensed files are not bundled).
FastSRS, six dataset archives and baseline provenance are included.

python fetch_prospective_dependencies.py --verify-only
python generate_rule_generalization.py
python generate_major_revision.py --replay-workers 16
python -m pytest -q tests/test_rule_generalization.py tests/test_lfio_generalized.py
python build_paper.py --all

Install Tectonic separately. The flat source ZIP also builds all three PDFs.
See docs/research/RULE_GENERALIZATION_2026-10-16.md for the public API, results,
calibration assumptions and fresh training in a separate directory. This is
a new-split extension of known datasets, not new external validation.
The code ZIP contains focused extension tests and retained regression reports;
it omits historical application datasets required by the full workspace suite.
'''
    else:
        files.update({str(p.relative_to(ROOT)):p.read_bytes() for p in RUN.rglob('*')
                      if p.is_file() and p.suffix in ('.json','.npz','.log')})
        files.update(reports())
        files['README_EXPERIMENT_RECORDS.md']=b'''# Allocation and explicit-generalization records

The earlier frozen cohort remains under results/prospective_20261015.
New seeds 7001--7003 are under results/rule_generalization_20261016: 72 teacher
settings, 648 base/budget fits, six variants (3,888 rows). Separate selection
and calibration partitions are retained. Fit and calibration choices were
persisted before test inference. Exact stages preserve global predictions;
approximate stages have separate empirical and binomial-bound interpretations.
The records retain proof traces, paths, calibration decisions, predictions,
timings, warnings and source locks. Study-level population guarantees and
human understanding are not inferred from these benchmark observations.
'''
    return archive(name,files)


def main():
    assert json.loads((OUT/'audit.json').read_text())['variant_rows']==3888
    original=legacy.archive
    try:
        legacy.archive=emit
        packages=[legacy.source_package(),legacy.code_package(),legacy.record_package()]
    finally:
        legacy.archive=original
    (OUT/'package_manifest.json').write_text(json.dumps(packages,indent=2)+'\n')
    print(json.dumps(packages,indent=2))


if __name__=='__main__':
    main()
