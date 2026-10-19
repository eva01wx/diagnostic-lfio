"""Package the current manuscript and independent-cohort reproducibility release."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'docs/research/major_revision_2026-10-15'
RUN = ROOT/'results/prospective_20261015'
PATTERN = r'(\\(?:input|includegraphics)(?:\[[^\]]*\])?\{)([^}]+)(\})'
DOCS = ('LFIO_Negation_Paper.tex', 'LFIO_Negation_Paper_review.tex', 'LFIO_Development_Supplement.tex')
TEMPLATE = ('references.bib', 'cas-dc.cls', 'cas-sc.cls', 'cas-common.sty', 'elsarticle-num.bst')
SKIP_REPORTS = {'package_manifest.json', 'delivery_manifest.json', 'relocation_verification.json', 'validation.json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive(name, files):
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files)
    manifest = {n: sha(v) for n, v in sorted(files.items())}
    files = dict(files, **{'BUNDLE_CONTENTS_SHA256.json': (json.dumps(manifest, indent=2)+'\n').encode()})
    target = OUT/name
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n, data in sorted(files.items()):
            z.writestr(n, data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n, expected in manifest.items():
            assert sha(z.read(n)) == expected
    return dict(file=name, sha256=sha(target.read_bytes()), bytes=target.stat().st_size,
                entries=len(files), entries_verified=True)


def scientific_dependencies():
    paths = set()
    pending = [ROOT/n for n in DOCS]
    while pending:
        source = pending.pop()
        for _, name, _ in re.findall(PATTERN, source.read_text()):
            path = ROOT/name
            if not path.is_file():
                raise FileNotFoundError(path)
            if path not in paths:
                paths.add(path)
                if path.suffix == '.tex' and path.name not in DOCS:
                    pending.append(path)
    return paths


def source_package():
    files, names = {}, {document: document for document in DOCS}

    def add(path):
        key = str(path.relative_to(ROOT))
        if key in names:
            return names[key]
        name, data = path.name, path.read_bytes()
        if name in files and files[name] != data:
            name = path.parent.name+'_'+name
        if name in files and files[name] != data:
            name = sha(data)[:8]+'_'+name
        names[key] = name
        if path.suffix == '.tex' and path.name not in DOCS:
            text = data.decode()
            text = re.sub(PATTERN, lambda m: m[1]+add(ROOT/m[2])+m[3], text)
            data = text.encode()
        files[name] = data
        return name

    for document in DOCS:
        text = (ROOT/document).read_text()
        files[document] = re.sub(PATTERN, lambda m: m[1]+add(ROOT/m[2])+m[3], text).encode()
    for name in (*TEMPLATE, 'KBS_Highlights.txt', 'KBS_Highlights.docx'):
        files[name] = (ROOT/name).read_bytes()
    for name in ('README', 'manifest.txt', 'provenance.json'):
        files['CAS_'+name] = (ROOT/'docs/vendor/els-cas'/name).read_bytes()
    for name in DOCS:
        bbl = ROOT/Path(name).with_suffix('.bbl')
        if bbl.exists():
            files[bbl.name] = bbl.read_bytes()
    files['BUILD.txt'] = b'''Current cohort: protocol frozen 2026-10-15 UTC; completed 2026-10-16 CST.
Main: LFIO_Negation_Paper.tex (CAS double column).
Review: LFIO_Negation_Paper_review.tex (CAS single column).
Historical development: LFIO_Development_Supplement.tex.
All scientific inputs are at one level. Use Tectonic or pdflatex/bibtex.
The main manuscript's independent evidence is separate from development data.
Source-file basename conflicts are resolved without changing table contents.
Separate dated code and experiment-record ZIPs contain executable evidence.
'''
    assert all('/' not in n for n in files)
    return archive('KBS_LaTeX_sources_2026-10-15.zip', files)


def code_package():
    paths = {ROOT/n for n in (*DOCS, *TEMPLATE, 'build_paper.py', 'prepare_kbs_submission.py',
        'audit_recent_manuscript.py', 'run_prospective_revision.py', 'certify_real_revision.py',
        'generate_major_revision.py', 'fetch_prospective_dependencies.py', 'package_major_revision.py',
        'requirements-major-revision.txt', 'KBS_Highlights.txt', 'KBS_Highlights.docx',
        'tests/conftest.py', 'tests/test_diagnostic_revision.py',
        'docs/research/MAJOR_REVISION_PROTOCOL_2026-10-15.md', 'docs/research/REPRODUCE.md',
        'docs/research/MAJOR_REVISION_2026-10-16.md')}
    paths.update(scientific_dependencies())
    paths.update((ROOT/'src').glob('*.py'))
    for n in ('adapters.py', 'brcg.py', 'prospective.py', '__init__.py'):
        p = ROOT/'research_baselines'/n
        if p.is_file():
            paths.add(p)
    paths.update(p for p in (ROOT/'research_baselines/vendor_brcg').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths.update(p for p in (ROOT/'data/research/prospective_2026-10-15').rglob('*') if p.is_file())
    paths.update(p for p in OUT.iterdir() if p.is_file() and p.suffix != '.zip' and p.name not in SKIP_REPORTS)
    paths.update(p for p in (ROOT/'docs/vendor/els-cas').iterdir() if p.is_file())
    pin = json.loads((OUT/'fastsrs_pin.json').read_text())
    paths.update(ROOT/'.tools/baselines/fastsrs'/name for name in pin['files'])
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    files['README_MAJOR_REVISION.md'] = b'''# Current independent-cohort release

Extract this ZIP and KBS_Experiment_Records_2026-10-15.zip into one directory.
Python 3.12 and requirements-major-revision.txt define the tested environment.
Run fetch_prospective_dependencies.py to obtain hash-pinned NeuRules files.
FastSRS's MIT-licensed official implementation and six CC BY 4.0 datasets are
included with source provenance; NeuRules is downloaded rather than bundled.

python fetch_prospective_dependencies.py --verify-only
python generate_major_revision.py --replay-workers 16
python -m pytest -q tests/test_diagnostic_revision.py
python build_paper.py --all

Install Tectonic separately, or compile the flat source ZIP with a LaTeX editor.
The complete workspace regression report is retained as validation_tests.txt;
this package includes the new algorithm's focused tests, not every historical
application dataset or development experiment needed by the old full suite.
The separate development PDF can be rebuilt from retained scientific inputs.
See docs/research/REPRODUCE.md for fresh training in a separate directory.
'''
    return archive('KBS_Reproducibility_Code_2026-10-15.zip', files)


def record_package():
    paths = [p for p in RUN.rglob('*') if p.is_file() and p.suffix in ('.json', '.npz', '.log')]
    paths += [p for p in OUT.iterdir() if p.is_file() and p.suffix != '.zip' and p.name not in SKIP_REPORTS]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    files['README_EXPERIMENT_RECORDS.md'] = b'''# Independent-cohort records

72 teacher settings, eight model families, 1,728 budget rows, including native
infeasibility. One clean-RF Climate test teacher is single-class: the frozen
observed-class macro score and the two-class-only sensitivity are distinguished.
All fitting-only internal probes are reproducibly replayed, with 216 decisions.
24 real-task head solves give six optima and valid bounds where available.
The original certificate runner's JSON-boolean failure and serialization-only
source amendment are preserved. Timeouts do not imply optimality.
Earlier September studies are not pooled with these observations.
'''
    return archive('KBS_Experiment_Records_2026-10-15.zip', files)


def main():
    audit = json.loads((OUT/'audit.json').read_text())
    assert audit['settings'] == 72 and audit['budget_rows'] == 1728
    packages = [source_package(), code_package(), record_package()]
    (OUT/'package_manifest.json').write_text(json.dumps(packages, indent=2)+'\n')
    print(json.dumps(packages, indent=2))


if __name__ == '__main__':
    main()
