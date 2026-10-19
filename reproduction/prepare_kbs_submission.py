"""Adapt audited tables and create a flat Elsevier LaTeX source archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'docs/research/paper_2026-09-24'
OUTPUT = ROOT / 'docs/research/kbs_2026-09-24'


def prepare_tables():
    """Change float wrappers and caption placement, preserving every table cell."""
    directory = OUTPUT / 'tables'
    directory.mkdir(parents=True, exist_ok=True)
    record = {}
    for source in sorted((EVIDENCE / 'tables').glob('*.tex')):
        original = source.read_text(encoding='utf-8')
        caption = re.search(r'\\caption\{[^\n]*\}\n\\label\{[^\n]*\}\n', original)
        if caption is None:
            raise ValueError(f'Expected one caption and label in {source}')
        adapted = original[:caption.start()] + original[caption.end():]
        adapted = adapted.replace('\\begin{table}[htbp]', '\\begin{kbtable}')
        adapted = adapted.replace('\\end{table}', '\\end{kbtable}')
        if source.stem == 'ablations':
            adapted = adapted.replace('{kbtable}', '{kbsmalltable}')
        adapted = adapted.replace('\\centering\n', '\\centering\n' + caption.group(), 1)
        old_cells = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', original, re.S).group()
        new_cells = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', adapted, re.S).group()
        assert old_cells == new_cells, source
        (directory / source.name).write_text(adapted, encoding='utf-8')
        record[source.name] = hashlib.sha256(old_cells.encode()).hexdigest()
    (OUTPUT / 'table_content_sha256.json').write_text(json.dumps(record, indent=2) + '\n')
    return directory


def package_sources():
    """Flatten input paths for Editorial Manager and include vendored template sources."""
    prepare_tables()
    source = (ROOT / 'LFIO_Negation_Paper.tex').read_text()
    files = {}
    # Collect exactly the referenced scientific inputs, including new follow-ups.
    pattern = r'(\\(?:input|includegraphics)(?:\[[^\]]*\])?\{)([^}]+)(\})'
    def flatten(match):
        path = ROOT / match.group(2)
        if not path.is_file():
            raise ValueError(f'Missing manuscript input: {path}')
        name = ('table_' if path.parent.name == 'tables' else '') + path.name
        data = path.read_bytes()
        if name in files and files[name] != data:
            raise ValueError(f'Conflicting flattened name: {name}')
        files[name] = data
        return match.group(1) + name + match.group(3)
    source = re.sub(pattern, flatten, source)
    files['LFIO_Negation_Paper.tex'] = source.encode()
    for name in ('LFIO_Negation_Paper_review.tex', 'references.bib', 'cas-dc.cls',
                 'cas-sc.cls', 'cas-common.sty', 'elsarticle-num.bst', 'KBS_Highlights.txt',
                 'KBS_Highlights.docx'):
        files[name] = (ROOT / name).read_bytes()
    for name in ('README', 'manifest.txt', 'provenance.json'):
        files['CAS_' + name] = (ROOT / 'docs/vendor/els-cas' / name).read_bytes()
    # Include BibTeX output as a convenience; .bib + .bst remain authoritative.
    for name in ('LFIO_Negation_Paper.bbl', 'LFIO_Negation_Paper_review.bbl'):
        path = ROOT / name
        if path.exists():
            files[name] = path.read_bytes()
    files['BUILD.txt'] = (
        'Main file: LFIO_Negation_Paper.tex (CAS 2.4, cas-dc, double column).\n'
        'Review file: LFIO_Negation_Paper_review.tex (cas-sc, single column, review spacing).\n'
        'Both use elsarticle-num.bst, configured for numbered citations.\n'
        'Use Tectonic, latexmk -pdf, or pdflatex / bibtex / pdflatex / pdflatex.\n'
        'Set the chosen main file in Overleaf. All dependencies are at one folder level.\n'
        'KBS_Highlights.txt and KBS_Highlights.docx are separate editable highlights documents.\n'
        'Code and recorded predictions are provided in the accompanying reproducibility archives.\n'
        'Funding, competing-interest and contribution declarations still require author-supplied text.\n'
        'This is a formatting package, not a submitted or accepted article.\n'
    ).encode()
    archive = OUTPUT / 'KBS_LaTeX_sources.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(files.items()):
            assert '/' not in name, name
            bundle.writestr(name, data)
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    (OUTPUT / 'submission_sources_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Created {archive.relative_to(ROOT)} ({len(files)} files, no subfolders).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args()
    if args.tables_only:
        print(f'Prepared tables in {prepare_tables().relative_to(ROOT)}')
    else:
        package_sources()
