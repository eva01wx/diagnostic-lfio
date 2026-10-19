"""Build the manuscript with Tectonic and a project-local Windows font cache."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', action='store_true', help='Build the compatibility review entry (elsarticle review,12pt).')
    parser.add_argument('--all', action='store_true', help='Build the submission entry, compatibility PDFs and development supplement.')
    parser.add_argument('--supplement', action='store_true', help='Build the chronological development supplement.')
    args = parser.parse_args()
    local = ROOT / '.tools/tectonic' / ('tectonic.exe' if os.name == 'nt' else 'tectonic')
    executable = str(local) if local.is_file() else shutil.which('tectonic')
    if not executable:
        raise SystemExit('Install Tectonic or place its executable under .tools/tectonic/. See docs/research/REPRODUCE.md.')
    evidence = ROOT / 'docs/research/paper_2026-09-24'
    current = ROOT/'docs/research/major_revision_2026-10-15'
    if 'major_revision_2026-10-15' in (ROOT/'LFIO_Negation_Paper.tex').read_text():
        if not all((current/name).exists() for name in ('numbers.tex', 'audit.json', 'main.tex', 'certificates.tex')):
            raise SystemExit('Run generate_major_revision.py to audit and generate the current evidence before building.')
    elif not all((evidence / name).exists() for name in ('numbers.tex', 'strengthening_numbers.tex',
               'strengthening_findings.tex', 'ablation_findings.tex', 'sensitivity_findings.tex',
               'followup_findings.tex', 'case_studies.tex', 'work_findings.tex', 'mdl_findings.tex')):
        raise SystemExit('Generate the dated development and strengthening evidence before building; see docs/research/REPRODUCE.md.')
    env = os.environ.copy()
    # Keep downloaded TeX resources writable in restricted project environments.
    env['XDG_CACHE_HOME'] = str(ROOT / '.tools/tectonic/cache')
    if os.name == 'nt' and not env.get('FONTCONFIG_FILE'):
        fonts = Path(env.get('SystemRoot', 'C:/Windows')) / 'Fonts'
        cache = ROOT / '.tools/tectonic/font-cache'
        cache.mkdir(parents=True, exist_ok=True)
        config = cache.parent / 'fonts.conf'
        config.write_text('<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
                          '<fontconfig><dir>'+escape(fonts.as_posix())+'</dir><cachedir>'
                          +escape(cache.as_posix())+'</cachedir></fontconfig>\n', encoding='utf-8')
        env['FONTCONFIG_FILE'] = str(config)
    if (evidence/'tables').is_dir():
        from prepare_kbs_submission import prepare_tables
        prepare_tables()
    targets = ['elsarticle-template-num.tex', 'LFIO_Negation_Paper.tex']
    if args.all:
        targets.append('LFIO_Negation_Paper_review.tex')
        if (ROOT/'LFIO_Development_Supplement.tex').exists():
            targets.append('LFIO_Development_Supplement.tex')
    elif args.review:
        targets = ['LFIO_Negation_Paper_review.tex']
    elif args.supplement:
        targets = ['LFIO_Development_Supplement.tex']
    for target in targets:
        result = subprocess.run([executable, '--keep-logs', '--keep-intermediates', target], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
