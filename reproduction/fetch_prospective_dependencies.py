"""Fetch or verify SHA-256-pinned October data and official FastSRS sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def obtain(path, url, expected, verify_only):
    if not path.exists():
        if verify_only:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Download hash mismatch: '+url)
        path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    data = ROOT/'data/research/prospective_2026-10-15'
    sources = json.loads((data/'sources.json').read_text())
    for name, record in sources.items():
        folder = data/name
        archive = folder/'source.zip'
        obtain(archive, record['download'], record['archive_sha256'], args.verify_only)
        if not args.verify_only:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(folder)
            if name == 'raisin':
                with zipfile.ZipFile(folder/'Raisin_Dataset.zip') as bundle:
                    bundle.extractall(folder)
    pin = json.loads((ROOT/'docs/research/major_revision_2026-10-15/fastsrs_pin.json').read_text())
    for name, expected in pin['files'].items():
        url = f"https://raw.githubusercontent.com/mmolerous/FastSRS/{pin['commit']}/{name}"
        obtain(ROOT/'.tools/baselines/fastsrs'/name, url, expected, args.verify_only)
    neural = json.loads((ROOT/'docs/research/major_revision_2026-10-15/neurules_pin.json').read_text())
    for name, expected in neural['files'].items():
        url = f"https://raw.githubusercontent.com/Schascha1/neurules/{neural['commit']}/{name}"
        obtain(ROOT/'.tools/baselines/neurules'/name, url, expected, args.verify_only)
    print('Verified six independent source archives, official FastSRS and pinned NeuRules sources.')


if __name__ == '__main__':
    main()
