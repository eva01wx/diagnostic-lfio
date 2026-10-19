"""Check repository file hashes and frozen experiment record checksums."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text())
    for name, record in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('Missing file or size mismatch: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Checksum mismatch: ' + name)
    payloads = 0
    archives = list((ROOT / 'artifacts').glob('*.zip'))
    for archive in sorted(archives):
        with zipfile.ZipFile(archive) as bundle:
            expected = json.loads(bundle.read('BUNDLE_CONTENTS_SHA256.json'))
            names = [x.filename for x in bundle.infolist() if not x.is_dir()]
            if len(names) != len(set(names)) or set(names) != set(expected) | {'BUNDLE_CONTENTS_SHA256.json'}:
                raise ValueError('Archive member list mismatch: ' + archive.name)
            for name, digest in expected.items():
                if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                    raise ValueError('Archive payload mismatch: ' + archive.name + ': ' + name)
                payloads += 1
    print(json.dumps({'status': 'passed', 'copied_files': len(manifest['files']),
                      'archive_payloads': payloads, 'release': manifest['release']}, indent=2))


if __name__ == '__main__':
    main()
