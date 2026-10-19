"""Verify and safely restore frozen experiment records."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'artifacts/experiment_records_2026-10-16.zip'
INDEX = 'BUNDLE_CONTENTS_SHA256.json'


def restore(destination, check_only=False):
    destination = destination.resolve()
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text())
    record = manifest['files'][str(ARCHIVE.relative_to(ROOT))]
    if (ARCHIVE.stat().st_size != record['bytes'] or
            hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != record['sha256']):
        raise ValueError('Record archive checksum mismatch')
    with zipfile.ZipFile(ARCHIVE) as bundle:
        infos = bundle.infolist()
        names = [x.filename for x in infos if not x.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        expected = json.loads(bundle.read(INDEX))
        if set(names) != set(expected) | {INDEX}:
            raise ValueError('Archive member list differs from its checksum index')
        missing = []
        existing = 0
        for name, digest in expected.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('Unsafe archive member: ' + name)
            target = destination.joinpath(*relative.parts)
            if target.is_symlink() or not target.resolve().is_relative_to(destination):
                raise ValueError('Destination escapes through a symbolic link: ' + name)
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Member checksum mismatch: ' + name)
            if target.exists():
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError('Refusing to replace a conflicting file: ' + str(target))
                existing += 1
            else:
                missing.append((name, target))
        # Complete preflight before writing any payload. The archive index is
        # preserved separately in provenance/ and is not extracted over code.
        if not check_only:
            for name, target in missing:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(bundle.read(name))
    return {'mode': 'check' if check_only else 'restore',
            'verified_payloads': len(expected), 'existing_identical': existing,
            'missing' if check_only else 'restored': len(missing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without extracting records.')
    parser.add_argument('--destination', type=Path, default=ROOT / 'reproduction')
    args = parser.parse_args()
    print(json.dumps(restore(args.destination, args.check), indent=2))


if __name__ == '__main__':
    main()
