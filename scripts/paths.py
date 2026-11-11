"""Project paths and explicit provenance checks for the script-folder migration.

Data and result digests always require an exact byte match. For pre-migration
source digests only, a recorded before/after pair AND the local original source
snapshot are required. This never rewrites a historical experiment protocol.
"""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / 'scripts'
LAYOUT_FILE = SCRIPTS / 'source_layout.json'


def _layout():
    return json.loads(LAYOUT_FILE.read_text())


def resolve_file(path):
    """Resolve a historical flat script path to its categorized source file."""
    path = Path(path)
    absolute = path.absolute()
    if absolute.parent == SCRIPTS and path.name not in ('paths.py', 'source_layout.json'):
        entry = _layout()['files'].get(path.name)
        if entry:
            return PROJECT_ROOT / entry['path']
    return path


def source_names(names):
    """Include layout infrastructure in the source seal of new experiments."""
    return tuple(dict.fromkeys((*names, 'paths.py', 'source_layout.json')))


def hash_matches(path, expected):
    """Check exact bytes, or the explicitly recorded source-only migration.

    If the current source is edited after migration, the historical fallback
    fails. If the snapshot is missing or modified, it also fails. Unrelated
    historical source changes are not accepted by this migration.
    """
    current = resolve_file(path)
    if not current.is_file():
        return False
    actual = hashlib.sha256(current.read_bytes()).hexdigest()
    if actual == expected:
        return True
    if current.suffix != '.py':
        return False
    layout = _layout()
    entry = layout['files'].get(current.name)
    if not entry or current.absolute() != PROJECT_ROOT / entry['path']:
        return False
    if expected != entry['before_sha256'] or actual != entry['after_sha256']:
        return False
    original = PROJECT_ROOT / layout['snapshot'] / 'scripts' / current.name
    return original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest() == expected
