"""Check folder migration without weakening saved-artifact integrity checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import paths


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.current = self.root/'scripts/models/example.py'
        self.original = self.root/'.local_archive/snapshot/scripts/example.py'
        for file in (self.current,self.original):file.parent.mkdir(parents=True,exist_ok=True)
        self.current.write_bytes(b'new location, equivalent code')
        self.original.write_bytes(b'original source')
        self.before = hashlib.sha256(self.original.read_bytes()).hexdigest()
        self.after = hashlib.sha256(self.current.read_bytes()).hexdigest()
        manifest=self.root/'scripts/source_layout.json'
        manifest.write_text(json.dumps({'snapshot':'.local_archive/snapshot','files':{
            'example.py':{'path':'scripts/models/example.py','before_sha256':self.before,'after_sha256':self.after}}}))
        for name,value in [('PROJECT_ROOT',self.root),('SCRIPTS',self.root/'scripts'),('LAYOUT_FILE',manifest)]:
            p=patch.object(paths,name,value);p.start();self.addCleanup(p.stop)

    def test_old_and_current_path_verify_recorded_migration(self):
        self.assertEqual(paths.resolve_file(self.root/'scripts/example.py'),self.current)
        self.assertTrue(paths.hash_matches(self.root/'scripts/example.py',self.before))
        self.assertTrue(paths.hash_matches(self.current,self.before))
        self.assertTrue(paths.hash_matches(self.current,self.after))

    def test_changed_current_source_is_rejected(self):
        self.current.write_bytes(b'changed implementation')
        self.assertFalse(paths.hash_matches(self.current,self.before))
        self.assertFalse(paths.hash_matches(self.current,self.after))

    def test_missing_or_changed_snapshot_is_rejected(self):
        self.original.write_bytes(b'changed archive')
        self.assertFalse(paths.hash_matches(self.current,self.before))
        self.original.unlink()
        self.assertFalse(paths.hash_matches(self.current,self.before))

    def test_artifacts_and_unregistered_versions_require_exact_bytes(self):
        data=self.root/'predictions.csv';data.write_bytes(b'unchanged')
        digest=hashlib.sha256(data.read_bytes()).hexdigest()
        self.assertTrue(paths.hash_matches(data,digest))
        data.write_bytes(b'changed')
        self.assertFalse(paths.hash_matches(data,digest))
        self.assertFalse(paths.hash_matches(self.current,'0'*64))
        other=self.root/'example.py';other.write_bytes(self.current.read_bytes())
        self.assertFalse(paths.hash_matches(other,self.before))


if __name__=='__main__':unittest.main()
