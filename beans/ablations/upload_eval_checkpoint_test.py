import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import upload_eval_checkpoint as uploader


class FakeApi:
    def __init__(self, private=True, existing=(), bad_sizes=False):
        self.private = private
        self.existing = existing
        self.bad_sizes = bad_sizes
        self.commits = []

    def create_repo(self, *args, **kwargs):
        self.create_kwargs = kwargs

    def repo_info(self, *args, **kwargs):
        if kwargs.get("files_metadata"):
            siblings = []
            for operation in self.commits[-1]["operations"]:
                src = operation.path_or_fileobj
                size = len(src) if isinstance(src, bytes) else Path(src).stat().st_size
                siblings.append(SimpleNamespace(rfilename=operation.path_in_repo, size=size + self.bad_sizes))
            return SimpleNamespace(siblings=siblings)
        return SimpleNamespace(private=self.private, sha="a" * 40)

    def list_repo_files(self, *args, **kwargs):
        return self.existing

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(oid="b" * 40)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / "beans/checkpoints/pi05_yam_beans0922_ab_snap_token_mlp3_a9align/token_test_B/1000"
        params = self.ckpt / "params"
        (params / "d").mkdir(parents=True)
        (params / "_METADATA").write_text("{}")
        (params / "manifest.ocdbt").write_bytes(b"manifest")
        (params / "d/abc123").write_bytes(b"fake parameter chunk")
        (self.ckpt / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 123}))
        (self.ckpt / "train_state").mkdir()
        (self.ckpt / "train_state/not_for_upload").write_text("optimizer")

    def publish(self, api, public=False):
        ckpt, config, exp, prefix, files, sizes = uploader.inspect_checkpoint(self.ckpt)
        manifest = {"experiment_name": exp, "checkpoint_step": 1000}
        return uploader.publish(api, SimpleNamespace, "test/beans-eval", prefix, files, sizes, manifest, public)

    def test_params_only_and_params_argument(self):
        result = uploader.inspect_checkpoint(self.ckpt / "params")
        self.assertEqual(result[0], self.ckpt)
        self.assertFalse(any("train_state" in name for name in result[4]))
        self.assertIn("params/d/abc123", result[4])

    def test_unfinished_checkpoint_rejected(self):
        (self.ckpt / "_CHECKPOINT_METADATA").write_text("{}")
        with self.assertRaisesRegex(ValueError, "no commit timestamp"):
            uploader.inspect_checkpoint(self.ckpt)

    def test_temporary_checkpoint_rejected(self):
        tmp_ckpt = self.ckpt.with_name("1000.orbax-checkpoint-tmp-0")
        self.ckpt.rename(tmp_ckpt)
        with self.assertRaisesRegex(ValueError, "finalized numeric"):
            uploader.inspect_checkpoint(tmp_ckpt)

    def test_secret_or_unknown_file_rejected(self):
        (self.ckpt / "params/.env").write_text("do not upload")
        with self.assertRaisesRegex(ValueError, "Unexpected file"):
            uploader.inspect_checkpoint(self.ckpt)

    def test_symlink_rejected(self):
        (self.ckpt / "params/d/fff").symlink_to(self.ckpt / "train_state/not_for_upload")
        with self.assertRaisesRegex(ValueError, "symlink"):
            uploader.inspect_checkpoint(self.ckpt)

    def test_missing_data_rejected(self):
        (self.ckpt / "params/d/abc123").unlink()
        with self.assertRaisesRegex(ValueError, "No Orbax"):
            uploader.inspect_checkpoint(self.ckpt)

    def test_atomic_private_commit_and_verify(self):
        api = FakeApi()
        self.assertEqual(self.publish(api), "b" * 40)
        self.assertTrue(api.create_kwargs["private"])
        self.assertEqual(api.commits[0]["parent_commit"], "a" * 40)
        self.assertTrue(api.commits[0]["operations"][-1].path_in_repo.endswith("eval_manifest.json"))
        self.assertFalse(any("train_state" in op.path_in_repo for op in api.commits[0]["operations"]))

    def test_public_repo_requires_explicit_flag(self):
        api = FakeApi(private=False)
        with self.assertRaisesRegex(ValueError, "repository is public"):
            self.publish(api)
        self.assertFalse(api.commits)
        self.assertEqual(self.publish(api, public=True), "b" * 40)

    def test_existing_snapshot_never_overwritten(self):
        prefix = uploader.inspect_checkpoint(self.ckpt)[3]
        api = FakeApi(existing=[prefix + "/params/_METADATA"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.publish(api)
        self.assertFalse(api.commits)

    def test_bad_remote_size_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.publish(FakeApi(bad_sizes=True))

    def test_manifest_does_not_copy_logs_or_private_paths(self):
        logs = self.root / "beans/ablations/logs"
        logs.mkdir(parents=True)
        (logs / "train_token_test_B.log").write_text("launch config=x code=ccf063d\nSECRET=not-for-upload\n")
        (logs / "token_test_B.recipe").write_text("v1 row=snap_token_mlp3_a9align stage=B batch=12 accum=3 steps=3000 A=500 base=/private/path")
        ckpt, config, exp, _, _, sizes = uploader.inspect_checkpoint(self.ckpt)
        with patch.object(uploader, "git_value", side_effect=["c" * 40, ""]):
            manifest = uploader.build_manifest(self.root, ckpt, config, exp, sizes)
        encoded = json.dumps(manifest)
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("/private/path", encoded)
        self.assertEqual(manifest["recipe"]["accum"], "3")
        self.assertEqual(manifest["first_launch_code_from_log"], "ccf063d")


if __name__ == "__main__":
    unittest.main()
