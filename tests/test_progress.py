from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from support import PKG  # noqa: F401

from domain import Action
from progress import (
    ProgressEvent,
    append_log,
    apply_log_dir,
    download_english,
    english_for_action,
    new_apply_log,
)


class ProgressTests(unittest.TestCase):
    def test_english_for_known_actions(self) -> None:
        self.assertIn(
            "Ubuntu packages",
            english_for_action(Action("apt_install", "apt", ("whisper.cpp", "rhvoice"))),
        )
        self.assertIn("graphics groups", english_for_action(Action("groups", "g", ("render",))))
        self.assertIn("model store", english_for_action(Action("model_dirs", "d", ("gguf",))))
        self.assertIn("openmoss", english_for_action(Action("vendor", "v", ("openmoss",))).lower())
        self.assertIn("model files", english_for_action(Action("weights", "w", ("whisper-base-en",))))

    def test_download_english_has_sizes(self) -> None:
        msg = download_english("OpenMOSS", 1024, 2048)
        self.assertIn("OpenMOSS", msg)
        self.assertIn("of", msg)

    def test_log_file_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = new_apply_log(home)
            self.assertTrue(path.is_file())
            append_log(path, ProgressEvent("Installing Ubuntu packages.", "apt-get install rhvoice"))
            text = path.read_text(encoding="utf-8")
            self.assertIn("Installing Ubuntu packages.", text)
            self.assertIn("apt-get install rhvoice", text)
            last = home / ".local" / "share" / "ubuntuai" / "logs" / "last.log"
            self.assertTrue(last.exists() or last.is_symlink())

    def test_new_apply_log_chowns_dir_and_file(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            uid = os.getuid()
            gid = os.getgid()
            path = new_apply_log(home, uid=uid, gid=gid)
            folder = apply_log_dir(home)
            self.assertEqual(path.stat().st_uid, uid)
            self.assertEqual(folder.stat().st_uid, uid)
            self.assertEqual((home / ".local" / "share" / "ubuntuai").stat().st_uid, uid)
            append_log(path, ProgressEvent("Installing Ubuntu packages."))
            self.assertIn("Installing Ubuntu packages.", path.read_text(encoding="utf-8"))

    def test_new_apply_log_chowns_requested_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("progress.os.chown") as chown:
                path = new_apply_log(home, uid=4242, gid=4243)
            folder = apply_log_dir(home)
            owned = [(call.args[0], call.args[1], call.args[2]) for call in chown.call_args_list]
            paths = {p for p, _, _ in owned}
            self.assertIn(path, paths)
            self.assertIn(folder, paths)
            self.assertIn(home / ".local" / "share" / "ubuntuai", paths)
            self.assertTrue(all(u == 4242 and g == 4243 for _, u, g in owned))

    def test_new_apply_log_chowns_last_log_symlink_inode(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("progress.os.chown") as chown:
                path = new_apply_log(home, uid=4242, gid=4243)
            last = apply_log_dir(home) / "last.log"
            self.assertTrue(last.is_symlink())
            self.assertEqual(last.readlink(), Path(path.name))
            last_calls = [c for c in chown.call_args_list if c.args[0] == last]
            self.assertEqual(len(last_calls), 1)
            self.assertEqual(last_calls[0].args[1], 4242)
            self.assertEqual(last_calls[0].args[2], 4243)
            self.assertIs(last_calls[0].kwargs.get("follow_symlinks"), False)
            path_calls = [c for c in chown.call_args_list if c.args[0] == path]
            self.assertTrue(path_calls)
            self.assertNotIn("follow_symlinks", path_calls[0].kwargs)

    def test_new_apply_log_chowns_regular_last_log(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch.object(Path, "symlink_to", side_effect=OSError("no symlink")):
                with patch("progress.os.chown") as chown:
                    new_apply_log(home, uid=4242, gid=4243)
            last = apply_log_dir(home) / "last.log"
            self.assertTrue(last.is_file())
            self.assertFalse(last.is_symlink())
            last_calls = [c for c in chown.call_args_list if c.args[0] == last]
            self.assertEqual(len(last_calls), 1)
            self.assertEqual(last_calls[0].args[1], 4242)
            self.assertEqual(last_calls[0].args[2], 4243)

    def test_new_apply_log_uses_guess_user_when_ids_omitted(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            target = SimpleNamespace(uid=4242, gid=4243)
            with patch("users.guess_user", return_value="desktop") as guess, patch(
                "users.target_for", return_value=target
            ) as target_for, patch("progress.os.chown") as chown:
                new_apply_log(home)
            guess.assert_called_once_with()
            target_for.assert_called_once()
            self.assertTrue(chown.call_args_list)
            self.assertTrue(all(c.args[1] == 4242 and c.args[2] == 4243 for c in chown.call_args_list))


if __name__ == "__main__":
    unittest.main()
