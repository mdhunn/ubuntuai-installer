from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from support import PKG  # noqa: F401

from domain import Action
from progress import (
    ProgressEvent,
    append_log,
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


if __name__ == "__main__":
    unittest.main()
