from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG  # noqa: F401

from domain import UserTarget
from lemonade import detect, extra_dir, gguf_sources


def _target(home: Path) -> UserTarget:
    return UserTarget(
        name="alice",
        uid=1000,
        gid=1000,
        home=home,
        model_root=home / "Models",
        extra_model_paths=(home / "AI models",),
        bind="127.0.0.1",
    )


class LemonadePublishTests(unittest.TestCase):
    def test_symlink_store_is_not_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            real.mkdir()
            blob = real / "tiny.gguf"
            blob.write_bytes(b"G" * 2048)
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "tiny.gguf").symlink_to(blob)
            src = gguf_sources(_target(home))
            self.assertEqual(src, (real.resolve(),))
            self.assertNotIn((home / "Models" / "gguf").resolve(), src)

    def test_real_store_gguf_is_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "tiny.gguf").write_bytes(b"G" * 2048)
            src = gguf_sources(_target(home))
            self.assertIn(store.resolve(), src)

    def test_snap_extra_dir(self) -> None:
        self.assertEqual(
            extra_dir("snap"),
            Path("/var/snap/lemonade-server/common/ubuntuai-models"),
        )

    def test_detect_snap_when_common_exists(self) -> None:
        with patch("lemonade.SNAP_COMMON") as common:
            common.is_dir.return_value = True
            self.assertEqual(detect(), "snap")
