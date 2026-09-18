from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from support import PKG  # noqa: F401

from domain import Action, Device, Hardware, UserTarget
from lemonade import (
    HUGE_GGUF_BYTES,
    LOAD_RISK_POLICY,
    load_risk,
    model_gguf_bytes,
    risk_english,
)
from load_warn import (
    SCOPE,
    SOFT_BODY,
    SOFT_PRIMARY,
    SOFT_TITLE,
    STRONG_BODY,
    STRONG_PRIMARY,
    STRONG_TITLE,
    plan_publishes_lemonade,
    warn_copy,
    warn_for_bytes,
    warn_for_chat_model,
    warn_for_publish,
)
from weights import human_bytes


def _target(home: Path) -> UserTarget:
    return UserTarget(
        name="owner",
        uid=1000,
        gid=1000,
        home=home,
        model_root=home / "Models",
        extra_model_paths=(home / "AI models",),
        bind="127.0.0.1",
    )


def _igpu(ram_gib: int) -> Hardware:
    return Hardware(
        cpu_name="strix",
        ram_bytes=ram_gib * 1024**3,
        devices=(
            Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


class LoadWarnCopyTests(unittest.TestCase):
    def test_strong_uses_locked_english(self) -> None:
        ram = 122 * 1024**3
        size = 70 * 1024**3
        warn = warn_for_bytes(size, ram, "vulkan")
        self.assertEqual(warn.level, "strong")
        self.assertEqual(warn.policy, "warn_only")
        self.assertEqual(warn.action, "warn")
        self.assertEqual(LOAD_RISK_POLICY, "warn_only")
        self.assertTrue(warn.should_prompt)
        self.assertEqual(warn.title, STRONG_TITLE)
        self.assertEqual(warn.title, "This model may strain this computer")
        self.assertEqual(warn.primary, STRONG_PRIMARY)
        self.assertEqual(warn.primary, "Continue anyway")
        self.assertEqual(
            warn.body,
            STRONG_BODY.format(size=human_bytes(size), ram=human_bytes(ram)),
        )
        self.assertIn(human_bytes(size), warn.body)
        self.assertIn(human_bytes(ram), warn.body)
        self.assertIn(SCOPE, warn.body)
        self.assertIn("Installer publish", warn.body)
        self.assertIn("Config Save", warn.body)
        self.assertIn("Lemonade Desktop Load", warn.body)
        self.assertIn("not covered", warn.body)
        self.assertNotEqual(warn.body, risk_english(load_risk(size / ram, size, "vulkan"), ram))

    def test_soft_uses_locked_english(self) -> None:
        ram = 100 * 1024**3
        size = 40 * 1024**3
        warn = warn_for_bytes(size, ram, "vulkan")
        self.assertEqual(warn.level, "warn")
        self.assertTrue(warn.should_prompt)
        self.assertEqual(warn.title, SOFT_TITLE)
        self.assertEqual(warn.title, "Large model on this computer")
        self.assertEqual(warn.primary, SOFT_PRIMARY)
        self.assertEqual(warn.primary, "Continue")
        self.assertEqual(
            warn.body,
            SOFT_BODY.format(size=human_bytes(size), ram=human_bytes(ram)),
        )
        self.assertIn(SCOPE, warn.body)
        self.assertIn("Installer publish", warn.body)
        self.assertIn("Lemonade Desktop Load", warn.body)

    def test_small_does_not_prompt(self) -> None:
        warn = warn_for_bytes(2 * 1024**3, 122 * 1024**3, "vulkan")
        self.assertEqual(warn.level, "ok")
        self.assertFalse(warn.should_prompt)
        self.assertEqual(warn.title, "")
        self.assertEqual(warn.body, "")
        self.assertEqual(warn.primary, "")

    def test_absolute_huge_is_strong(self) -> None:
        ram = 512 * 1024**3
        size = 105 * 1024**3
        self.assertGreaterEqual(size, HUGE_GGUF_BYTES)
        self.assertLess(size / ram, 0.35)
        warn = warn_for_bytes(size, ram, "vulkan")
        self.assertEqual(warn.level, "strong")
        self.assertTrue(warn.should_prompt)
        self.assertEqual(warn.primary, "Continue anyway")

    def test_half_ram_is_strong(self) -> None:
        ram = 64 * 1024**3
        size = 32 * 1024**3
        warn = warn_for_bytes(size, ram, "vulkan")
        self.assertEqual(warn.level, "strong")

    def test_copy_never_refuses(self) -> None:
        risk = load_risk(0.90, 200 * 1024**3, "vulkan")
        warn = warn_copy(risk, 122 * 1024**3)
        self.assertEqual(warn.action, "warn")
        self.assertNotEqual(warn.action, "refuse")
        self.assertTrue(warn.should_prompt)
        self.assertEqual(warn.primary, "Continue anyway")

    def test_locked_bodies_name_installer_and_config_scope(self) -> None:
        self.assertIn(SCOPE, STRONG_BODY)
        self.assertIn(SCOPE, SOFT_BODY)
        self.assertEqual(STRONG_TITLE, "This model may strain this computer")
        self.assertEqual(SOFT_TITLE, "Large model on this computer")
        self.assertEqual(STRONG_PRIMARY, "Continue anyway")
        self.assertEqual(SOFT_PRIMARY, "Continue")


class LoadWarnSurfaceTests(unittest.TestCase):
    def test_publish_uses_largest_on_disk(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            extra.mkdir()
            (extra / "huge.gguf").write_bytes(b"G" * 2048)
            (home / "Models" / "gguf").mkdir(parents=True)
            hw = _igpu(1)
            # 2048 B on 1 GiB RAM is ok.
            warn = warn_for_publish(_target(home), hw)
            self.assertEqual(warn.level, "ok")
            self.assertFalse(warn.should_prompt)

    def test_chat_model_uses_selected_file(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "tiny.gguf").write_bytes(b"G" * 64)
            shard = store / "Huge"
            shard.mkdir()
            for i in range(1, 5):
                (shard / f"huge-0000{i}-of-00004.gguf").write_bytes(b"G" * 1000)
            self.assertEqual(model_gguf_bytes(store / "tiny.gguf"), 64)
            self.assertEqual(model_gguf_bytes(shard), 4000)
            hw = _igpu(122)
            small = warn_for_chat_model(_target(home), hw, "tiny.gguf")
            self.assertEqual(small.level, "ok")
            self.assertEqual(small.bytes, 64)

    def test_auto_chat_model_uses_publish_largest(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            extra.mkdir()
            (extra / "big.gguf").write_bytes(b"G" * 4096)
            (home / "Models" / "gguf").mkdir(parents=True)
            (home / "Models" / "gguf" / "tiny.gguf").write_bytes(b"G" * 64)
            hw = _igpu(122)
            auto = warn_for_chat_model(_target(home), hw, "")
            picked = warn_for_publish(_target(home), hw)
            self.assertEqual(auto.bytes, picked.bytes)
            self.assertGreaterEqual(auto.bytes, 4096)

    def test_plan_detects_lemonade_publish(self) -> None:
        self.assertTrue(
            plan_publishes_lemonade(
                (Action("lemonade", "publish GGUF files to Lemonade", ("owner",)),)
            )
        )
        self.assertFalse(
            plan_publishes_lemonade((Action("apt_install", "install", ("foo",)),))
        )


if __name__ == "__main__":
    unittest.main()
