from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG  # noqa: F401

from catalog import load_workflows
from configstore import load as load_config
from configstore import record_installed
from domain import Check, Device, Hardware, UserTarget
from runtime import app_statuses, backend_choices, chat_models, endpoint_for, is_present
from validate import format_health


def _target(home: Path) -> UserTarget:
    return UserTarget(
        name="alice",
        uid=1000,
        gid=1000,
        home=home,
        model_root=home / "Models",
        extra_model_paths=(),
        bind="127.0.0.1",
    )


class RuntimeTests(unittest.TestCase):
    def test_endpoint_chat_uses_v1(self) -> None:
        wf = next(w for w in load_workflows() if w.id == "ubuntuai-chat")
        self.assertEqual(endpoint_for(wf, "127.0.0.1"), "http://127.0.0.1:8080/v1")
        self.assertEqual(endpoint_for(wf, "0.0.0.0"), "http://0.0.0.0:8080/v1")

    def test_espeak_only_when_recorded(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            espeak = next(w for w in load_workflows() if w.id == "ubuntuai-tts-espeak")
            with patch("runtime.recorded_ids", return_value=frozenset()):
                self.assertFalse(is_present(espeak, "alice", t))
            with patch("runtime.recorded_ids", return_value=frozenset({"ubuntuai-tts-espeak"})):
                self.assertTrue(is_present(espeak, "alice", t))

    def test_chat_live_without_record(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            chat = next(w for w in load_workflows() if w.id == "ubuntuai-chat")
            with patch("runtime.recorded_ids", return_value=frozenset()), patch(
                "runtime.shutil.which", return_value="/usr/bin/llama-server"
            ):
                self.assertTrue(is_present(chat, "alice", t))

    def test_chat_present_when_flm_missing(self) -> None:
        def fake_which(name: str) -> str | None:
            if name == "llama-server":
                return "/usr/bin/llama-server"
            return None

        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            wfs = load_workflows()
            chat = next(w for w in wfs if w.id == "ubuntuai-chat")
            hybrid = next(w for w in wfs if w.id == "ubuntuai-hybrid")
            with patch("runtime.recorded_ids", return_value=frozenset()), patch(
                "runtime.shutil.which", side_effect=fake_which
            ), patch("runtime.dpkg_installed", return_value=False):
                self.assertTrue(is_present(chat, "alice", t))
                self.assertFalse(is_present(hybrid, "alice", t))

    def test_chat_models_lists_gguf(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gguf").mkdir()
            (root / "gguf" / "tiny.gguf").write_bytes(b"g" * 10)
            (root / "gguf" / "skip.txt").write_text("x")
            self.assertEqual(chat_models(root), ("tiny.gguf",))

    def test_backend_choices_include_rocm_on_amd(self) -> None:
        hw = Hardware(
            cpu_name="x",
            ram_bytes=1,
            devices=(
                Device("igpu", "amd", "radeon", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        choices = backend_choices(hw)
        self.assertEqual(choices[0], "auto")
        self.assertIn("rocm", choices)
        self.assertIn("vulkan", choices)

    def test_app_statuses_hide_missing_endpoints(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            with patch("runtime.recorded_ids", return_value=frozenset()), patch(
                "runtime.live_present", return_value=False
            ):
                apps = app_statuses("alice", t)
            present = [a for a in apps if a.present]
            self.assertEqual(present, [])

    def test_record_installed_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            with patch("configstore.target_for", return_value=t):
                record_installed("alice", ("ubuntuai-chat", "ubuntuai-chat"))
                data = load_config("alice")
            self.assertEqual(data["installed_workflows"], ["ubuntuai-chat"])

    def test_format_health_english(self) -> None:
        ok = format_health((Check("cpu", "ok", "fine"),))
        self.assertIn("ready", ok.lower())
        bad = format_health(
            (Check("group:render", "fail", "user is not in render"),)
        )
        self.assertIn("need a fix", bad.lower())
        self.assertIn("not in render", bad)
        self.assertNotIn("FAIL", bad)


if __name__ == "__main__":
    unittest.main()
