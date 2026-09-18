from __future__ import annotations

import io
import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG  # noqa: F401

from catalog import load_workflows
from configstore import load as load_config
from configstore import record_installed
from configstore import save as save_config
from domain import Check, Device, Hardware, UserTarget
from main import config_main
from runtime import (
    BACKEND_LABELS,
    RECORD_ONLY,
    app_statuses,
    backend_choices,
    chat_models,
    chat_weight_ids,
    endpoint_for,
    explain_setup,
    is_present,
    listen_words,
    machine_words,
    no_chat_gguf,
    workflow_row_title,
)
from users import load_saved_bind, load_saved_model_root
from validate import format_health


def _target(home: Path, bind: str = "127.0.0.1") -> UserTarget:
    return UserTarget(
        name="alice",
        uid=1000,
        gid=1000,
        home=home,
        model_root=home / "Models",
        extra_model_paths=(),
        bind=bind,
    )


def _cpu() -> Hardware:
    return Hardware(
        cpu_name="cpu",
        ram_bytes=8 * 1024**3,
        devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
    )


def _nvidia() -> Hardware:
    return Hardware(
        cpu_name="nvidia-box",
        ram_bytes=16 * 1024**3,
        devices=(
            Device("dgpu", "nvidia", "rtx", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


class ConfigAppContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows()

    def test_statuses_cover_every_workflow(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            with patch("runtime.recorded_ids", return_value=frozenset()), patch(
                "runtime.live_present", return_value=False
            ):
                apps = app_statuses("alice", t)
        self.assertEqual([a.id for a in apps], [w.id for w in self.wfs])
        self.assertTrue(all(a.title for a in apps))
        self.assertTrue(all(a.summary for a in apps))

    def test_endpoints_match_catalog_ports(self) -> None:
        for wf in self.wfs:
            url = endpoint_for(wf, "127.0.0.1")
            if not wf.ports:
                self.assertEqual(url, "")
                continue
            self.assertIn(f":{wf.ports[0]}", url)
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            if wf.id in {"ubuntuai-chat", "ubuntuai-coding", "ubuntuai-serve"}:
                self.assertTrue(url.endswith("/v1"))
            else:
                self.assertFalse(url.endswith("/v1"))
            lan = endpoint_for(wf, "0.0.0.0")
            self.assertIn("0.0.0.0", lan)
            self.assertNotIn("127.0.0.1", lan)

    def test_record_only_workflows_ignore_live_binaries(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            for wid in RECORD_ONLY:
                wf = next(w for w in self.wfs if w.id == wid)
                with patch("runtime.recorded_ids", return_value=frozenset()), patch(
                    "runtime.shutil.which", return_value="/usr/bin/ffmpeg"
                ), patch("runtime.dpkg_installed", return_value=True):
                    self.assertFalse(is_present(wf, "alice", t), wid)

    def test_recorded_image_is_present(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            image = next(w for w in self.wfs if w.id == "ubuntuai-image")
            with patch(
                "runtime.recorded_ids",
                return_value=frozenset({"ubuntuai-image"}),
            ):
                self.assertTrue(is_present(image, "alice", t))
                apps = app_statuses("alice", t)
            shown = next(a for a in apps if a.id == "ubuntuai-image")
            self.assertTrue(shown.present)
            self.assertIn("8188", shown.endpoint)

    def test_explain_uses_plain_listen_words(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            t = _target(home)
            hw = _cpu()
            with patch("runtime.recorded_ids", return_value=frozenset({"ubuntuai-core"})), patch(
                "runtime.live_present",
                side_effect=lambda wf, _t: wf.id == "ubuntuai-core",
            ), patch("runtime.load_config", return_value={"bind": "127.0.0.1", "chat_model": "", "primary_backend": ""}):
                text = explain_setup("alice", t, hw)
        self.assertIn("this computer only", text)
        self.assertNotIn("127.0.0.1", text.split("Listen:", 1)[1].splitlines()[0])
        self.assertIn("Installed", text)
        self.assertIn("Not installed", text)
        self.assertIn("Open Ubuntu AI Installer", text)
        missing = text.split("Not installed", 1)[1]
        self.assertNotIn("Image", missing)
        self.assertNotIn("Video", missing)
        self.assertNotIn("RAG", missing)
        self.assertNotIn("Coding", missing)

    def test_explain_lan_bind(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp), bind="0.0.0.0")
            with patch("runtime.recorded_ids", return_value=frozenset()), patch(
                "runtime.live_present", return_value=False
            ), patch(
                "runtime.load_config",
                return_value={"bind": "0.0.0.0", "chat_model": "", "primary_backend": ""},
            ):
                text = explain_setup("alice", t, _cpu())
        self.assertIn("other devices on your network", text)

    def test_backend_labels_cover_choices(self) -> None:
        for key in backend_choices(_nvidia()):
            self.assertIn(key, BACKEND_LABELS)
        nvidia = backend_choices(_nvidia())
        self.assertIn("cuda", nvidia)
        self.assertNotIn("rocm", nvidia)
        cpu = backend_choices(_cpu())
        self.assertEqual(cpu, ("auto", "cpu"))

    def test_chat_models_skip_junk_and_keep_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            gguf = root / "gguf"
            gguf.mkdir()
            (gguf / "desktop.ini").write_text("x")
            (gguf / "model.gguf").write_bytes(b"g" * 8)
            (gguf / "hf-style").mkdir()
            names = chat_models(root)
            self.assertEqual(names, ("hf-style", "model.gguf"))
            self.assertEqual(chat_models(root / "missing"), ())
            self.assertFalse(no_chat_gguf(root))
            self.assertTrue(no_chat_gguf(root / "missing"))
        self.assertIn("qwen3-0.6b-q8_0", chat_weight_ids(self.wfs))


class ConfigStoreTests(unittest.TestCase):
    def test_save_rejects_bad_bind_and_escaped_root(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            with patch("configstore.target_for", return_value=t):
                with self.assertRaises(ValueError):
                    save_config("alice", {"bind": "1.2.3.4"})
                with self.assertRaises(ValueError):
                    save_config("alice", {"model_root": "/etc/passwd"})
                path = save_config(
                    "alice",
                    {
                        "bind": "0.0.0.0",
                        "model_root": str(t.home / "Models"),
                        "chat_model": "tiny.gguf",
                        "tts_engine": "ubuntuai-tts-rhvoice",
                        "stt_engine": "ubuntuai-stt-whisper",
                        "openai_base_url": "http://127.0.0.1:8080/v1",
                        "openai_api_key": "sk-test",
                    },
                )
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertEqual(mode, 0o600)
                data = load_config("alice")
                self.assertEqual(data["bind"], "0.0.0.0")
                self.assertEqual(data["chat_model"], "tiny.gguf")
                self.assertEqual(data["tts_engine"], "ubuntuai-tts-rhvoice")
                self.assertEqual(data["openai_api_key"], "sk-test")

    def test_record_installed_skips_empty_and_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            t = _target(Path(tmp))
            with patch("configstore.target_for", return_value=t):
                record_installed("alice", ("", "ubuntuai-chat", "ubuntuai-chat", "ubuntuai-core"))
                self.assertEqual(
                    load_config("alice")["installed_workflows"],
                    ["ubuntuai-chat", "ubuntuai-core"],
                )

    def test_users_read_saved_bind_and_root(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = home / ".config" / "ubuntuai" / "config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps(
                    {
                        "bind": "0.0.0.0",
                        "model_root": str(home / "Weights"),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_saved_bind(home), "0.0.0.0")
            self.assertEqual(load_saved_model_root(home), home / "Weights")
            cfg.write_text(json.dumps({"bind": "9.9.9.9", "model_root": "/etc"}), encoding="utf-8")
            self.assertEqual(load_saved_bind(home), "127.0.0.1")
            self.assertIsNone(load_saved_model_root(home))


class ConfigHealthAndCliTests(unittest.TestCase):
    def test_health_lists_fails_before_warns(self) -> None:
        text = format_health(
            (
                Check("a", "ok", "fine"),
                Check("b", "warn", "whisper missing"),
                Check("c", "fail", "not in render"),
            )
        )
        self.assertLess(text.find("need a fix"), text.find("worth a look"))
        self.assertLess(text.find("not in render"), text.find("whisper missing"))
        self.assertIn("Repair", text)

    def test_explain_cli(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            rc = config_main(["--explain"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("Listen:", text)
        self.assertIn("Installed", text)
        self.assertIn("Health", text)
        self.assertIn("this computer only", text.lower())

    def test_show_cli_is_json(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            rc = config_main(["--show"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertIn("bind", data)
        self.assertIn("model_root", data)
        self.assertIn("installed_workflows", data)
        self.assertIn(data["bind"], {"127.0.0.1", "0.0.0.0"})

    def test_listen_and_machine_words_are_plain(self) -> None:
        self.assertEqual(listen_words("127.0.0.1"), "This computer only.")
        self.assertIn("home network", listen_words("0.0.0.0"))
        self.assertNotIn("127.0.0.1", listen_words("127.0.0.1"))
        cpu = machine_words(_cpu())
        self.assertIn("CPU", cpu)
        self.assertNotIn("Backend", cpu)
        self.assertNotIn("Hybrid", cpu)
        npu = Hardware(
            cpu_name="strix",
            ram_bytes=16 * 1024**3,
            devices=(
                Device("npu", "amd", "xdna", "/dev/accel/accel0", "xdna"),
                Device("igpu", "amd", "radeon", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        together = machine_words(npu)
        self.assertIn("NPU and GPU can work together", together)
        self.assertEqual(
            workflow_row_title("Image", "ubuntuai-image", frozenset({"ubuntuai-image"})),
            "Image (helpers only)",
        )
        self.assertEqual(
            workflow_row_title(
                "Image helpers", "ubuntuai-image", frozenset({"ubuntuai-image"})
            ),
            "Image helpers",
        )

    def test_gtk_qt_twins_share_presentment(self) -> None:
        from support import PKG

        gtk = (PKG / "ui" / "gtk_ui.py").read_text(encoding="utf-8")
        qt = (PKG / "ui" / "qt_ui.py").read_text(encoding="utf-8")
        for src in (gtk, qt):
            self.assertIn("listen_words", src)
            self.assertIn("machine_words", src)
            self.assertIn("CHAT_MODEL_CTA", src)
            self.assertIn("CHAT_MODEL_NEEDED_CONFIG", src)
            self.assertIn("HELPERS_SECTION", src)
            self.assertIn("prepare_chat_download", src)
            self.assertIn("no_chat_gguf", src)
            self.assertIn("warn_for_chat_model", src)
            self.assertIn("warn_for_publish", src)
            self.assertIn("_confirm_load_warn", src)
            self.assertIn("This computer only", src)
            self.assertNotIn("This model may strain this computer", src)
            self.assertNotIn("Continue anyway", src)
            self.assertNotIn("Backends:", src)
            self.assertNotIn("Hybrid available", src)
            self.assertNotIn("Hybrid not detected", src)
            self.assertNotIn("Bind {", src)
            self.assertNotIn("No GGUF chat files", src)
            self.assertIn("then copy into", src)
            self.assertNotIn("then symlink into", src)
            self.assertNotIn('"Symlink"', src)
            self.assertNotIn("mode = \"link\"", src)

    def test_help_lists_explain_and_chat_model(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as ctx:
                config_main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        help_text = buf.getvalue()
        self.assertIn("--explain", help_text)
        self.assertIn("--chat-model", help_text)
        self.assertIn("--openai-uri", help_text)


if __name__ == "__main__":
    unittest.main()
