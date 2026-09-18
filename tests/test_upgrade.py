from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from support import PKG  # noqa: F401

from unittest.mock import patch

from domain import Device, Hardware, UserTarget
from lemonade import load_tuning
from upgrade import (
    _sanitize,
    classical_plan,
    execute_plan_steps,
    format_plan,
    step_english,
)
from weights import detect_shard_bundle, scan


def _target_stub() -> UserTarget:
    return UserTarget(
        name="owner",
        uid=1000,
        gid=1000,
        home=Path("/tmp/ubuntuai-owner"),
        model_root=Path("/tmp/ubuntuai-owner/Models"),
        extra_model_paths=(),
        bind="127.0.0.1",
    )


def _strix(ram_gib: int = 122) -> Hardware:
    return Hardware(
        cpu_name="strix",
        ram_bytes=ram_gib * 1024**3,
        devices=(
            Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "rocm"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


class ShardBundleTests(unittest.TestCase):
    def test_shard_folder_stays_one_entry(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            src = home / "AI models" / "Huge-120B"
            src.mkdir(parents=True)
            for i in range(1, 5):
                (src / f"Huge-120B-Q4_K_M-{i:05d}-of-00004.gguf").write_bytes(b"G" * (70 * 1024))
            (src / "mmproj-model.gguf").write_bytes(b"M" * (70 * 1024))
            store = home / "Models"
            found = scan((home / "AI models",), store)
            dirs = [f for f in found if f.kind == "dir"]
            self.assertEqual(len(dirs), 1)
            self.assertEqual(dirs[0].subdir, "gguf")
            self.assertTrue(detect_shard_bundle(src))
            files = [f for f in found if f.kind == "file" and "00001-of" in f.dest_name]
            self.assertEqual(files, [])


class TuningTests(unittest.TestCase):
    def test_large_model_gets_small_context(self) -> None:
        hw = _strix(122)
        huge = int(0.6 * hw.ram_bytes)
        tun = load_tuning(hw, huge)
        self.assertEqual(tun["max_loaded_models"], 1)
        self.assertEqual(tun["llamacpp_backend"], "vulkan")
        self.assertEqual(tun["ctx_size"], 4096)
        self.assertGreaterEqual(int(tun["global_timeout"]), 1800)
        self.assertEqual(tun["llamacpp_args"], "--load-mode mmap")

    def test_small_model_keeps_auto_context(self) -> None:
        hw = _strix(122)
        tun = load_tuning(hw, 2 * 1024**3)
        self.assertEqual(tun["ctx_size"], -1)
        self.assertNotIn("llamacpp_args", tun)

    def test_navi_dgpu_keeps_rocm_tuning(self) -> None:
        hw = Hardware(
            cpu_name="amd",
            ram_bytes=32 * 1024**3,
            devices=(
                Device("dgpu", "amd", "Radeon RX 7900 XT", "/dev/dri/renderD128", "rocm"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        tun = load_tuning(hw, 2 * 1024**3)
        self.assertEqual(tun["llamacpp_backend"], "rocm")


class UpgradePlanTests(unittest.TestCase):
    def test_sanitize_drops_shell(self) -> None:
        plan = _sanitize(
            {
                "summary": "x",
                "steps": [
                    {"kind": "shell", "text": "rm -rf /"},
                    {"kind": "upgrade_vendor", "id": "openmoss", "version": "v9"},
                ],
            }
        )
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertNotIn("shell", kinds)
        self.assertIn("upgrade_vendor", kinds)

    def test_classical_plan_english(self) -> None:
        diag = {
            "ram_bytes": 122 * 1024**3,
            "largest_gguf_bytes": 70 * 1024**3,
            "vendors": [
                {
                    "id": "openmoss",
                    "title": "OpenMOSS",
                    "pinned": "v0.3.0",
                    "installed": "v0.3.0",
                    "latest": "v0.4.0",
                    "newer": True,
                    "url": "https://example.invalid/openmoss.tar.gz",
                    "present": True,
                }
            ],
            "models": [
                {
                    "id": "whisper-base-en",
                    "title": "whisper",
                    "present": True,
                    "stale": True,
                    "algo": "sha256",
                    "hash": "abc",
                }
            ],
            "lemonade": "snap",
            "tuning": {
                "ctx_size": 4096,
                "global_timeout": 1800,
                "max_loaded_models": 1,
                "llamacpp_backend": "rocm",
            },
            "lemonade_updates": "Qwen3-0.6B-GGUF update available",
        }
        plan = classical_plan(diag)
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertIn("upgrade_vendor", kinds)
        self.assertIn("redownload", kinds)
        self.assertIn("lemonade_optimize", kinds)
        self.assertIn("lemonade_update_models", kinds)
        text = format_plan(plan)
        self.assertIn("What this means", text)
        self.assertIn("Nothing will change until you approve", text)
        self.assertIn("openmoss", step_english(plan["steps"][0]))

    def test_classical_plan_warns_without_blocking(self) -> None:
        diag = {
            "ram_bytes": 122 * 1024**3,
            "largest_gguf_bytes": 105 * 1024**3,
            "vendors": [],
            "models": [],
            "lemonade": "snap",
            "tuning": {
                "ctx_size": 2048,
                "global_timeout": 2400,
                "max_loaded_models": 1,
                "llamacpp_backend": "vulkan",
                "llamacpp_args": "--load-mode mmap",
            },
            "load_risk": {
                "level": "strong",
                "policy": "warn_only",
                "action": "warn",
                "frac": 0.86,
                "bytes": 105 * 1024**3,
                "backend": "vulkan",
            },
            "lemonade_updates": "",
        }
        plan = classical_plan(diag)
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertIn("lemonade_optimize", kinds)
        self.assertIn("note", kinds)
        notes = [s["text"] for s in plan["steps"] if s.get("kind") == "note"]
        self.assertTrue(any("Strong warning" in text for text in notes))
        self.assertTrue(any("continue" in text.lower() for text in notes))
        why = next(s["why"] for s in plan["steps"] if s["kind"] == "lemonade_optimize")
        self.assertIn("--load-mode mmap", why)
        self.assertNotIn("refuse", format_plan(plan).lower())

    def test_execute_optimize_uses_live_tuning(self) -> None:
        target = _target_stub()
        with (
            patch("upgrade.target_for", return_value=target),
            patch("upgrade.probe", return_value=_strix()),
            patch(
                "upgrade.report_load_tuning",
                return_value="lemonade load settings updated\nStrong warning. continue.",
            ) as report,
        ):
            log = execute_plan_steps(
                target.name,
                {
                    "steps": [{"kind": "lemonade_optimize", "why": "tune"}],
                    "tuning": {"max_loaded_models": 1},
                },
            )
        report.assert_called_once()
        self.assertIn("lemonade load settings updated", log)
        self.assertTrue(any("Strong warning" in line for line in log))
