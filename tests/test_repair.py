from __future__ import annotations

import os
import pwd
import unittest

from support import PKG  # noqa: F401

from repair import _sanitize_plan, classical_plan, execute_plan_steps, format_plan
from weights import parse_hf_url


class RepairTests(unittest.TestCase):
    def test_sanitize_drops_shell(self) -> None:
        plan = _sanitize_plan(
            {
                "summary": "bad",
                "steps": [
                    {"kind": "shell", "text": "rm -rf /"},
                    {"kind": "note", "text": "safe"},
                    {"kind": "redownload", "id": "whisper-base-en"},
                    {"kind": "install_workflow", "id": "ubuntuai-stt-whisper"},
                ],
            },
            "openai",
        )
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertNotIn("shell", kinds)
        self.assertIn("note", kinds)
        self.assertIn("redownload", kinds)
        self.assertIn("install_workflow", kinds)
        self.assertTrue(any("Dropped" in (s.get("text") or "") for s in plan["steps"]))

    def test_classical_redownload_on_mismatch(self) -> None:
        diag = {
            "checks": [],
            "bind": "127.0.0.1",
            "checksums": [
                {
                    "id": "whisper-base-en",
                    "path": "/tmp/x",
                    "present": True,
                    "algo": "md5",
                    "expected": "aaa",
                    "actual": "bbb",
                    "match": False,
                }
            ],
        }
        plan = classical_plan(diag)
        self.assertEqual(plan["source"], "classical")
        step = next(s for s in plan["steps"] if s.get("kind") == "redownload")
        self.assertEqual(step.get("id"), "whisper-base-en")
        self.assertEqual(step.get("hash"), "aaa")
        self.assertEqual(step.get("algo"), "md5")

    def test_classical_whisper_binary_is_install(self) -> None:
        diag = {
            "checks": [
                {
                    "name": "whisper.cpp",
                    "status": "warn",
                    "detail": "whisper-cli not installed",
                }
            ],
            "bind": "127.0.0.1",
            "checksums": [],
        }
        plan = classical_plan(diag)
        self.assertTrue(
            any(
                s.get("kind") == "install_workflow" and s.get("id") == "ubuntuai-stt-whisper"
                for s in plan["steps"]
            )
        )
        self.assertFalse(any(s.get("kind") == "redownload" for s in plan["steps"]))

    def test_classical_notes_missing_groups(self) -> None:
        diag = {
            "checks": [
                {
                    "name": "group:render",
                    "status": "fail",
                    "detail": "user not in render",
                }
            ],
            "bind": "127.0.0.1",
            "checksums": [],
        }
        plan = classical_plan(diag)
        self.assertTrue(any(s.get("kind") == "note" for s in plan["steps"]))

    def test_classical_probe_note_not_doubled(self) -> None:
        diag = {
            "checks": [
                {
                    "name": "note",
                    "status": "warn",
                    "detail": "/dev/kfd is present. rocminfo is not.",
                }
            ],
            "bind": "127.0.0.1",
            "checksums": [],
        }
        plan = classical_plan(diag)
        texts = [s.get("text") or "" for s in plan["steps"]]
        self.assertTrue(any(t.startswith("/dev/kfd") for t in texts))
        self.assertFalse(any(t.startswith("note:") for t in texts))

    def test_parse_hf_url(self) -> None:
        parsed = parse_hf_url(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
        )
        self.assertEqual(parsed, ("ggerganov/whisper.cpp", "ggml-base.en.bin"))
        self.assertIsNone(parse_hf_url("http://127.0.0.1/file.gguf"))

    def test_execute_dry_run_does_not_apply(self) -> None:
        try:
            user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            self.skipTest("no current user")
        log = execute_plan_steps(
            user,
            {
                "summary": "test",
                "source": "classical",
                "steps": [
                    {"kind": "note", "text": "hello"},
                    {"kind": "redownload", "id": "whisper-base-en"},
                    {"kind": "install_workflow", "id": "ubuntuai-stt-whisper"},
                    {"kind": "shell", "text": "rm -rf /"},
                ],
            },
            dry_run=True,
        )
        self.assertIn("hello", log)
        self.assertTrue(any("would redownload" in line for line in log))
        self.assertTrue(any("would install_workflow" in line for line in log))
        self.assertTrue(any("skip unsupported" in line for line in log))

    def test_format_plan_is_plain_english(self) -> None:
        plan = {
            "summary": "test",
            "source": "classical",
            "steps": [
                {
                    "kind": "install_workflow",
                    "id": "ubuntuai-stt-whisper",
                    "why": "missing",
                },
                {"kind": "note", "text": "Firmware will not be rewritten."},
            ],
        }
        text = format_plan(plan)
        self.assertIn("What this means", text)
        self.assertIn("Nothing will change until you approve", text)
        self.assertIn("whisper", text.lower())
        self.assertNotIn('"kind":', text)
        self.assertIn("Steps", text)
