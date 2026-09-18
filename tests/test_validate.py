from __future__ import annotations

import os
import pwd
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from support import CPUINFO, MEMINFO, STRIX_LSPCI

from catalog import by_id, load_workflows, recommended_ids
from domain import Device, Hardware, UserTarget
from probe import HYBRID_FLM_MISSING, probe
from validate import collect, hybrid_engine_check, worst


def _strix() -> Hardware:
    return Hardware(
        cpu_name="AMD RYZEN AI MAX+ 395",
        ram_bytes=122 * 1024**3,
        devices=(
            Device("npu", "amd", "xdna", "/dev/accel/accel0", "xdna"),
            Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


def _cpu() -> Hardware:
    return Hardware(
        cpu_name="cpu",
        ram_bytes=8 * 1024**3,
        devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
    )


def _target() -> UserTarget:
    pw = pwd.getpwuid(os.getuid())
    return UserTarget(
        name=pw.pw_name,
        uid=pw.pw_uid,
        gid=pw.pw_gid,
        home=Path(pw.pw_dir),
        model_root=Path(pw.pw_dir) / "Models",
    )


def _named(checks, name: str):
    return [c for c in checks if c.name == name]


class HybridEngineCheckTests(unittest.TestCase):
    def test_xdna_without_flm_warns(self) -> None:
        check = hybrid_engine_check(_strix(), which=lambda name: None)
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.name, "fastflowlm")
        self.assertEqual(check.status, "warn")
        self.assertEqual(check.detail, HYBRID_FLM_MISSING)
        self.assertIn("llama.cpp", check.detail)
        self.assertNotEqual(check.status, "fail")
        self.assertEqual(worst((check,)), "warn")

    def test_xdna_with_flm_is_ok(self) -> None:
        check = hybrid_engine_check(
            _strix(), which=lambda name: "/tmp/flm" if name == "flm" else None
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "ok")
        self.assertEqual(check.detail, "/tmp/flm")

    def test_cpu_without_flm_has_no_hybrid_warning(self) -> None:
        self.assertIsNone(hybrid_engine_check(_cpu(), which=lambda name: None))

    def test_path_injector_finds_flm(self) -> None:
        with TemporaryDirectory() as tmp:
            flm = Path(tmp) / "flm"
            flm.write_text("#!/bin/sh\n")
            flm.chmod(flm.stat().st_mode | stat.S_IEXEC)
            check = hybrid_engine_check(_strix(), path=tmp)
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "ok")
        self.assertTrue(check.detail.endswith("/flm"))

    def test_path_injector_empty_dir_warns_on_xdna(self) -> None:
        with TemporaryDirectory() as tmp:
            check = hybrid_engine_check(_strix(), path=tmp)
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "warn")


class ValidateCollectHybridTests(unittest.TestCase):
    def test_collect_warns_once_when_xdna_lacks_flm(self) -> None:
        checks = collect(_target(), _strix(), which=lambda name: None)
        flm = _named(checks, "fastflowlm")
        self.assertEqual(len(flm), 1)
        self.assertEqual(flm[0].status, "warn")
        self.assertEqual(flm[0].detail, HYBRID_FLM_MISSING)
        notes = [c for c in checks if c.name == "note" and "flm" in c.detail]
        self.assertEqual(notes, [])

    def test_collect_ok_when_flm_injected(self) -> None:
        checks = collect(
            _target(),
            _strix(),
            which=lambda name: "/opt/flm" if name == "flm" else None,
        )
        flm = _named(checks, "fastflowlm")
        self.assertEqual(len(flm), 1)
        self.assertEqual(flm[0].status, "ok")

    def test_collect_skips_hybrid_warning_without_xdna(self) -> None:
        checks = collect(_target(), _cpu(), which=lambda name: None)
        self.assertEqual(_named(checks, "fastflowlm"), [])


class ChatNotGatedByFlmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows()
        self.index = by_id(self.wfs)

    def test_chat_catalog_has_no_flm_dependency(self) -> None:
        chat = self.index["ubuntuai-chat"]
        self.assertEqual(chat.runtime_bins, ())
        self.assertNotIn("flm", chat.apt)
        self.assertEqual(chat.hide_unless_backend, ())
        self.assertEqual(chat.needs_any_backend, ())
        self.assertTrue(chat.default)
        self.assertTrue(chat.offered(_strix()))
        self.assertTrue(chat.eligible(_strix()))
        self.assertTrue(chat.runtime_ok())

    def test_recommended_still_includes_chat_without_flm(self) -> None:
        rec = recommended_ids(self.wfs, _strix())
        self.assertIn("ubuntuai-chat", rec)
        self.assertIn("ubuntuai-hybrid", rec)
        hybrid = self.index["ubuntuai-hybrid"]
        self.assertEqual(hybrid.hide_unless_backend, ("xdna",))
        self.assertTrue(hybrid.offered(_strix()))
        self.assertTrue(hybrid.eligible(_strix()))


class ProbeHybridNoteTests(unittest.TestCase):
    def test_strix_without_flm_notes_hybrid_lane(self) -> None:
        hw = probe(
            lspci_text=STRIX_LSPCI,
            cpuinfo=CPUINFO,
            meminfo=MEMINFO,
            which=lambda name: None,
        )
        self.assertIn("xdna", hw.backends())
        self.assertTrue(hw.hybrid_ok())
        self.assertIn(HYBRID_FLM_MISSING, hw.notes)

    def test_strix_with_flm_has_no_hybrid_flm_note(self) -> None:
        hw = probe(
            lspci_text=STRIX_LSPCI,
            cpuinfo=CPUINFO,
            meminfo=MEMINFO,
            which=lambda name: "/usr/bin/flm" if name == "flm" else None,
        )
        self.assertTrue(hw.hybrid_ok())
        self.assertNotIn(HYBRID_FLM_MISSING, hw.notes)

    def test_cpu_without_flm_has_no_hybrid_flm_note(self) -> None:
        hw = probe(
            lspci_text="",
            cpuinfo=CPUINFO,
            meminfo=MEMINFO,
            which=lambda name: None,
        )
        self.assertFalse(hw.hybrid_ok())
        self.assertNotIn(HYBRID_FLM_MISSING, hw.notes)


if __name__ == "__main__":
    unittest.main()
