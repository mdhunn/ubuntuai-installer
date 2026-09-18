from __future__ import annotations

import os
import pwd
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import CPUINFO, MEMINFO, PKG, STRIX_LSPCI

from probe import inspect_npu_firmware, probe
from repair import classical_plan
from users import target_for
from validate import collect, npu_firmware_check


REWRITE_DEFS = (
    "def rewrite_firmware",
    "def flash_firmware",
    "def write_firmware",
    "def install_firmware",
    "def swap_firmware",
    "def update_npu_firmware",
    "def flash_npu",
)


def _blob(root: Path, name: str, data: bytes = b"fw") -> Path:
    path = root / name
    path.write_bytes(data)
    return path


def _link(root: Path, name: str, target: str) -> Path:
    path = root / name
    path.symlink_to(target)
    return path


def _mismatch_tree(root: Path) -> Path:
    _blob(root, "npu.sbin.1.0.0.166.zst")
    _blob(root, "npu.sbin.1.1.2.65.zst")
    _link(root, "npu.sbin.zst", "npu.sbin.1.0.0.166.zst")
    _link(root, "npu_7.sbin.zst", "npu.sbin.1.1.2.65.zst")
    return root / "npu.sbin.zst"


def _matched_tree(root: Path) -> Path:
    _blob(root, "npu.sbin.1.1.2.65.zst")
    _link(root, "npu.sbin.zst", "npu.sbin.1.1.2.65.zst")
    _link(root, "npu_7.sbin.zst", "npu.sbin.1.1.2.65.zst")
    return root / "npu.sbin.zst"


def _target():
    user = pwd.getpwuid(os.getuid()).pw_name
    return target_for(user)


class NpuFirmwareInspectTests(unittest.TestCase):
    def test_mismatch_166_vs_11265(self) -> None:
        with TemporaryDirectory() as tmp:
            link = _mismatch_tree(Path(tmp))
            pair = inspect_npu_firmware(firmware_link=link)
        self.assertEqual(pair.state, "mismatch")
        self.assertEqual(pair.active_version, "1.0.0.166")
        self.assertEqual(pair.disk_version, "1.1.2.65")
        self.assertEqual(pair.disk_name, "npu_7.sbin.zst")
        self.assertIn("mismatched pair", pair.detail)
        self.assertIn("1.0.0.166", pair.detail)
        self.assertIn("1.1.2.65", pair.detail)
        self.assertIn("npu_7.sbin", pair.detail)

    def test_matched_pair_ok(self) -> None:
        with TemporaryDirectory() as tmp:
            link = _matched_tree(Path(tmp))
            pair = inspect_npu_firmware(firmware_link=link)
        self.assertEqual(pair.state, "matched")
        self.assertEqual(pair.active_version, "1.1.2.65")
        self.assertEqual(pair.disk_version, "")
        self.assertIn("matched pair", pair.detail)
        self.assertNotIn("mismatched pair", pair.detail)

    def test_missing_firmware_path(self) -> None:
        with TemporaryDirectory() as tmp:
            link = Path(tmp) / "npu.sbin.zst"
            pair = inspect_npu_firmware(firmware_link=link)
        self.assertEqual(pair.state, "missing")
        self.assertEqual(pair.detail, "firmware path missing")

    def test_probe_injector_uses_fake_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            link = _mismatch_tree(Path(tmp))
            accel = Path(tmp) / "accel0"
            accel.write_bytes(b"")
            with patch("probe.shutil.which", return_value=None):
                hw = probe(
                    lspci_text=STRIX_LSPCI,
                    cpuinfo=CPUINFO,
                    meminfo=MEMINFO,
                    npu_firmware_link=link,
                    npu_accel_node=accel,
                )
        npu = next(d for d in hw.devices if d.backend == "xdna")
        self.assertIn("mismatched pair", npu.detail)
        self.assertEqual(npu.node, str(accel))
        self.assertFalse(any("1.1.2.65 is on disk as npu_7.sbin." == n for n in hw.notes))


class NpuFirmwareValidateTests(unittest.TestCase):
    def test_mismatch_is_warn_validate_only(self) -> None:
        with TemporaryDirectory() as tmp:
            link = _mismatch_tree(Path(tmp))
            pair = inspect_npu_firmware(firmware_link=link)
        check = npu_firmware_check(pair.detail)
        self.assertEqual(check.name, "npu-firmware")
        self.assertEqual(check.status, "warn")
        self.assertIn("mismatched pair", check.detail)
        self.assertIn("1.0.0.166", check.detail)
        self.assertIn("1.1.2.65", check.detail)
        self.assertIn("Validate-only", check.detail)
        self.assertIn("will not rewrite", check.detail)
        self.assertIn("human approval", check.detail.lower())

    def test_matched_is_ok(self) -> None:
        with TemporaryDirectory() as tmp:
            link = _matched_tree(Path(tmp))
            pair = inspect_npu_firmware(firmware_link=link)
        check = npu_firmware_check(pair.detail)
        self.assertEqual(check.status, "ok")
        self.assertIn("matched pair", check.detail)

    def test_missing_is_warn_without_install(self) -> None:
        check = npu_firmware_check("firmware path missing")
        self.assertEqual(check.status, "warn")
        self.assertIn("firmware path missing", check.detail)
        self.assertIn("will not install", check.detail)
        self.assertNotIn("apt", check.detail.lower())

    def test_collect_uses_probe_injector(self) -> None:
        with TemporaryDirectory() as tmp:
            link = _mismatch_tree(Path(tmp))
            with patch("probe.shutil.which", return_value=None):
                hw = probe(
                    lspci_text=STRIX_LSPCI,
                    cpuinfo=CPUINFO,
                    meminfo=MEMINFO,
                    npu_firmware_link=link,
                )
            checks = collect(_target(), hw)
        fw = next(c for c in checks if c.name == "npu-firmware")
        self.assertEqual(fw.status, "warn")
        self.assertIn("mismatched pair", fw.detail)
        self.assertIn("Validate-only", fw.detail)


class NpuFirmwareNoRewriteTests(unittest.TestCase):
    def test_no_rewrite_functions_added(self) -> None:
        root = PKG
        helper = PKG.parents[1] / "sbin" / "ubuntuai-installer-helper"
        files = [
            root / "probe.py",
            root / "validate.py",
            root / "apply.py",
            root / "repair.py",
            helper,
        ]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for snippet in REWRITE_DEFS:
                self.assertNotIn(snippet, text, f"{path.name} has {snippet}")
        helper_text = helper.read_text(encoding="utf-8")
        self.assertNotIn("firmware", helper_text.lower())
        apply_text = (root / "apply.py").read_text(encoding="utf-8")
        self.assertNotIn("firmware", apply_text.lower())

    def test_repair_stays_note_only(self) -> None:
        plan = classical_plan(
            {
                "checks": [
                    {
                        "name": "npu-firmware",
                        "status": "warn",
                        "detail": (
                            "npu.sbin -> npu.sbin.1.0.0.166.zst; mismatched pair; "
                            "1.1.2.65 on disk as npu_7.sbin.zst. Validate-only."
                        ),
                    }
                ],
                "bind": "127.0.0.1",
                "checksums": [],
            }
        )
        kinds = {s["kind"] for s in plan["steps"]}
        self.assertEqual(kinds, {"note"})
        self.assertFalse(
            any("rewrite" in (s.get("kind") or "") for s in plan["steps"])
        )


if __name__ == "__main__":
    unittest.main()
