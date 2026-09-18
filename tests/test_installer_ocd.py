from __future__ import annotations

import io
import json
import os
import pwd
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from support import PKG

from apply import (
    ApplyError,
    PKG_RE,
    build_plan,
    execute_plan,
    explain_apt_failure,
    explain_helper_failure,
    format_failure,
    run_privileged,
)
from catalog import by_id, expand_selection, load_workflows, recommended_ids
from domain import Action, Device, Hardware, UserTarget
from main import installer_main
from weights import KNOWN_SUBDIRS, load_catalog


WF_ID = re.compile(r"^ubuntuai-[a-z0-9-]+$")
BACKENDS = {"cpu", "vulkan", "rocm", "cuda", "xdna"}


def _strix() -> Hardware:
    return Hardware(
        cpu_name="AMD RYZEN AI MAX+ 395",
        ram_bytes=128 * 1024**3,
        devices=(
            Device("npu", "amd", "XDNA", "/dev/accel/accel0", "xdna"),
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


def _intel() -> Hardware:
    return Hardware(
        cpu_name="intel",
        ram_bytes=16 * 1024**3,
        devices=(
            Device("igpu", "intel", "arc", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


def _nvidia_amd() -> Hardware:
    return Hardware(
        cpu_name="mix",
        ram_bytes=32 * 1024**3,
        devices=(
            Device("dgpu", "nvidia", "rtx", "/dev/dri/renderD128", "vulkan"),
            Device("igpu", "amd", "radeon", "/dev/dri/renderD129", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
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


class InstallerCatalogContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows(PKG / "workflows.json")
        self.index = by_id(self.wfs)
        self.weights = {w.id for w in load_catalog(PKG / "weights.json")}
        self.vendors = set(json.loads((PKG / "vendors.json").read_text()).keys())

    def test_every_workflow_id_is_stable(self) -> None:
        for wf in self.wfs:
            self.assertRegex(wf.id, WF_ID)
        self.assertIn("ubuntuai-core", self.index)
        self.assertTrue(self.index["ubuntuai-core"].always)
        self.assertTrue(self.index["ubuntuai-core"].default)

    def test_requires_point_at_known_ids_and_are_acyclic(self) -> None:
        ids = set(self.index)
        for wf in self.wfs:
            for req in wf.requires:
                self.assertIn(req, ids, f"{wf.id} requires unknown {req}")
                self.assertNotEqual(req, wf.id)
        for wf in self.wfs:
            expand_selection((wf.id,), self.wfs)

    def test_apt_names_and_subdirs_are_legal(self) -> None:
        for wf in self.wfs:
            for pkg in wf.apt:
                self.assertTrue(PKG_RE.match(pkg), pkg)
            for backend, pkgs in wf.apt_for_backend:
                self.assertIn(backend, BACKENDS)
                for pkg in pkgs:
                    self.assertTrue(PKG_RE.match(pkg), pkg)
            for sub in wf.model_subdirs:
                self.assertIn(sub, KNOWN_SUBDIRS)
            for b in wf.needs_any_backend + wf.hide_unless_backend:
                self.assertIn(b, BACKENDS)
            for port in wf.ports:
                self.assertIsInstance(port, int)
                self.assertGreater(port, 0)
                self.assertLess(port, 65536)
            if wf.vendor:
                self.assertIn(wf.vendor, self.vendors)
            for wid in wf.required_weights:
                self.assertIn(wid, self.weights)

    def test_serve_and_train_declare_gpu_packages(self) -> None:
        serve = self.index["ubuntuai-serve"]
        train = self.index["ubuntuai-train"]
        self.assertEqual(set(serve.hide_unless_backend), {"cuda", "rocm"})
        self.assertEqual(set(train.hide_unless_backend), {"cuda", "rocm"})
        backends = {b for b, _pkgs in serve.apt_for_backend}
        self.assertEqual(backends, {"cuda", "rocm"})

    def test_speech_roles_cover_tts_and_stt(self) -> None:
        roles = {wf.role for wf in self.wfs if wf.role}
        self.assertEqual(roles, {"tts", "stt"})
        self.assertGreaterEqual(sum(1 for w in self.wfs if w.role == "tts"), 2)
        self.assertGreaterEqual(sum(1 for w in self.wfs if w.role == "stt"), 1)

    def test_recommended_always_includes_core(self) -> None:
        rec = recommended_ids(self.wfs, _strix())
        self.assertIn("ubuntuai-core", rec)
        rec_cpu = recommended_ids(self.wfs, _cpu())
        self.assertIn("ubuntuai-core", rec_cpu)
        self.assertNotIn("ubuntuai-hybrid", rec_cpu)


class InstallerPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows(PKG / "workflows.json")
        self.target = _target()

    def test_duplicate_apt_packages_are_unique(self) -> None:
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-chat", "ubuntuai-hybrid"),
                _strix(),
                self.target,
                self.wfs,
            )
        apt = next(a.payload for a in actions if a.kind == "apt_install")
        self.assertEqual(len(apt), len(set(apt)))
        self.assertEqual(apt.count("llama.cpp-tools"), 1)

    def test_hybrid_skipped_without_npu(self) -> None:
        hw = Hardware(
            cpu_name="amd",
            ram_bytes=16 * 1024**3,
            devices=(
                Device("igpu", "amd", "radeon", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-hybrid",),
                hw,
                self.target,
                self.wfs,
            )
        self.assertTrue(
            any("ubuntuai-hybrid" in a.payload for a in actions if a.kind == "skip")
        )

    def test_serve_hidden_on_intel_gpu(self) -> None:
        serve = by_id(self.wfs)["ubuntuai-serve"]
        self.assertFalse(serve.offered(_intel()))

    def test_serve_on_nvidia_plus_amd_asks_both_stacks(self) -> None:
        serve = by_id(self.wfs)["ubuntuai-serve"]
        pkgs = serve.packages_for(_nvidia_amd())
        self.assertIn("rocminfo", pkgs)
        self.assertIn("nvidia-cuda-toolkit", pkgs)

    def test_video_pulls_image_and_core(self) -> None:
        got = expand_selection(("ubuntuai-video",), self.wfs)
        self.assertEqual(got[0], "ubuntuai-core")
        self.assertIn("ubuntuai-image", got)
        self.assertIn("ubuntuai-video", got)

    def test_execute_plan_vendor_needs_probe(self) -> None:
        actions = (Action("vendor", "install openmoss", ("openmoss",)),)
        with self.assertRaises(ApplyError) as ctx:
            execute_plan(actions, self.target, hw=None, dry_run=False)
        self.assertIn("Hardware probe", ctx.exception.english)

    def test_execute_plan_unknown_weight(self) -> None:
        actions = (Action("weights", "missing", ("not-a-weight",)),)
        with self.assertRaises(ApplyError) as ctx:
            execute_plan(actions, self.target, hw=_strix(), dry_run=False)
        self.assertIn("catalog", ctx.exception.english.lower())

    def test_execute_plan_checksum_mismatch_english(self) -> None:
        actions = (Action("weights", "whisper", ("whisper-base-en",)),)
        with patch(
            "apply.ensure_weight",
            side_effect=RuntimeError("md5 mismatch for whisper-base-en"),
        ):
            with self.assertRaises(ApplyError) as ctx:
                execute_plan(actions, self.target, hw=_strix(), dry_run=False)
        self.assertIn("checksum", ctx.exception.english.lower())
        self.assertIn("md5 mismatch", ctx.exception.technical)

    def test_execute_plan_incomplete_download_english(self) -> None:
        actions = (Action("weights", "whisper", ("whisper-base-en",)),)
        with patch(
            "apply.ensure_weight",
            side_effect=RuntimeError("incomplete download for whisper-base-en (4 B of 125 B)"),
        ):
            with self.assertRaises(ApplyError) as ctx:
                execute_plan(actions, self.target, hw=_strix(), dry_run=False)
        self.assertIn("incomplete", ctx.exception.english.lower())
        self.assertNotIn("checksum", ctx.exception.english.lower())
        self.assertIn("incomplete download", ctx.exception.technical)

    def test_execute_plan_groups_failure_english(self) -> None:
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=False
        ):
            actions = build_plan(("ubuntuai-core",), _strix(), self.target, self.wfs)
        with patch("apply.run_privileged", return_value=(1, "usermod: group 'render' does not exist")):
            with self.assertRaises(ApplyError) as ctx:
                execute_plan(actions, self.target, hw=_strix(), dry_run=False)
        self.assertIn("groups", ctx.exception.english.lower())
        self.assertIn("usermod", ctx.exception.technical)

    def test_run_privileged_dry_run_is_silent(self) -> None:
        self.assertEqual(run_privileged("install", ["rocminfo"], dry_run=True), (0, ""))


class InstallerFailureCopyTests(unittest.TestCase):
    def test_apt_english_matrix(self) -> None:
        cases = [
            ("E: Could not get lock /var/lib/dpkg/lock", 100, "another install"),
            ("unmet dependencies:\n foo : Depends: bar but it is not going to be installed", 100, "clash"),
            ("libfoo Conflicts: libbar", 100, "conflict"),
            ("No space left on device", 100, "disk is full"),
            ("E: Unable to locate package nope", 100, "does not know"),
            ("404  Not Found", 100, "network"),
            ("pkexec: dismissed", 126, "permission"),
            ("", 1, "no extra text"),
            ("some other apt rant", 100, "refused"),
        ]
        for output, rc, needle in cases:
            with self.subTest(needle=needle):
                msg = explain_apt_failure(output, rc)
                self.assertIn(needle, msg.lower())
                self.assertNotIn(str(rc), msg)

    def test_helper_verbs(self) -> None:
        self.assertIn("groups", explain_helper_failure("groups", "fail", 1).lower())
        self.assertIn("core files", explain_helper_failure("core-files", "fail", 1).lower())
        self.assertIn("privileged", explain_helper_failure("other", "fail", 1).lower())
        self.assertIn("permission", explain_helper_failure("install", "", 127).lower())

    def test_format_failure_omits_empty_details(self) -> None:
        err = ApplyError("Could not install the Ubuntu packages.")
        text = format_failure(err)
        self.assertEqual(text, err.english)
        self.assertNotIn("Technical details", text)


class InstallerCliTests(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as ctx:
                installer_main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--install", buf.getvalue())
        self.assertIn("--repair", buf.getvalue())

    def test_list_marks_serve_install_on_amd_gpu(self) -> None:
        buf = io.StringIO()
        with patch("main.probe", return_value=_strix()), patch("sys.stdout", buf):
            rc = installer_main(["--list", "--cli"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("ubuntuai-core", text)
        self.assertRegex(text, r"ubuntuai-serve\s+install")
        self.assertNotRegex(text, r"ubuntuai-serve\s+unavailable")
        self.assertRegex(text, r"ubuntuai-hybrid\s+on")


if __name__ == "__main__":
    unittest.main()
