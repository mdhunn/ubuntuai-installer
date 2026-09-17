from __future__ import annotations

import os
import pwd
import unittest
from pathlib import Path
from unittest.mock import patch

from support import PKG

from apply import PKG_RE, assert_model_root, build_plan
from catalog import load_workflows
from domain import Device, Hardware, UserTarget


def _hw_strix() -> Hardware:
    return Hardware(
        cpu_name="AMD RYZEN AI MAX+ 395",
        ram_bytes=128 * 1024**3,
        devices=(
            Device("npu", "amd", "XDNA", "/dev/accel/accel0", "xdna"),
            Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


class ApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows(PKG / "workflows.json")
        pw = pwd.getpwuid(os.getuid())
        self.target = UserTarget(
            name=pw.pw_name,
            uid=pw.pw_uid,
            gid=pw.pw_gid,
            home=Path(pw.pw_dir),
            model_root=Path(pw.pw_dir) / "Models",
        )

    def test_package_regex(self) -> None:
        self.assertTrue(PKG_RE.match("libggml0-backend-vulkan"))
        self.assertFalse(PKG_RE.match("foo; rm -rf /"))
        self.assertFalse(PKG_RE.match("../evil"))

    def test_model_root_must_stay_home(self) -> None:
        name = self.target.name
        home = self.target.home
        with self.assertRaises(ValueError):
            assert_model_root(name, Path("/etc/passwd"))
        root = assert_model_root(name, home / "Models")
        self.assertEqual(root, (home / "Models").resolve())

    def test_plan_installs_rocm_for_serve_on_strix(self) -> None:
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-core", "ubuntuai-serve"),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        kinds = {a.kind: a for a in actions}
        self.assertNotIn("skip", kinds)
        self.assertIn("apt_install", kinds)
        pkgs = kinds["apt_install"].payload
        self.assertIn("rocminfo", pkgs)
        self.assertIn("libggml0-backend-hip", pkgs)
        self.assertNotIn("nvidia-cuda-toolkit", pkgs)

    def test_plan_skips_serve_on_cpu_only(self) -> None:
        cpu = Hardware(
            cpu_name="cpu",
            ram_bytes=8 * 1024**3,
            devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
        )
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-core", "ubuntuai-serve"),
                cpu,
                self.target,
                self.wfs,
            )
        self.assertTrue(
            any("ubuntuai-serve" in a.payload for a in actions if a.kind == "skip")
        )

    def test_plan_installs_missing_apt(self) -> None:
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=False
        ):
            actions = build_plan(
                ("ubuntuai-chat",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        kinds = {a.kind: a for a in actions}
        self.assertIn("apt_install", kinds)
        self.assertIn("llama.cpp-tools", kinds["apt_install"].payload)
        self.assertIn("groups", kinds)

    def test_openmoss_plan_installs_runtime_and_weights(self) -> None:
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-tts-openmoss",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        kinds = {a.kind: a for a in actions}
        self.assertIn("vendor", kinds)
        self.assertEqual(kinds["vendor"].payload, ("openmoss",))
        self.assertIn("weights", kinds)
        self.assertIn("moss-tts-local-q8", kinds["weights"].payload)
        self.assertIn("moss-tts-local-q8-extras", kinds["weights"].payload)

    def test_whisper_plan_requires_ggml(self) -> None:
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-stt-whisper",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        weight_ids = []
        for a in actions:
            if a.kind == "weights":
                weight_ids.extend(a.payload)
        self.assertIn("whisper-base-en", weight_ids)

    def test_dry_run_does_not_write(self) -> None:
        from apply import execute_plan

        t = self.target
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(("ubuntuai-core",), _hw_strix(), t, self.wfs)
        with patch("apply.helper_path") as hp:
            log = execute_plan(actions, t, dry_run=True)
        self.assertTrue(log)
        hp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
