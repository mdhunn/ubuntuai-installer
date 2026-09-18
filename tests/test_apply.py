from __future__ import annotations

import os
import pwd
import unittest
from pathlib import Path
from unittest.mock import patch

from support import PKG

from apply import (
    APPLY_PUBLISH_VERB,
    ApplyError,
    PKG_RE,
    assert_model_root,
    build_plan,
    ensure_ggml_vulkan,
    execute_plan,
    explain_apt_failure,
    format_failure,
)
from catalog import load_workflows
from domain import Action, Device, Hardware, UserTarget, Workflow


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


def _wf(
    wid: str,
    *,
    always: bool = False,
    requires: tuple[str, ...] = (),
    apt: tuple[str, ...] = (),
) -> Workflow:
    return Workflow(
        id=wid,
        title=wid,
        summary=wid,
        default=False,
        always=always,
        requires=requires,
        apt=apt,
        groups=(),
        model_subdirs=(),
        needs_any_backend=(),
        hide_unless_backend=(),
        ports=(),
        notes="",
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
        self.assertIn("libggml0-backend-vulkan", kinds["apt_install"].payload)
        self.assertIn("groups", kinds)

    def test_ensure_ggml_vulkan_adds_backend_for_llama_and_whisper(self) -> None:
        self.assertEqual(
            ensure_ggml_vulkan(["llama.cpp-tools"]),
            ["llama.cpp-tools", "libggml0-backend-vulkan"],
        )
        self.assertEqual(
            ensure_ggml_vulkan(["whisper.cpp"]),
            ["whisper.cpp", "libggml0-backend-vulkan"],
        )
        both = ensure_ggml_vulkan(["llama.cpp-tools", "libggml0-backend-vulkan"])
        self.assertEqual(both.count("libggml0-backend-vulkan"), 1)
        self.assertIn("llama.cpp-tools", both)
        self.assertEqual(ensure_ggml_vulkan(["rhvoice"]), ["rhvoice"])

    def test_plan_adds_vulkan_when_catalog_omits_it(self) -> None:
        wfs = (
            _wf("ubuntuai-core", always=True, apt=("pciutils",)),
            _wf(
                "ubuntuai-chat",
                requires=("ubuntuai-core",),
                apt=("llama.cpp-tools",),
            ),
        )
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-chat",),
                _hw_strix(),
                self.target,
                wfs,
            )
        pkgs = next(a.payload for a in actions if a.kind == "apt_install")
        self.assertIn("llama.cpp-tools", pkgs)
        self.assertIn("libggml0-backend-vulkan", pkgs)

    def test_plan_adds_vulkan_when_whisper_catalog_omits_it(self) -> None:
        wfs = (
            _wf("ubuntuai-core", always=True, apt=("pciutils",)),
            _wf(
                "ubuntuai-stt-whisper",
                requires=("ubuntuai-core",),
                apt=("whisper.cpp",),
            ),
        )
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-stt-whisper",),
                _hw_strix(),
                self.target,
                wfs,
            )
        pkgs = next(a.payload for a in actions if a.kind == "apt_install")
        self.assertIn("whisper.cpp", pkgs)
        self.assertIn("libggml0-backend-vulkan", pkgs)

    def test_whisper_plan_requires_vulkan_backend(self) -> None:
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-stt-whisper",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        pkgs = next(a.payload for a in actions if a.kind == "apt_install")
        self.assertIn("whisper.cpp", pkgs)
        self.assertIn("libggml0-backend-vulkan", pkgs)

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

    def test_explain_apt_unable_to_locate_mentions_universe(self) -> None:
        for output in (
            "E: Unable to locate package llama.cpp-tools",
            "E: Package 'whisper.cpp' has no installation candidate",
        ):
            with self.subTest(output=output):
                msg = explain_apt_failure(output, 100)
                lower = msg.lower()
                self.assertIn("universe", lower)
                self.assertIn("apt update", lower)
                self.assertIn("26.04", msg)
                self.assertNotIn("100", msg)

    def test_chat_plan_requires_default_gguf(self) -> None:
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-chat",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        weight_ids = []
        for a in actions:
            if a.kind == "weights":
                weight_ids.extend(a.payload)
        self.assertIn("qwen3-0.6b-q8_0", weight_ids)

    def test_explain_apt_lock_is_english(self) -> None:
        msg = explain_apt_failure(
            "E: Could not get lock /var/lib/dpkg/lock-frontend",
            100,
        )
        self.assertIn("another install is running", msg.lower())
        self.assertNotIn("100", msg)

    def test_explain_apt_conflict_is_english(self) -> None:
        msg = explain_apt_failure(
            "The following packages have unmet dependencies:\n libfoo : Conflicts: libbar",
            100,
        )
        self.assertTrue("clash" in msg.lower() or "conflict" in msg.lower())

    def test_execute_plan_apt_failure_keeps_details(self) -> None:
        with patch("apply.dpkg_installed", return_value=False), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(
                ("ubuntuai-chat",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        apt_log = "E: Unable to locate package llama.cpp-tools"
        with patch("apply.run_privileged", return_value=(100, apt_log)):
            with self.assertRaises(ApplyError) as ctx:
                execute_plan(actions, self.target, hw=_hw_strix(), dry_run=False)
        err = ctx.exception
        self.assertIn("universe", err.english.lower())
        self.assertIn("apt update", err.english.lower())
        self.assertIn("26.04", err.english)
        self.assertIn("Unable to locate package", err.technical)
        self.assertIn("exit: 100", err.technical)
        text = format_failure(err)
        self.assertIn("Technical details:", text)
        self.assertTrue(text.startswith(err.english))

    def test_dry_run_does_not_write(self) -> None:
        t = self.target
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ):
            actions = build_plan(("ubuntuai-core",), _hw_strix(), t, self.wfs)
        with patch("apply.helper_path") as hp:
            log = execute_plan(actions, t, dry_run=True)
        self.assertTrue(log)
        hp.assert_not_called()

    def test_plan_publishes_lemonade_when_snap(self) -> None:
        with patch("apply.dpkg_installed", return_value=True), patch(
            "apply.user_in_group", return_value=True
        ), patch("apply.lemonade_detect", return_value="snap"):
            actions = build_plan(
                ("ubuntuai-core",),
                _hw_strix(),
                self.target,
                self.wfs,
            )
        lemon = [a for a in actions if a.kind == "lemonade"]
        self.assertEqual(len(lemon), 1)
        self.assertEqual(lemon[0].payload, (self.target.name,))
        self.assertEqual(APPLY_PUBLISH_VERB, "lemonade-publish")

    def test_execute_lemonade_calls_publish_verb(self) -> None:
        actions = (
            Action("lemonade", "publish GGUF files to Lemonade", (self.target.name,)),
        )
        with patch(
            "apply.run_privileged", return_value=(0, "lemonade extra_models_dir=/var/snap")
        ) as rp:
            log = execute_plan(actions, self.target, hw=_hw_strix(), dry_run=True)
        rp.assert_called_once()
        self.assertEqual(rp.call_args[0][0], APPLY_PUBLISH_VERB)
        self.assertEqual(rp.call_args[0][1], [self.target.name])
        self.assertTrue(any("lemonade" in line.lower() for line in log))


if __name__ == "__main__":
    unittest.main()
