from __future__ import annotations

import os
import pwd
import unittest
from pathlib import Path
from unittest.mock import patch

from support import PKG  # noqa: F401

from catalog import load_workflows
from domain import Device, Hardware, UserTarget
from probe import apt_cache_policy, apt_candidate, split_apt_policies, universe_in_sources
from validate import archive_gate_checks, canary_apt_names, collect


IRON_CANARIES = (
    "llama.cpp-tools",
    "libggml0-backend-vulkan",
    "python3-gguf",
    "whisper.cpp",
    "pciutils",
    "mesa-vulkan-drivers",
    "vulkan-tools",
)

UNIVERSE_ON = """
Types: deb
URIs: http://archive.ubuntu.com/ubuntu
Suites: resolute resolute-updates
Components: main restricted universe multiverse
"""

UNIVERSE_OFF = """
deb http://archive.ubuntu.com/ubuntu resolute main restricted
deb http://security.ubuntu.com/ubuntu resolute-security main restricted
"""


def _policy(name: str, candidate: str | None, installed: str = "(none)") -> str:
    cand = candidate if candidate else "(none)"
    return (
        f"{name}:\n"
        f"  Installed: {installed}\n"
        f"  Candidate: {cand}\n"
        f"  Version table:\n"
    )


def _known_map(names: tuple[str, ...]) -> dict[str, str]:
    return {name: _policy(name, "1.0") for name in names}


def _unknown_map(names: tuple[str, ...]) -> dict[str, str]:
    return {name: _policy(name, None) for name in names}


def _target() -> UserTarget:
    pw = pwd.getpwuid(os.getuid())
    return UserTarget(
        name=pw.pw_name,
        uid=pw.pw_uid,
        gid=pw.pw_gid,
        home=Path(pw.pw_dir),
        model_root=Path(pw.pw_dir) / "Models",
    )


def _cpu() -> Hardware:
    return Hardware(
        cpu_name="cpu",
        ram_bytes=8 * 1024**3,
        devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
    )


class AptPolicyParseTests(unittest.TestCase):
    def test_candidate_none_is_unknown(self) -> None:
        self.assertEqual(apt_candidate(_policy("llama.cpp-tools", None)), "")

    def test_candidate_present_when_not_installed(self) -> None:
        text = _policy("llama.cpp-tools", "6643-4")
        self.assertEqual(apt_candidate(text), "6643-4")

    def test_split_keeps_each_package_block(self) -> None:
        blob = _policy("pciutils", "1:3.13.0-2") + _policy("whisper.cpp", None)
        blocks = split_apt_policies(blob)
        self.assertEqual(apt_candidate(blocks["pciutils"]), "1:3.13.0-2")
        self.assertEqual(apt_candidate(blocks["whisper.cpp"]), "")

    def test_injector_skips_apt_binary(self) -> None:
        with patch("probe.subprocess.run") as run:
            text = apt_cache_policy(("pciutils",), text=_policy("pciutils", "1"))
        self.assertIn("Candidate: 1", text)
        run.assert_not_called()

    def test_live_policy_is_apt_cache_only(self) -> None:
        with patch("probe.subprocess.run") as run:
            run.return_value = type(
                "P", (), {"stdout": _policy("pciutils", "1"), "stderr": ""}
            )()
            apt_cache_policy(("pciutils",))
        args = run.call_args[0][0]
        self.assertEqual(args[0], "apt-cache")
        self.assertEqual(args[1], "policy")
        self.assertNotIn("install", args)
        self.assertNotIn("update", args)
        self.assertNotIn("apt-get", args)

    def test_universe_deb822_and_list(self) -> None:
        self.assertTrue(universe_in_sources(UNIVERSE_ON))
        self.assertFalse(universe_in_sources(UNIVERSE_OFF))
        self.assertFalse(universe_in_sources("# Components: universe\n"))


class ArchiveGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows()
        self.names = canary_apt_names(self.wfs)

    def test_canaries_come_from_core_chat_stt(self) -> None:
        self.assertGreaterEqual(len(self.names), 3)
        for pkg in IRON_CANARIES:
            self.assertIn(pkg, self.names)

    def test_known_candidate_is_ok_when_not_installed(self) -> None:
        checks = archive_gate_checks(
            policy_text=_known_map(self.names),
            sources_text=UNIVERSE_ON,
            workflows=self.wfs,
        )
        by_name = {c.name: c for c in checks}
        for pkg in self.names:
            row = by_name[f"apt-known:{pkg}"]
            self.assertEqual(row.status, "ok")
            self.assertIn("knows", row.detail)
            self.assertNotIn("bad package name", row.detail.lower())
        self.assertFalse(any(c.name == "apt-universe" for c in checks))
        self.assertFalse(any(c.status == "fail" for c in checks))

    def test_unknown_candidate_fails_without_blaming_the_catalog(self) -> None:
        checks = archive_gate_checks(
            policy_text=_unknown_map(self.names),
            sources_text=UNIVERSE_ON,
            workflows=self.wfs,
        )
        fails = [c for c in checks if c.status == "fail"]
        self.assertTrue(fails)
        for row in fails:
            self.assertTrue(row.name.startswith("apt-known:"))
            lower = row.detail.lower()
            self.assertNotIn("bad package name", lower)
            self.assertTrue(
                "universe" in lower or "lists" in lower or "suite" in lower
            )

    def test_universe_heuristic_when_several_unknown(self) -> None:
        checks = archive_gate_checks(
            policy_text=_unknown_map(self.names),
            sources_text=UNIVERSE_OFF,
            workflows=self.wfs,
        )
        heuristic = next(c for c in checks if c.name == "apt-universe")
        self.assertEqual(heuristic.status, "warn")
        lower = heuristic.detail.lower()
        self.assertIn("universe", lower)
        self.assertIn("apt update", lower)
        self.assertNotIn("bad package name", lower)

    def test_no_universe_heuristic_when_packages_are_known(self) -> None:
        checks = archive_gate_checks(
            policy_text=_known_map(self.names),
            sources_text=UNIVERSE_OFF,
            workflows=self.wfs,
        )
        self.assertFalse(any(c.name == "apt-universe" for c in checks))

    def test_one_unknown_does_not_trigger_universe_heuristic(self) -> None:
        policy = _known_map(self.names)
        victim = "whisper.cpp"
        policy[victim] = _policy(victim, None)
        checks = archive_gate_checks(
            policy_text=policy,
            sources_text=UNIVERSE_OFF,
            workflows=self.wfs,
        )
        self.assertFalse(any(c.name == "apt-universe" for c in checks))
        row = next(c for c in checks if c.name == "apt-known:whisper.cpp")
        self.assertEqual(row.status, "fail")
        self.assertIn("universe", row.detail.lower())

    def test_combined_policy_text_injector(self) -> None:
        blob = "".join(_policy(name, "2.0") for name in self.names)
        checks = archive_gate_checks(
            policy_text=blob,
            sources_text=UNIVERSE_ON,
            workflows=self.wfs,
        )
        self.assertTrue(all(c.status == "ok" for c in checks if c.name.startswith("apt-known:")))

    def test_missing_apt_cache_warns_and_skips(self) -> None:
        with patch("validate.shutil.which", return_value=None):
            checks = archive_gate_checks(workflows=self.wfs)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].name, "apt-cache")
        self.assertEqual(checks[0].status, "warn")
        self.assertIn("skipped", checks[0].detail.lower())

    def test_preflight_never_calls_install(self) -> None:
        with patch("probe.subprocess.run") as run:
            archive_gate_checks(
                policy_text=_known_map(self.names),
                sources_text=UNIVERSE_ON,
                workflows=self.wfs,
            )
        run.assert_not_called()

    def test_collect_wires_archive_gate(self) -> None:
        checks = collect(
            _target(),
            _cpu(),
            apt_policy=_known_map(self.names),
            apt_sources=UNIVERSE_ON,
        )
        names = [c.name for c in checks]
        self.assertIn("cpu", names)
        self.assertIn("apt-known:llama.cpp-tools", names)
        self.assertIn("apt-known:whisper.cpp", names)
        known = next(c for c in checks if c.name == "apt-known:llama.cpp-tools")
        self.assertEqual(known.status, "ok")


if __name__ == "__main__":
    unittest.main()
