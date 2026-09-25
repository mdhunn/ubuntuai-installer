from __future__ import annotations

import inspect
import os
import pwd
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from support import PKG

from domain import Device, Hardware, UserTarget
from probe import (
    dpkg_query_status,
    fc_match_text,
    katex_family_resolves,
    ubuntu_needs_katex_fonts,
    ubuntu_version,
)
from repair import classical_plan
from validate import (
    FONTS_KATEX_MISSING,
    LEMONADE_APT_PRESENT,
    collect,
    fonts_katex_check,
    lemonade_apt_check,
)


INSTALLED = "install ok installed"
OS_2604 = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="26.04"\n'
OS_2604_POINT = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="26.04.1"\n'
OS_2610 = 'NAME="Ubuntu"\nID=Ubuntu\nVERSION_ID=26.10\n'
OS_2404 = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="24.04"\n'
OS_DEBIAN = 'NAME="Debian"\nID=debian\nVERSION_ID="13"\n'
OS_MINT = 'NAME="Linux Mint"\nID=linuxmint\nID_LIKE=ubuntu\nVERSION_ID="22.04"\n'
FC_RESOLVED = 'KaTeX_Main-Regular.ttf: "KaTeX_Main" "Regular"'
FC_FALLBACK = 'DejaVuSans.ttf: "DejaVu Sans" "Book"'
FC_POISON = 'KaTeX_AMS-Regular.woff: "Noto Sans" "<unknown style>"'

_NEW_FUNCS = (
    ubuntu_version,
    dpkg_query_status,
    fc_match_text,
    katex_family_resolves,
    fonts_katex_check,
    lemonade_apt_check,
)
_FORBIDDEN = (
    "apt-get",
    "apt install",
    "purge",
    "fc-cache",
    "fontconfig",
    "Conflicts",
    "write_text",
    "shell=True",
    "unlink",
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


def _cpu() -> Hardware:
    return Hardware(
        cpu_name="cpu",
        ram_bytes=8 * 1024**3,
        devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
    )


def _named(checks, name: str):
    return [c for c in checks if c.name == name]


class UbuntuReleaseTests(unittest.TestCase):
    def test_parses_ubuntu_versions(self) -> None:
        self.assertEqual(ubuntu_version(OS_2604), (26, 4))
        self.assertEqual(ubuntu_version(OS_2604_POINT), (26, 4))
        self.assertEqual(ubuntu_version(OS_2610), (26, 10))
        self.assertEqual(ubuntu_version(OS_2404), (24, 4))
        self.assertTrue(ubuntu_needs_katex_fonts((26, 4)))
        self.assertTrue(ubuntu_needs_katex_fonts((26, 10)))
        self.assertFalse(ubuntu_needs_katex_fonts((24, 4)))
        self.assertFalse(ubuntu_needs_katex_fonts((26, 3)))
        self.assertFalse(ubuntu_needs_katex_fonts(None))

    def test_unreadable_or_not_ubuntu_is_none(self) -> None:
        self.assertIsNone(ubuntu_version(""))
        self.assertIsNone(ubuntu_version("ID=ubuntu\n"))
        self.assertIsNone(ubuntu_version(OS_DEBIAN))
        self.assertIsNone(ubuntu_version(OS_MINT))

    def test_injector_does_not_read_os_release(self) -> None:
        with patch("probe._OS_RELEASE") as release:
            release.read_text.side_effect = AssertionError("os-release")
            self.assertEqual(ubuntu_version(OS_2604), (26, 4))
        release.read_text.assert_not_called()


class KatexMatchTests(unittest.TestCase):
    def test_family_resolves_only_when_named(self) -> None:
        self.assertTrue(katex_family_resolves(FC_RESOLVED))
        self.assertTrue(katex_family_resolves("KaTeX_Main"))
        self.assertFalse(katex_family_resolves(FC_FALLBACK))
        self.assertFalse(katex_family_resolves(FC_POISON))
        self.assertFalse(katex_family_resolves(""))


class FontsKatexCheckTests(unittest.TestCase):
    def test_2604_with_package_does_not_warn(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
            fc_match=FC_POISON,
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.name, "fonts-katex")
        self.assertEqual(check.status, "ok")
        self.assertNotEqual(check.status, "warn")

    def test_2604_missing_package_warns(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": ""},
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.name, "fonts-katex")
        self.assertEqual(check.status, "warn")
        self.assertEqual(check.detail, FONTS_KATEX_MISSING)
        self.assertIn("Math rendering", check.detail)
        self.assertIn("Lemonade", check.detail)
        self.assertIn("fonts-katex", check.detail)
        self.assertNotEqual(check.status, "fail")

    def test_2604_missing_package_fallback_font_warns(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": "deinstall ok config-files"},
            fc_match=FC_FALLBACK,
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "warn")

    def test_2604_missing_package_but_family_resolves_is_ok(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status=lambda name: "" if name == "fonts-katex" else INSTALLED,
            fc_match=FC_RESOLVED,
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "ok")
        self.assertIn("KaTeX", check.detail)

    def test_newer_than_2604_missing_warns(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2610,
            dpkg_status={"fonts-katex": ""},
            fc_match=FC_FALLBACK,
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "warn")

    def test_pre_2604_is_skipped(self) -> None:
        with patch("probe.subprocess.run", side_effect=AssertionError):
            check = fonts_katex_check(
                os_release=OS_2404,
                dpkg_status={"fonts-katex": ""},
                fc_match=FC_FALLBACK,
            )
        self.assertIsNone(check)

    def test_unreadable_release_is_skipped(self) -> None:
        self.assertIsNone(fonts_katex_check(os_release="", dpkg_status={}))
        self.assertIsNone(fonts_katex_check(os_release=OS_DEBIAN, dpkg_status={}))
        self.assertIsNone(fonts_katex_check(os_release="ID=ubuntu\n", dpkg_status={}))


class LemonadeAptCheckTests(unittest.TestCase):
    def test_apt_package_warns_for_snap(self) -> None:
        with patch("lemonade.detect", side_effect=AssertionError):
            check = lemonade_apt_check(
                dpkg_status={"lemonade-server": INSTALLED},
            )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.name, "lemonade-apt")
        self.assertEqual(check.status, "warn")
        self.assertEqual(check.detail, LEMONADE_APT_PRESENT)
        self.assertIn("snap", check.detail)
        self.assertIn("lemonade-server", check.detail)
        self.assertNotEqual(check.status, "fail")

    def test_snap_only_or_absent_has_no_apt_warn(self) -> None:
        for status in ("", "deinstall ok config-files", "unknown ok not-installed"):
            check = lemonade_apt_check(dpkg_status={"lemonade-server": status})
            self.assertIsNone(check, status)
        check = lemonade_apt_check(dpkg_status={})
        self.assertIsNone(check)
        check = lemonade_apt_check(
            dpkg_status=lambda name: "" if name == "lemonade-server" else INSTALLED
        )
        self.assertIsNone(check)


class CollectHostGateTests(unittest.TestCase):
    def _collect(self, **kwargs):
        with (
            patch("validate.dpkg_installed", return_value=False),
            patch("validate.lemonade_detect", return_value=""),
            patch("probe.subprocess.run") as run,
        ):
            checks = collect(
                _target(),
                _cpu(),
                which=lambda name: None,
                apt_policy={},
                apt_sources="",
                **kwargs,
            )
        self.assertFalse(run.called)
        return checks

    def test_collect_2604_present_has_no_font_warn(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED, "lemonade-server": ""},
        )
        fonts = _named(checks, "fonts-katex")
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].status, "ok")
        self.assertEqual(_named(checks, "lemonade-apt"), [])

    def test_collect_2604_missing_warns(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": "", "lemonade-server": ""},
            fc_match=FC_FALLBACK,
        )
        fonts = _named(checks, "fonts-katex")
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].status, "warn")
        self.assertEqual(fonts[0].detail, FONTS_KATEX_MISSING)

    def test_collect_pre_2604_skips_fonts(self) -> None:
        checks = self._collect(
            os_release=OS_2404,
            dpkg_status={"fonts-katex": "", "lemonade-server": INSTALLED},
            fc_match=FC_FALLBACK,
        )
        self.assertEqual(_named(checks, "fonts-katex"), [])
        apt = _named(checks, "lemonade-apt")
        self.assertEqual(len(apt), 1)
        self.assertEqual(apt[0].status, "warn")
        self.assertIn("snap", apt[0].detail)

    def test_collect_absent_lemonade_has_no_apt_warn(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
        )
        self.assertEqual(_named(checks, "lemonade-apt"), [])


class ReadOnlyCommandTests(unittest.TestCase):
    def test_injectors_skip_binaries(self) -> None:
        with patch("probe.subprocess.run", side_effect=AssertionError):
            fonts_katex_check(
                os_release=OS_2604,
                dpkg_status={"fonts-katex": ""},
                fc_match=FC_FALLBACK,
            )
            lemonade_apt_check(dpkg_status={"lemonade-server": INSTALLED})
            ubuntu_version(OS_2604)
            fc_match_text(text=FC_RESOLVED)
            dpkg_query_status("fonts-katex", {"fonts-katex": INSTALLED})

    def test_live_dpkg_query_is_status_only(self) -> None:
        with patch("probe.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=INSTALLED + "\n",
                stderr="",
            )
            text = dpkg_query_status("lemonade-server")
        self.assertEqual(text, INSTALLED)
        args = run.call_args[0][0]
        self.assertEqual(args, ["dpkg-query", "-W", "-f=${Status}", "lemonade-server"])
        self.assertNotIn("install", args)
        self.assertNotIn("purge", args)
        self.assertNotIn("remove", args)
        self.assertNotIn("--install", args)

    def test_live_fc_match_is_a_query(self) -> None:
        with patch("probe.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=FC_RESOLVED + "\n",
                stderr="",
            )
            text = fc_match_text()
        self.assertEqual(text, FC_RESOLVED)
        args = run.call_args[0][0]
        self.assertEqual(args, ["fc-match", "KaTeX_Main"])
        self.assertNotIn("-i", args)

    def test_unsafe_names_do_not_run(self) -> None:
        with patch("probe.subprocess.run") as run:
            self.assertEqual(dpkg_query_status("fonts katex"), "")
            self.assertEqual(fc_match_text("KaTeX Main"), "")
        run.assert_not_called()


class NoInstallPathTests(unittest.TestCase):
    def test_new_checks_have_no_install_purge_or_font_write(self) -> None:
        blob = "\n".join(inspect.getsource(fn) for fn in _NEW_FUNCS)
        for token in _FORBIDDEN:
            self.assertNotIn(token, blob, token)
        validate_text = (PKG / "validate.py").read_text(encoding="utf-8")
        for token in ("subprocess", "apt-get", "purge", "fc-cache", "fontconfig"):
            self.assertNotIn(token, validate_text, token)
        quiet = (
            PKG / "apply.py",
            PKG / "vendor.py",
            PKG / "lemonade.py",
            PKG / "workflows.json",
            PKG / "vendors.json",
            PKG.parents[1] / "sbin" / "ubuntuai-installer-helper",
        )
        for path in quiet:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("fonts-katex", text, path.name)
            self.assertNotIn("lemonade-apt", text, path.name)

    def test_repair_keeps_warnings_as_notes(self) -> None:
        plan = classical_plan(
            {
                "checks": [
                    {
                        "name": "fonts-katex",
                        "status": "warn",
                        "detail": FONTS_KATEX_MISSING,
                    },
                    {
                        "name": "lemonade-apt",
                        "status": "warn",
                        "detail": LEMONADE_APT_PRESENT,
                    },
                ],
                "bind": "127.0.0.1",
                "checksums": [],
            }
        )
        self.assertTrue(plan["steps"])
        self.assertTrue(all(step["kind"] == "note" for step in plan["steps"]))
        joined = " ".join(step.get("text", "") for step in plan["steps"])
        self.assertIn("fonts-katex", joined)
        self.assertIn("snap", joined)


if __name__ == "__main__":
    unittest.main()
