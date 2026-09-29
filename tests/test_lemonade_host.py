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
    dpkg_status_installed,
    fc_match_text,
    katex_woff_selected,
    plasma_session,
    ubuntu_katex_hazard,
    ubuntu_version,
)
from repair import classical_plan
from validate import (
    LEMONADE_APT_PRESENT,
    LP_KATEX,
    collect,
    fonts_katex_check,
    fonts_katex_detail,
    lemonade_apt_check,
)


INSTALLED = "install ok installed"
HELD = "hold ok installed"
CONFIG_FILES = "deinstall ok config-files"
OS_2604 = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="26.04"\n'
OS_2604_POINT = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="26.04.1"\n'
OS_2610 = 'NAME="Ubuntu"\nID=Ubuntu\nVERSION_ID=26.10\n'
OS_2404 = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="24.04"\n'
OS_DEBIAN = 'NAME="Debian"\nID=debian\nVERSION_ID="13"\n'
OS_MINT = 'NAME="Linux Mint"\nID=linuxmint\nID_LIKE=ubuntu\nVERSION_ID="22.04"\n'
FC_FALLBACK = 'DejaVuSans.ttf: "DejaVu Sans" "Book"'
FC_TTF = 'KaTeX_Main-Regular.ttf: "KaTeX_Main" "Regular"'
FC_WOFF = 'KaTeX_AMS-Regular.woff: "Noto Sans" "<unknown style>"'
GNOME = {"XDG_CURRENT_DESKTOP": "ubuntu:GNOME", "DESKTOP_SESSION": "ubuntu"}
KDE = {"XDG_CURRENT_DESKTOP": "KDE", "DESKTOP_SESSION": "plasma"}

_NEW_FUNCS = (
    ubuntu_version,
    ubuntu_katex_hazard,
    dpkg_query_status,
    dpkg_status_installed,
    fc_match_text,
    katex_woff_selected,
    plasma_session,
    fonts_katex_check,
    lemonade_apt_check,
)
_FORBIDDEN = (
    "apt-get",
    "apt install",
    "fc-cache",
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
        self.assertTrue(ubuntu_katex_hazard((26, 4)))
        self.assertTrue(ubuntu_katex_hazard((26, 10)))
        self.assertFalse(ubuntu_katex_hazard((24, 4)))
        self.assertFalse(ubuntu_katex_hazard((26, 3)))
        self.assertFalse(ubuntu_katex_hazard(None))

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
    def test_woff_selection_and_desktop(self) -> None:
        self.assertTrue(katex_woff_selected(FC_WOFF))
        self.assertFalse(katex_woff_selected(FC_TTF))
        self.assertFalse(katex_woff_selected(FC_FALLBACK))
        self.assertFalse(katex_woff_selected(""))
        self.assertTrue(plasma_session(KDE))
        self.assertTrue(plasma_session({"DESKTOP_SESSION": "plasma"}))
        self.assertFalse(plasma_session(GNOME))
        self.assertFalse(plasma_session({}))


class FontsKatexCheckTests(unittest.TestCase):
    def test_2604_with_package_warns(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
            fc_match=FC_FALLBACK,
            desktop=GNOME,
        )
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.name, "fonts-katex")
        self.assertEqual(check.status, "warn")
        self.assertNotEqual(check.status, "fail")
        self.assertIn("Qt 6.10", check.detail)
        self.assertIn("snap", check.detail)
        self.assertIn("Only remove it or change fontconfig if you decide to.", check.detail)
        self.assertIn("See the linked bug.", check.detail)
        self.assertNotIn("Mark", check.detail)
        self.assertIn(LP_KATEX, check.detail)
        self.assertNotIn(".woff", check.detail)
        self.assertNotIn("This session is KDE Plasma.", check.detail)

    def test_2604_absent_package_does_not_warn(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": ""},
            fc_match=FC_WOFF,
            desktop=KDE,
        )
        self.assertIsNone(check)

    def test_config_files_do_not_warn(self) -> None:
        check = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": CONFIG_FILES},
            fc_match=FC_WOFF,
            desktop=KDE,
        )
        self.assertIsNone(check)

    def test_woff_escalates(self) -> None:
        plain = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
            fc_match=FC_FALLBACK,
            desktop=GNOME,
        )
        woff = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
            fc_match=FC_WOFF,
            desktop=GNOME,
        )
        assert plain is not None and woff is not None
        self.assertEqual(woff.status, "warn")
        self.assertIn("fc-match sans resolves to a KaTeX .woff file.", woff.detail)
        self.assertGreater(len(woff.detail), len(plain.detail))
        self.assertNotIn("This session is KDE Plasma.", woff.detail)

    def test_kde_escalates(self) -> None:
        woff = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
            fc_match=FC_WOFF,
            desktop=GNOME,
        )
        kde = fonts_katex_check(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED},
            fc_match=FC_WOFF,
            desktop=KDE,
        )
        assert woff is not None and kde is not None
        self.assertEqual(kde.status, "warn")
        self.assertIn("fc-match sans resolves to a KaTeX .woff file.", kde.detail)
        self.assertIn("This session is KDE Plasma.", kde.detail)
        self.assertGreater(len(kde.detail), len(woff.detail))
        self.assertEqual(
            kde.detail,
            fonts_katex_detail(woff=True, plasma=True),
        )

    def test_pre_2604_is_skipped(self) -> None:
        with patch("probe.subprocess.run", side_effect=AssertionError):
            check = fonts_katex_check(
                os_release=OS_2404,
                dpkg_status={"fonts-katex": INSTALLED},
                fc_match=FC_WOFF,
                desktop=KDE,
            )
        self.assertIsNone(check)

    def test_unreadable_release_is_skipped(self) -> None:
        self.assertIsNone(
            fonts_katex_check(os_release="", dpkg_status={"fonts-katex": INSTALLED})
        )
        self.assertIsNone(
            fonts_katex_check(os_release=OS_DEBIAN, dpkg_status={"fonts-katex": INSTALLED})
        )
        self.assertIsNone(
            fonts_katex_check(os_release="ID=ubuntu\n", dpkg_status={"fonts-katex": INSTALLED})
        )


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
        self.assertIn("snap-only", check.detail)
        self.assertIn("Remove the apt package and use the snap if you decide to.", check.detail)
        self.assertNotIn("Mark", check.detail)
        self.assertNotEqual(check.status, "fail")

    def test_hold_counts_as_installed(self) -> None:
        self.assertTrue(dpkg_status_installed(INSTALLED))
        self.assertTrue(dpkg_status_installed(HELD))
        self.assertFalse(dpkg_status_installed(CONFIG_FILES))
        self.assertFalse(dpkg_status_installed(""))
        check = lemonade_apt_check(dpkg_status={"lemonade-server": HELD})
        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(check.status, "warn")
        self.assertEqual(check.detail, LEMONADE_APT_PRESENT)

    def test_snap_only_or_absent_has_no_apt_warn(self) -> None:
        for status in ("", CONFIG_FILES, "unknown ok not-installed"):
            check = lemonade_apt_check(dpkg_status={"lemonade-server": status})
            self.assertIsNone(check, status)
        self.assertIsNone(lemonade_apt_check(dpkg_status={}))
        self.assertIsNone(
            lemonade_apt_check(
                dpkg_status=lambda name: "" if name == "lemonade-server" else INSTALLED
            )
        )


class CollectHostGateTests(unittest.TestCase):
    def _collect(self, *, lemonade: str = "", **kwargs):
        with (
            patch("validate.dpkg_installed", return_value=False),
            patch("validate.lemonade_detect", return_value=lemonade),
            patch("validate.largest_gguf_bytes", return_value=0),
            patch("probe.subprocess.run") as run,
            patch("probe._OS_RELEASE") as release,
        ):
            release.read_text.side_effect = AssertionError("os-release")
            which = kwargs.pop("which", lambda name: None)
            checks = collect(
                _target(),
                _cpu(),
                which=which,
                apt_policy={},
                apt_sources="",
                **kwargs,
            )
        self.assertFalse(run.called)
        release.read_text.assert_not_called()
        return checks

    def test_collect_2604_present_warns(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED, "lemonade-server": ""},
            fc_match=FC_FALLBACK,
            desktop=GNOME,
        )
        fonts = _named(checks, "fonts-katex")
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].status, "warn")
        self.assertNotIn(".woff", fonts[0].detail)
        self.assertEqual(_named(checks, "lemonade-apt"), [])

    def test_collect_2604_absent_does_not_warn(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": "", "lemonade-server": ""},
            fc_match=FC_WOFF,
            desktop=KDE,
        )
        self.assertEqual(_named(checks, "fonts-katex"), [])
        self.assertEqual(_named(checks, "lemonade-apt"), [])

    def test_collect_woff_escalates(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED, "lemonade-server": ""},
            fc_match=FC_WOFF,
            desktop=GNOME,
        )
        fonts = _named(checks, "fonts-katex")
        self.assertEqual(len(fonts), 1)
        self.assertIn("KaTeX .woff", fonts[0].detail)
        self.assertNotIn("This session is KDE Plasma.", fonts[0].detail)

    def test_collect_kde_escalates(self) -> None:
        checks = self._collect(
            os_release=OS_2604,
            dpkg_status={"fonts-katex": INSTALLED, "lemonade-server": ""},
            fc_match=FC_WOFF,
            desktop=KDE,
        )
        fonts = _named(checks, "fonts-katex")
        self.assertEqual(len(fonts), 1)
        self.assertIn("KaTeX .woff", fonts[0].detail)
        self.assertIn("This session is KDE Plasma.", fonts[0].detail)
        self.assertIn(LP_KATEX, fonts[0].detail)

    def test_collect_pre_2604_skips_fonts(self) -> None:
        checks = self._collect(
            os_release=OS_2404,
            dpkg_status={"fonts-katex": INSTALLED, "lemonade-server": INSTALLED},
            fc_match=FC_WOFF,
            desktop=KDE,
        )
        self.assertEqual(_named(checks, "fonts-katex"), [])
        apt = _named(checks, "lemonade-apt")
        self.assertEqual(len(apt), 1)
        self.assertEqual(apt[0].status, "warn")
        self.assertIn("snap", apt[0].detail)

    def test_snap_present_without_deb_has_no_apt_warn(self) -> None:
        checks = self._collect(
            lemonade="snap",
            os_release=OS_2604,
            dpkg_status={"fonts-katex": "", "lemonade-server": ""},
            fc_match=FC_FALLBACK,
            desktop={},
            which=lambda name: "/snap/bin/lemonade-server" if name == "lemonade-server" else None,
        )
        self.assertEqual(_named(checks, "lemonade-apt"), [])
        self.assertEqual(_named(checks, "fonts-katex"), [])


class ReadOnlyCommandTests(unittest.TestCase):
    def test_injectors_skip_binaries(self) -> None:
        with patch("probe.subprocess.run", side_effect=AssertionError):
            fonts_katex_check(
                os_release=OS_2604,
                dpkg_status={"fonts-katex": INSTALLED},
                fc_match=FC_WOFF,
                desktop=KDE,
            )
            lemonade_apt_check(dpkg_status={"lemonade-server": HELD})
            ubuntu_version(OS_2604)
            fc_match_text(text=FC_WOFF)
            dpkg_query_status("fonts-katex", {"fonts-katex": INSTALLED})

    def test_live_dpkg_query_is_status_only(self) -> None:
        with patch("probe.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=HELD + "\n",
                stderr="",
            )
            text = dpkg_query_status("lemonade-server")
        self.assertEqual(text, HELD)
        self.assertTrue(dpkg_status_installed(text))
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
                stdout=FC_WOFF + "\n",
                stderr="",
            )
            text = fc_match_text()
        self.assertEqual(text, FC_WOFF)
        args = run.call_args[0][0]
        self.assertEqual(args, ["fc-match", "sans"])
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
        for token in ("subprocess", "apt-get", "fc-cache", "Conflicts"):
            self.assertNotIn(token, validate_text, token)
        for token in ('["purge"', "apt purge", "dpkg --purge", "dpkg -P"):
            self.assertNotIn(token, validate_text, token)
        detail = fonts_katex_detail(woff=True, plasma=True)
        self.assertIn("fontconfig", detail)
        self.assertIn("if you decide to", detail)
        self.assertIn("2168311", detail)
        self.assertNotIn("Mark", detail)
        self.assertNotIn("Mark", LEMONADE_APT_PRESENT)
        self.assertNotIn("Mark", validate_text)
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
                        "detail": fonts_katex_detail(woff=True, plasma=True),
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
        self.assertIn("2168311", joined)


if __name__ == "__main__":
    unittest.main()
