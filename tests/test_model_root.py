"""Default model folder is ~/AI models. Saved roots stay put."""

from __future__ import annotations

import io
import json
import os
import pwd
import re
import shlex
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import ROOT

from apply import quote_desktop_exec, run_privileged, write_core_files
from lemonade import (
    _escape_mount,
    _what_from_unit,
    _where_from_unit,
    mount_unit_text,
    quote_systemd_exec,
)
from main import _parser, config_main, installer_main
from paths import DEFAULT_MODEL_DIRNAME, LEGACY_MODEL_DIRNAME
from users import default_model_root, target_for
from vendor import LAUNCHER


HARDCODE = re.compile(r"~/Models|\$HOME/Models|(?<![A-Za-z0-9_])[\"']Models[\"']")

# Exact lines that still name the previous folder. None of them is the default.
ALLOWED_LINES = {
    (
        "DOCTRINE.md",
        "Default writable root is `~/AI models`. Existing trees such as `~/Models` are extra search paths. They are not overwritten.",
    ): "names the previous folder as a search path, not the writable default",
    (
        "DOCTRINE.md",
        "The Weights tab scans `~/AI models`, an existing `~/Models` tree, Downloads, Hugging Face cache, ComfyUI `models`, and `q4nx_files`, plus any folders the user adds. Added folders are stored in `~/.config/ubuntuai/config.json` as `scan_folders`. Scanning `/` is refused. New files are checkboxed. Organize offers copy or move. Symlink organize is not offered. Copy and move place a real file in the store. The original is removed only after the checksum of source and destination match. If the file is a catalog download, that checksum uses the algorithm published on its page. Move always asks for that removal. Copy asks with a checkbox. Catalog downloads are unchecked by default and land in the matching subdir.",
    ): "the scan list still includes the previous folder",
    (
        "usr/share/ubuntuai-installer/paths.py",
        'LEGACY_MODEL_DIRNAME = "Models"',
    ): "constant for the previous folder name. search only. not the default",
    (
        "AGENTS.md",
        "21. Lemonade must see the GGUF files the installer already has. The snap cannot follow `~/Models` symlinks and cannot read `/home` as `extra_models_dir`. Bind the real trees (often `~/AI models`) into `/var/snap/lemonade-server/common/ubuntuai-models` and set `extra_models_dir`.",
    ): "AGENTS.md is unchanged in this PR. The sentence names the previous folder in the Lemonade symlink note. A wording change needs owner approval.",
}

# ui/ is owned by another team. Hits belong here instead of an edit.
UI_FOLLOWUPS: tuple[str, ...] = ()


def _parse_desktop_exec(value: str) -> list[str]:
    args: list[str] = []
    buf: list[str] = []
    i = 0
    quoted = False
    while i < len(value):
        ch = value[i]
        if quoted:
            if ch == "\\" and i + 1 < len(value):
                buf.append(value[i + 1])
                i += 2
                continue
            if ch == '"':
                quoted = False
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        if ch.isspace():
            if buf:
                args.append("".join(buf))
                buf = []
            i += 1
            continue
        if ch == '"':
            quoted = True
            i += 1
            continue
        if ch == "\\" and i + 1 < len(value):
            buf.append(value[i + 1])
            i += 2
            continue
        buf.append(ch)
        i += 1
    if buf:
        args.append("".join(buf))
    return [arg.replace("%%", "%") for arg in args]


def _shipped_files() -> list[Path]:
    found: list[Path] = []
    for name in ("README.md", "DOCTRINE.md", "AGENTS.md", "STYLE.md", "Makefile"):
        path = ROOT / name
        if path.is_file():
            found.append(path)
    usr = ROOT / "usr"
    for path in usr.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        found.append(path)
    return found


class DefaultModelRootTests(unittest.TestCase):
    def _home_env(self, home: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("UBUNTUAI_MODELS", None)
        env.pop("XDG_CONFIG_HOME", None)
        env["HOME"] = str(home)
        return env

    def _pw(self, home: Path):
        class PW:
            pw_name = "tester"
            pw_uid = os.getuid()
            pw_gid = os.getgid()
            pw_dir = str(home)

        return PW()

    def test_default_is_ai_models_when_nothing_is_saved(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            legacy = home / "Models"
            legacy.mkdir()
            (legacy / "keep.gguf").write_bytes(b"g")
            self.assertEqual(default_model_root(home), home / "AI models")
            self.assertEqual(DEFAULT_MODEL_DIRNAME, "AI models")
            self.assertEqual(LEGACY_MODEL_DIRNAME, "Models")
            with (
                patch.dict(os.environ, self._home_env(home), clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", home / "missing.env"),
            ):
                target = target_for("tester")
            self.assertEqual(target.model_root, home / "AI models")
            self.assertEqual(str(target.model_root), str(home / "AI models"))
            self.assertIn(legacy, target.extra_model_paths)
            self.assertNotIn(home / "AI models", target.extra_model_paths)
            self.assertFalse((home / "AI models").exists())
            self.assertFalse((home / "AI models").is_symlink())
            self.assertEqual((legacy / "keep.gguf").read_bytes(), b"g")
            self.assertFalse(legacy.is_symlink())
            self.assertFalse(any(path.is_symlink() for path in home.rglob("*")))

    def test_saved_model_root_is_not_rewritten(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            saved = home / "Models"
            cfg = home / ".config" / "ubuntuai" / "config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps({"bind": "127.0.0.1", "model_root": str(saved)}, indent=2) + "\n",
                encoding="utf-8",
            )
            before = cfg.read_bytes()
            spaced = home / "Old models"
            env_path = home / "ubuntuai.env"
            env_path.write_text(
                f"UBUNTUAI_MODELS={shlex.quote(str(spaced))}\n",
                encoding="utf-8",
            )
            env_before = env_path.read_bytes()
            with (
                patch.dict(os.environ, self._home_env(home), clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", env_path),
            ):
                target = target_for("tester")
                again = target_for("tester", saved)
            self.assertEqual(str(target.model_root), str(saved))
            self.assertEqual(str(again.model_root), str(saved))
            self.assertEqual(cfg.read_bytes(), before)
            self.assertEqual(env_path.read_bytes(), env_before)
            self.assertFalse(saved.exists())
            self.assertFalse((home / "AI models").exists())

    def test_saved_spaced_root_and_env_value_stay_exact(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            saved = home / "Old models"
            cfg = home / ".config" / "ubuntuai" / "config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps({"model_root": str(saved)}) + "\n",
                encoding="utf-8",
            )
            before = cfg.read_bytes()
            with (
                patch.dict(os.environ, self._home_env(home), clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", home / "missing.env"),
            ):
                target = target_for("tester")
            self.assertEqual(str(target.model_root), str(saved))
            self.assertEqual(cfg.read_bytes(), before)

            cfg.unlink()
            env_path = home / "ubuntuai.env"
            env_path.write_text(
                f"UBUNTUAI_MODELS={shlex.quote(str(saved))}\n",
                encoding="utf-8",
            )
            env_before = env_path.read_bytes()
            live = home / "Live models"
            env = self._home_env(home)
            env["UBUNTUAI_MODELS"] = str(live)
            with (
                patch.dict(os.environ, env, clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", env_path),
            ):
                from_env = target_for("tester")
            self.assertEqual(str(from_env.model_root), str(live))
            self.assertEqual(env_path.read_bytes(), env_before)
            self.assertFalse(cfg.exists())

            env.pop("UBUNTUAI_MODELS")
            with (
                patch.dict(os.environ, env, clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", env_path),
            ):
                from_file = target_for("tester")
            self.assertEqual(str(from_file.model_root), str(saved))
            self.assertEqual(env_path.read_bytes(), env_before)


class SpaceSafeOutputTests(unittest.TestCase):
    def test_env_profile_and_argv_keep_the_spaced_path(self) -> None:
        pw = pwd.getpwuid(os.getuid())
        home = Path(pw.pw_dir)
        with TemporaryDirectory(dir=home) as tmp:
            root = Path(tmp) / "AI models"
            with TemporaryDirectory() as etc:
                etc_path = Path(etc)
                env = etc_path / "ubuntuai.env"
                profile = etc_path / "ubuntuai.sh"
                with (
                    patch("apply.ENV_FILE", env),
                    patch("apply.PROFILE_FILE", profile),
                    patch("apply.LIMITS_FILE", etc_path / "limits.conf"),
                ):
                    write_core_files(pw.pw_name, str(root), "127.0.0.1")
                text = env.read_text(encoding="utf-8")
                line = next(item for item in text.splitlines() if item.startswith("UBUNTUAI_MODELS="))
                self.assertEqual(line, f"UBUNTUAI_MODELS={shlex.quote(str(root.resolve()))}")
                sourced = subprocess.run(
                    ["sh", "-c", '. "$1"; printf %s "$UBUNTUAI_MODELS"', "sh", str(env)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(sourced.returncode, 0, sourced.stderr)
                self.assertEqual(sourced.stdout, str(root.resolve()))
                self.assertIn(" ", sourced.stdout)
                syntax = subprocess.run(
                    ["sh", "-n", str(profile)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                child = subprocess.run(
                    [
                        "sh",
                        "-c",
                        '. "$1"; printf %s "$UBUNTUAI_MODELS"',
                        "sh",
                        str(profile),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertEqual(child.stdout, str(root.resolve()))
            with (
                patch("apply.os.geteuid", return_value=0),
                patch("apply.subprocess.run") as run,
            ):
                run.return_value.returncode = 0
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                rc, _out = run_privileged(
                    "core-files",
                    [pw.pw_name, str(root), "127.0.0.1"],
                )
            self.assertEqual(rc, 0)
            cmd = run.call_args.args[0]
            self.assertIsInstance(cmd, list)
            self.assertIn(str(root), cmd)
            self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_launcher_shell_keeps_a_spaced_model_dir(self) -> None:
        script = LAUNCHER.format(
            libdir="/tmp/lib dir",
            port=8081,
            models_dir=DEFAULT_MODEL_DIRNAME,
        )
        self.assertIn('"${UBUNTUAI_MODELS:-$HOME/AI models}/openmoss"', script)
        self.assertNotIn("$HOME/Models", script)
        syntax = subprocess.run(
            ["sh", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "Ada Lovelace"
            home.mkdir()
            env = os.environ.copy()
            env.pop("UBUNTUAI_MODELS", None)
            env["HOME"] = str(home)
            unset = subprocess.run(
                ["sh", "-c", script],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn(str(home / "AI models" / "openmoss"), unset.stderr)
            env["UBUNTUAI_MODELS"] = str(home / "Old models")
            set_env = subprocess.run(
                ["sh", "-c", script],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn(str(home / "Old models" / "openmoss"), set_env.stderr)

    def test_desktop_exec_quotes_a_spaced_path(self) -> None:
        model = "/home/user/AI models/chat.gguf"
        argv = ["ubuntuai-openmoss", "--model", model]
        line = quote_desktop_exec(argv)
        self.assertIn('"/home/user/AI models/chat.gguf"', line)
        self.assertEqual(_parse_desktop_exec(line), argv)
        percent = "/tmp/AI 100% models"
        percent_line = quote_desktop_exec(["tool", percent])
        self.assertIn("100%%", percent_line)
        self.assertEqual(_parse_desktop_exec(percent_line), ["tool", percent])
        desktop_dir = ROOT / "usr" / "share" / "applications"
        seen = 0
        for path in sorted(desktop_dir.glob("*.desktop")):
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.startswith("Exec="):
                    continue
                seen += 1
                value = raw.split("=", 1)[1]
                parsed = _parse_desktop_exec(value)
                self.assertEqual(quote_desktop_exec(parsed), value)
                self.assertTrue(parsed)
                self.assertNotIn(" ", parsed[0])
        self.assertGreaterEqual(seen, 2)

    def test_systemd_escape_round_trip_for_a_spaced_path(self) -> None:
        escape = shutil.which("systemd-escape")
        if not escape:
            self.skipTest("systemd-escape is not installed")
        what = Path("/home/user/AI models/gguf")
        where = Path("/var/snap/lemonade-server/common/ubuntuai-models/AI models")
        text = mount_unit_text(what, where)
        self.assertEqual(_what_from_unit(text), str(what))
        self.assertEqual(_where_from_unit(text), where)
        self.assertNotIn('What="', text)
        self.assertNotIn('Where="', text)
        self.assertIn("What=/home/user/AI models/gguf\n", text)
        name = _escape_mount(where)
        self.assertNotIn(" ", name)
        self.assertIn(r"\x20", name)
        self.assertTrue(name.endswith(".mount"))
        back = subprocess.run(
            [escape, "--unescape", "-p", name.removesuffix(".mount")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(back.returncode, 0, back.stderr)
        self.assertEqual(back.stdout.strip(), str(where))
        analyze = shutil.which("systemd-analyze")
        # Unit-name escape already ran. ExecStart verify needs systemd-analyze.
        if not analyze:
            return
        model = "/home/user/AI models/chat.gguf"
        exec_line = quote_systemd_exec(["/usr/bin/true", model, "/tmp/100% x"])
        self.assertIn('"/home/user/AI models/chat.gguf"', exec_line)
        self.assertIn('"/tmp/100%% x"', exec_line)
        unit = (
            "[Unit]\n"
            "Description=quote test\n"
            "\n"
            "[Service]\n"
            "Type=oneshot\n"
            f"ExecStart={exec_line}\n"
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "quote-test.service"
            path.write_text(unit, encoding="utf-8")
            verified = subprocess.run(
                [analyze, "--man=no", "verify", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(verified.returncode, 0, verified.stderr)


class ShippedDefaultTests(unittest.TestCase):
    def test_help_names_the_new_default(self) -> None:
        installer = _parser().format_help()
        self.assertIn("~/AI models", installer)
        self.assertNotIn("~/Models", installer)
        buf = io.StringIO()
        with patch("sys.stdout", buf), self.assertRaises(SystemExit) as caught:
            config_main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        config = buf.getvalue()
        self.assertIn("~/AI models", config)
        self.assertNotIn("~/Models", config)
        installer_buf = io.StringIO()
        with patch("sys.stdout", installer_buf), self.assertRaises(SystemExit) as caught:
            installer_main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("~/AI models", installer_buf.getvalue())

    def test_no_shipped_file_hardcodes_the_old_default(self) -> None:
        hits: list[tuple[str, str]] = []
        ui_hits: list[str] = []
        seen: set[tuple[str, str]] = set()
        for path in _shipped_files():
            rel = path.relative_to(ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line in text.splitlines():
                if not HARDCODE.search(line):
                    continue
                key = (rel, line.strip())
                if rel.startswith("usr/share/ubuntuai-installer/ui/"):
                    ui_hits.append(f"{rel}: {line.strip()}")
                    continue
                seen.add(key)
                if key not in ALLOWED_LINES:
                    hits.append(key)
        self.assertEqual(hits, [], msg="\n".join(f"{path}: {line}" for path, line in hits))
        self.assertEqual(sorted(seen), sorted(ALLOWED_LINES))
        self.assertEqual(tuple(ui_hits), UI_FOLLOWUPS)
        for rel in (
            "README.md",
            "DOCTRINE.md",
            "usr/share/man/man1/ubuntuai-installer.1",
            "usr/share/man/man1/ubuntuai-config.1",
            "usr/share/man/man1/ubuntuai-validate.1",
            "usr/share/ubuntuai-installer/workflows.json",
        ):
            self.assertIn("~/AI models", (ROOT / rel).read_text(encoding="utf-8"), rel)


if __name__ == "__main__":
    unittest.main()
