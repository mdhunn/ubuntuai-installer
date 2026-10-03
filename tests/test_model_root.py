"""Default model folder is ~/AI models. Saved roots stay put."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import ROOT

from apply import execute_plan, run_privileged, write_core_files
from domain import Action, UserTarget
from lemonade import (
    _escape_mount,
    _what_from_unit,
    _where_from_unit,
    mount_unit_text,
)
from main import _parser, config_main, installer_main
from paths import DEFAULT_MODEL_DIRNAME, LEGACY_MODEL_DIRNAME
from users import default_model_root, target_for
from vendor import launcher_script
from weights import load_catalog


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


def _passwd(home: Path):
    class PW:
        pw_name = "tester"
        pw_uid = os.getuid()
        pw_gid = os.getgid()
        pw_dir = str(home)

    return PW()


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
        return _passwd(home)

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

    def test_tilde_uses_passwd_home_when_process_home_is_root(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = home / ".config" / "ubuntuai" / "config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps({"model_root": "~/Models"}) + "\n",
                encoding="utf-8",
            )
            before = cfg.read_bytes()
            env = self._home_env(home)
            env["HOME"] = "/root"
            with (
                patch.dict(os.environ, env, clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", home / "missing.env"),
            ):
                target = target_for("tester")
            self.assertEqual(target.model_root, home / "Models")
            self.assertNotEqual(Path(target.model_root), Path("/root/Models"))
            self.assertEqual(cfg.read_bytes(), before)

            cfg.unlink()
            env_path = home / "ubuntuai.env"
            env_path.write_text("UBUNTUAI_MODELS='~/Models'\n", encoding="utf-8")
            env_before = env_path.read_bytes()
            with (
                patch.dict(os.environ, env, clear=True),
                patch("users._pw", return_value=self._pw(home)),
                patch("users.ENV_FILE", env_path),
            ):
                from_file = target_for("tester")
            self.assertEqual(from_file.model_root, home / "Models")
            self.assertFalse(str(from_file.model_root).startswith("/root"))
            self.assertEqual(env_path.read_bytes(), env_before)


class SpaceSafeOutputTests(unittest.TestCase):
    def test_env_profile_and_argv_keep_the_spaced_path(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            pw = _passwd(home)
            root = home / "AI models"
            with TemporaryDirectory() as etc:
                etc_path = Path(etc)
                env = etc_path / "ubuntuai.env"
                profile = etc_path / "ubuntuai.sh"
                with (
                    patch("apply.pwd.getpwnam", return_value=pw),
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

    def _fake_openmoss(self, lib: Path) -> None:
        lib.mkdir(parents=True, exist_ok=True)
        server = lib / "moss-tts-server"
        server.write_text('#!/bin/sh\nprintf %s "$2"\n', encoding="utf-8")
        server.chmod(0o755)

    def _launcher_env(self, home: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("UBUNTUAI_MODELS", None)
        env.pop("XDG_CONFIG_HOME", None)
        env["HOME"] = str(home)
        env["UBUNTUAI_ENV_FILE"] = str(home / "missing.env")
        return env

    def _run_launcher(self, script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_launcher_shell_keeps_a_spaced_model_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "Ada Lovelace"
            lib = home / "lib dir"
            self._fake_openmoss(lib)
            baked = home / "AI models"
            script = launcher_script(lib, 8081, baked)
            syntax = subprocess.run(
                ["sh", "-n"],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            self.assertIn(shlex.quote(str(baked)), script)
            self.assertNotIn("$HOME/Models}/openmoss", script)
            env = self._launcher_env(home)
            missing = self._run_launcher(script, env)
            self.assertIn("no GGUF", missing.stderr)
            self.assertIn(str(baked / "openmoss"), missing.stderr)
            env["UBUNTUAI_MODELS"] = str(home / "Old models")
            overridden = self._run_launcher(script, env)
            self.assertIn(str(home / "Old models" / "openmoss"), overridden.stderr)
            self.assertNotIn("no GGUF in " + str(baked), overridden.stderr)

    def test_launcher_finds_legacy_openmoss_without_env(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            lib = home / "lib"
            self._fake_openmoss(lib)
            legacy = home / "Models" / "openmoss"
            legacy.mkdir(parents=True)
            gguf = legacy / "moss-tts-local-q.gguf"
            gguf.write_bytes(b"g" * 32)
            (legacy / "skip.extras.gguf").write_bytes(b"x")
            script = launcher_script(lib, 8081, home / "AI models")
            found = self._run_launcher(script, self._launcher_env(home))
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertEqual(found.stdout, str(gguf))
            self.assertNotIn("no GGUF", found.stderr)

    def test_launcher_honors_config_and_env_file(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            lib = home / "lib"
            self._fake_openmoss(lib)
            legacy = home / "Models" / "openmoss"
            legacy.mkdir(parents=True)
            (legacy / "legacy.gguf").write_bytes(b"L")
            custom = home / "Custom models" / "openmoss"
            custom.mkdir(parents=True)
            wanted = custom / "custom.gguf"
            wanted.write_bytes(b"C")
            other = home / "Other models" / "openmoss"
            other.mkdir(parents=True)
            other_file = other / "other.gguf"
            other_file.write_bytes(b"O")
            xdg = home / "cfg"
            cfg = xdg / "ubuntuai" / "config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps({"model_root": "~/Custom models"}) + "\n",
                encoding="utf-8",
            )
            script = launcher_script(lib, 8081, home / "AI models")
            env = self._launcher_env(home)
            env["XDG_CONFIG_HOME"] = str(xdg)
            from_config = self._run_launcher(script, env)
            self.assertEqual(from_config.returncode, 0, from_config.stderr)
            self.assertEqual(from_config.stdout, str(wanted))

            cfg.unlink()
            env_path = home / "ubuntuai.env"
            env_path.write_text(
                f"UBUNTUAI_MODELS={shlex.quote('~/Other models')}\n",
                encoding="utf-8",
            )
            env = self._launcher_env(home)
            env["UBUNTUAI_ENV_FILE"] = str(env_path)
            from_env = self._run_launcher(script, env)
            self.assertEqual(from_env.returncode, 0, from_env.stderr)
            self.assertEqual(from_env.stdout, str(other_file))

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


class LegacyWeightTests(unittest.TestCase):
    def test_dry_run_plans_no_symlink_for_a_legacy_whisper_file(self) -> None:
        model = next(w for w in load_catalog() if w.id == "whisper-base-en")
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            legacy_dir = home / "Models" / "whisper"
            legacy_dir.mkdir(parents=True)
            blob = legacy_dir / model.filename
            payload = b"g" * (128 * 1024)
            blob.write_bytes(payload)
            root = home / "AI models"
            target = UserTarget(
                name="tester",
                uid=os.getuid(),
                gid=os.getgid(),
                home=home,
                model_root=root,
            )
            actions = (
                Action("model_dirs", f"create dirs under {root}", ("whisper",)),
                Action("weights", "copy or download required weights", (model.id,)),
            )
            with (
                patch("apply.pwd.getpwnam", return_value=_passwd(home)),
                patch("apply.saved_scan_folders", return_value=()),
            ):
                log = execute_plan(actions, target, dry_run=True)
            joined = "\n".join(log)
            self.assertNotIn("symlink", joined)
            self.assertFalse(any(line.startswith("link ") for line in log))
            self.assertFalse(any(line.startswith("move ") for line in log))
            self.assertTrue(any(line.startswith("copy ") for line in log))
            self.assertIn(str(blob), joined)
            self.assertIn(str(root / "whisper" / model.filename), joined)
            self.assertEqual(blob.read_bytes(), payload)
            self.assertFalse(blob.is_symlink())
            self.assertFalse((root / "whisper" / model.filename).exists())
            self.assertFalse(any(path.is_symlink() for path in home.rglob("*")))


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
