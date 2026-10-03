"""Resolve the desktop user the installer is acting for."""

from __future__ import annotations

import json
import os
import pwd
import shlex
import subprocess
from pathlib import Path

from domain import UserTarget
from paths import (
    DEFAULT_MODEL_DIRNAME,
    ENV_FILE,
    LEGACY_MODEL_DIRNAME,
    user_config_path,
)


def _pw(name: str) -> pwd.struct_passwd:
    return pwd.getpwnam(name)


def guess_user(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    override = (os.environ.get("UBUNTUAI_USER") or "").strip()
    if override and override != "root":
        return override
    pk = os.environ.get("PKEXEC_UID")
    if pk and pk.isdigit() and int(pk) != 0:
        return pwd.getpwuid(int(pk)).pw_name
    sudo = (os.environ.get("SUDO_USER") or "").strip()
    if sudo and sudo != "root":
        return sudo
    uid = os.getuid()
    if uid != 0:
        return pwd.getpwuid(uid).pw_name
    seated = _seated_user()
    if seated:
        return seated
    raise RuntimeError(
        "cannot guess a desktop user from the environment. pass --user"
    )


def _seated_user() -> str | None:
    try:
        p = subprocess.run(
            ["loginctl", "list-sessions", "--no-legend"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] != "root":
            return parts[2]
    return None


def default_model_root(home: Path) -> Path:
    return home / DEFAULT_MODEL_DIRNAME


def extra_model_paths(home: Path, model_root: Path) -> tuple[Path, ...]:
    try:
        chosen = model_root.resolve()
    except OSError:
        chosen = model_root
    extras: list[Path] = []
    for name in (LEGACY_MODEL_DIRNAME, DEFAULT_MODEL_DIRNAME):
        path = home / name
        if not path.exists():
            continue
        try:
            same = path.resolve() == chosen
        except OSError:
            same = False
        if same:
            continue
        extras.append(path)
    return tuple(extras)


def _saved_cfg(home: Path) -> dict:
    cfg = user_config_path(home)
    if not cfg.is_file():
        return {}
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_saved_bind(home: Path) -> str:
    bind = _saved_cfg(home).get("bind") or "127.0.0.1"
    if bind in {"127.0.0.1", "0.0.0.0"}:
        return bind
    return "127.0.0.1"


def _path_under_home(home: Path, raw: str) -> Path | None:
    text = raw.strip()
    if not text:
        return None
    root = Path(text).expanduser()
    if not root.is_absolute():
        root = home / root
    try:
        root.resolve(strict=False).relative_to(home.resolve())
    except ValueError:
        return None
    return root


def load_saved_model_root(home: Path) -> Path | None:
    return _path_under_home(home, str(_saved_cfg(home).get("model_root") or ""))


def _env_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    name, sep, raw = text.partition("=")
    if sep != "=":
        return None
    name = name.strip()
    if not name:
        return None
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        return None
    if len(parts) != 1:
        return None
    return name, parts[0]


def _env_file_value(key: str, path: Path | None = None) -> str:
    src = path if path is not None else ENV_FILE
    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return ""
    found = ""
    for line in text.splitlines():
        parsed = _env_assignment(line)
        if parsed is None:
            continue
        name, value = parsed
        if name == key:
            found = value
    return found


def load_env_model_root(home: Path) -> Path | None:
    raw = os.environ.get("UBUNTUAI_MODELS") or ""
    if raw.strip():
        found = _path_under_home(home, raw)
        if found is not None:
            return found
    return _path_under_home(home, _env_file_value("UBUNTUAI_MODELS"))


def target_for(user: str, model_root: Path | None = None) -> UserTarget:
    pw = _pw(user)
    home = Path(pw.pw_dir)
    if model_root is not None:
        root = Path(model_root)
    else:
        root = (
            load_saved_model_root(home)
            or load_env_model_root(home)
            or default_model_root(home)
        )
    return UserTarget(
        name=pw.pw_name,
        uid=pw.pw_uid,
        gid=pw.pw_gid,
        home=home,
        model_root=root,
        extra_model_paths=extra_model_paths(home, root),
        bind=load_saved_bind(home),
    )
