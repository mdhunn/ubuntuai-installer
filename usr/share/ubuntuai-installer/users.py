"""Resolve the desktop user the installer is acting for."""

from __future__ import annotations

import json
import os
import pwd
import subprocess
from pathlib import Path

from domain import UserTarget
from paths import user_config_path


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
    preferred = home / "Models"
    legacy = home / "AI models"
    if preferred.exists():
        return preferred
    if legacy.exists() and os.access(legacy, os.W_OK):
        return legacy
    return preferred


def extra_model_paths(home: Path, model_root: Path) -> tuple[Path, ...]:
    extras: list[Path] = []
    legacy = home / "AI models"
    if legacy.exists() and legacy.resolve() != model_root.resolve():
        extras.append(legacy)
    return tuple(extras)


def load_saved_bind(home: Path) -> str:
    cfg = user_config_path(home)
    if not cfg.is_file():
        return "127.0.0.1"
    try:
        import json

        data = json.loads(cfg.read_text(encoding="utf-8"))
        bind = data.get("bind") or "127.0.0.1"
        if bind in {"127.0.0.1", "0.0.0.0"}:
            return bind
    except (OSError, json.JSONDecodeError):
        return "127.0.0.1"
    return "127.0.0.1"


def target_for(user: str, model_root: Path | None = None) -> UserTarget:
    pw = _pw(user)
    home = Path(pw.pw_dir)
    root = Path(model_root) if model_root else default_model_root(home)
    return UserTarget(
        name=pw.pw_name,
        uid=pw.pw_uid,
        gid=pw.pw_gid,
        home=home,
        model_root=root,
        extra_model_paths=extra_model_paths(home, root),
        bind=load_saved_bind(home),
    )
