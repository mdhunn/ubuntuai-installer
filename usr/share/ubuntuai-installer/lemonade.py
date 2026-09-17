"""Point Lemonade at GGUF files the installer already downloaded."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from domain import UserTarget
from weights import UA

SNAP_COMMON = Path("/var/snap/lemonade-server/common")
SNAP_EXTRA = SNAP_COMMON / "ubuntuai-models"
LEMONADE_API = "http://127.0.0.1:13305"


def detect() -> str:
    if SNAP_COMMON.is_dir() or Path("/snap/bin/lemonade-server").exists():
        return "snap"
    if shutil.which("lemonade-server") or shutil.which("lemonade"):
        return "cli"
    if Path("/var/lib/lemonade").is_dir():
        return "deb"
    return ""


def extra_dir(kind: str) -> Path:
    if kind == "snap":
        return SNAP_EXTRA
    return Path()


def gguf_sources(target: UserTarget) -> tuple[Path, ...]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in (*target.extra_model_paths, target.model_root / "gguf"):
        try:
            path = raw.resolve()
        except OSError:
            continue
        if not path.is_dir() or path in seen:
            continue
        if not _has_real_gguf(path):
            continue
        seen.add(path)
        out.append(path)
    return tuple(out)


def _has_real_gguf(root: Path) -> bool:
    for p in root.rglob("*.gguf"):
        if p.is_symlink():
            continue
        if p.is_file():
            return True
    return False


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _escape_mount(where: Path) -> str:
    p = _run(["systemd-escape", "-p", "--suffix=mount", str(where)])
    name = (p.stdout or "").strip()
    if p.returncode != 0 or not name:
        raise RuntimeError(f"systemd-escape failed for {where}")
    return name


def _write_bind_unit(what: Path, where: Path) -> str:
    where.mkdir(parents=True, exist_ok=True)
    unit = _escape_mount(where)
    text = (
        "[Unit]\n"
        "Description=Ubuntu AI models for Lemonade\n"
        "After=local-fs.target\n"
        "Before=snap.lemonade-server.daemon.service\n"
        "\n"
        "[Mount]\n"
        f"What={what}\n"
        f"Where={where}\n"
        "Type=none\n"
        "Options=bind\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )
    path = Path("/etc/systemd/system") / unit
    path.write_text(text, encoding="utf-8")
    _run(["systemctl", "daemon-reload"])
    enabled = _run(["systemctl", "enable", "--now", unit])
    if enabled.returncode != 0:
        mounted = _run(["mount", "--bind", str(what), str(where)])
        if mounted.returncode != 0:
            raise RuntimeError(
                (enabled.stderr or enabled.stdout or "")
                + (mounted.stderr or mounted.stdout or "")
            )
    return unit


def _set_extra_models_dir(path: Path) -> None:
    body = json.dumps({"extra_models_dir": str(path)}).encode("utf-8")
    req = urllib.request.Request(
        f"{LEMONADE_API}/internal/set",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read().decode("utf-8"))
            return
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    exe = shutil.which("lemonade-server") or shutil.which("lemonade")
    if not exe:
        raise RuntimeError("lemonade-server is not on PATH")
    p = _run([exe, "config", "set", f"extra_models_dir={path}"])
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "lemonade config set failed").strip())


def _restart_snap() -> None:
    _run(["snap", "restart", "lemonade-server.daemon"])


def publish(target: UserTarget) -> str:
    kind = detect()
    if not kind:
        return "lemonade not installed"
    sources = gguf_sources(target)
    if not sources:
        return "no real GGUF files to publish (Lemonade cannot follow store symlinks)"
    if kind == "snap":
        dest = extra_dir(kind)
        if len(sources) == 1:
            unit = _write_bind_unit(sources[0], dest)
            _set_extra_models_dir(dest)
            _restart_snap()
            return f"lemonade extra_models_dir={dest} via {unit}"
        dest.mkdir(parents=True, exist_ok=True)
        chat = dest / "chat"
        chat.mkdir(exist_ok=True)
        for i, src in enumerate(sources):
            _write_bind_unit(src, chat / f"src{i}")
        _set_extra_models_dir(dest)
        _restart_snap()
        return f"lemonade extra_models_dir={dest} ({len(sources)} trees)"
    dest = sources[0]
    _set_extra_models_dir(dest)
    return f"lemonade extra_models_dir={dest}"
