"""User config at ~/.config/ubuntuai/config.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from paths import user_config_path
from users import target_for
from weights import normalize_scan_folder


def load(user: str) -> dict:
    t = target_for(user)
    path = user_config_path(t.home)
    data = {
        "model_root": str(t.model_root),
        "extra_model_paths": [str(p) for p in t.extra_model_paths],
        "scan_folders": [],
        "bind": t.bind,
        "user": t.name,
        "chat_model": None,
        "primary_backend": None,
        "openai_base_url": "",
        "openai_api_key": "",
        "repair_advisor": "classical",
        "installed_workflows": [],
        "tts_engine": "",
        "stt_engine": "",
    }
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except json.JSONDecodeError:
            pass
    return data


def save(user: str, updates: dict) -> Path:
    t = target_for(user)
    path = user_config_path(t.home)
    data = load(user)
    data.update(updates)
    if data.get("bind") not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError("bind must be 127.0.0.1 or 0.0.0.0")
    raw_root = str(data.get("model_root") or "").strip()
    if raw_root:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = t.home / root
        try:
            root.resolve(strict=False).relative_to(t.home.resolve())
        except ValueError as exc:
            raise ValueError("model root must stay under the home directory") from exc
        data["model_root"] = str(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def saved_scan_folders(user: str) -> tuple[Path, ...]:
    t = target_for(user)
    raw = load(user).get("scan_folders") or []
    out: list[Path] = []
    seen: set[Path] = set()
    for item in raw:
        try:
            folder = normalize_scan_folder(item, t.home)
        except ValueError:
            continue
        if folder in seen:
            continue
        seen.add(folder)
        out.append(folder)
    return tuple(out)


def set_scan_folders(user: str, folders: tuple[Path, ...]) -> tuple[Path, ...]:
    t = target_for(user)
    unique: list[Path] = []
    seen: set[Path] = set()
    for raw in folders:
        folder = normalize_scan_folder(raw, t.home)
        if folder in seen:
            continue
        seen.add(folder)
        unique.append(folder)
    save(user, {"scan_folders": [str(p) for p in unique]})
    return tuple(unique)


def add_scan_folder(user: str, raw: str | Path) -> Path:
    folder = normalize_scan_folder(raw, target_for(user).home)
    current = saved_scan_folders(user)
    if folder not in current:
        set_scan_folders(user, current + (folder,))
    return folder


def remove_scan_folder(user: str, raw: str | Path) -> None:
    folder = normalize_scan_folder(raw, target_for(user).home)
    kept = tuple(p for p in saved_scan_folders(user) if p != folder)
    set_scan_folders(user, kept)


def record_installed(user: str, ids: tuple[str, ...]) -> None:
    have = list(load(user).get("installed_workflows") or [])
    for wid in ids:
        if wid and wid not in have:
            have.append(wid)
    save(user, {"installed_workflows": have})
