"""Apply progress events, English copy, and on-disk logs."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from domain import Action
from weights import human_bytes


@dataclass(frozen=True)
class ProgressEvent:
    english: str
    technical: str = ""
    fraction: float = 0.0
    done: bool = False
    failed: bool = False

    def __str__(self) -> str:
        return self.english


def apply_log_dir(home: Path) -> Path:
    return home / ".local" / "share" / "ubuntuai" / "logs"


def _log_owner(uid: int | None, gid: int | None) -> tuple[int, int]:
    if uid is not None and gid is not None:
        return uid, gid
    from users import guess_user, target_for

    try:
        target = target_for(guess_user())
    except (RuntimeError, KeyError, OSError):
        return os.getuid(), os.getgid()
    return target.uid, target.gid


def _own_log_paths(path: Path, last: Path, folder: Path, home: Path, uid: int, gid: int) -> None:
    os.chown(path, uid, gid)
    if last.exists(follow_symlinks=False):
        # last.log is usually a symlink. Default chown follows and leaves the inode root-owned.
        os.chown(last, uid, gid, follow_symlinks=False)
    home_r = home.resolve()
    cur = folder
    while True:
        try:
            cur.resolve().relative_to(home_r)
        except ValueError:
            break
        if cur.resolve() == home_r:
            break
        os.chown(cur, uid, gid)
        nxt = cur.parent
        if nxt == cur:
            break
        cur = nxt


def new_apply_log(
    home: Path, uid: int | None = None, gid: int | None = None
) -> Path:
    owner_uid, owner_gid = _log_owner(uid, gid)
    folder = apply_log_dir(home)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = folder / f"apply-{stamp}.log"
    last = folder / "last.log"
    try:
        if last.exists() or last.is_symlink():
            last.unlink()
        last.symlink_to(path.name)
    except OSError:
        last.write_text("", encoding="utf-8")
    path.write_text("", encoding="utf-8")
    # Apply may run as root. Own the tree as the desktop user from guess_user().
    _own_log_paths(path, last, folder, home, owner_uid, owner_gid)
    return path


def append_log(path: Path, event: ProgressEvent) -> None:
    lines = [time.strftime("%H:%M:%S") + "  " + event.english]
    if event.technical:
        for raw in event.technical.splitlines() or [event.technical]:
            lines.append("  " + raw)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def english_for_action(action: Action) -> str:
    kind = action.kind
    if kind == "skip":
        return f"Skipping {action.payload[0] if action.payload else 'a workflow'} on this hardware."
    if kind == "apt_install":
        names = ", ".join(action.payload[:5])
        more = " and more" if len(action.payload) > 5 else ""
        return f"Installing Ubuntu packages ({names}{more})."
    if kind == "groups":
        return "Adding this user to the graphics groups."
    if kind == "core_files":
        return "Writing environment, memlock, and PATH files."
    if kind == "model_dirs":
        return "Creating folders in the model store."
    if kind == "vendor":
        name = action.payload[0] if action.payload else "the extra runtime"
        return f"Installing {name} into your home folder."
    if kind == "weights":
        return "Finding or downloading required model files."
    if kind == "lemonade":
        return "Making downloaded GGUF files visible to Lemonade."
    return action.summary


def emit(on_progress: object | None, event: ProgressEvent, log_path: Path | None = None) -> None:
    if log_path is not None:
        append_log(log_path, event)
    if on_progress is None:
        return
    on_progress(event)


def download_english(title: str, done: int, total: int) -> str:
    if total > 0:
        return f"Downloading {title} ({human_bytes(done)} of {human_bytes(total)})."
    return f"Downloading {title} ({human_bytes(done)} so far)."
