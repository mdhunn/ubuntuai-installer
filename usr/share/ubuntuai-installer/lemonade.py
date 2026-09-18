"""Publish installer GGUF trees to Lemonade.

The snap cannot follow ~/Models symlinks and cannot read /home as
extra_models_dir. Bind the real trees into SNAP_LEMONADE_MODELS and set
extra_models_dir to that snap-common path.

Apply runs helper verb lemonade-publish after weights. That verb calls
publish(target_for(USER)). CLI --publish-lemonade is the same verb.
Weights owns source resolution and bind targets. Apply owns privilege
and order.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from domain import UserTarget
from paths import (
    SNAP_LEMONADE_COMMON,
    SNAP_LEMONADE_MODELS,
    is_home_path,
    lemonade_extra_models_dir,
)
from weights import UA

SNAP_COMMON = SNAP_LEMONADE_COMMON
SNAP_EXTRA = SNAP_LEMONADE_MODELS
LEMONADE_API = "http://127.0.0.1:13305"
APPLY_PUBLISH_VERB = "lemonade-publish"
OWNED_UNIT_DESC = "Ubuntu AI models for Lemonade"
SYSTEM_UNIT_DIR = Path("/etc/systemd/system")


@dataclass(frozen=True)
class BindMount:
    what: Path
    where: Path


def detect() -> str:
    if SNAP_COMMON.is_dir() or Path("/snap/bin/lemonade-server").exists():
        return "snap"
    if shutil.which("lemonade-server") or shutil.which("lemonade"):
        return "cli"
    if Path("/var/lib/lemonade").is_dir():
        return "deb"
    return ""


def extra_dir(kind: str) -> Path:
    return lemonade_extra_models_dir(kind)


def gguf_sources(target: UserTarget) -> tuple[Path, ...]:
    """Real GGUF trees. Store symlinks are followed. HF and Diffusers stay put."""
    trees: list[Path] = []
    seen: set[Path] = set()
    for root in _search_roots(target):
        for tree in _trees_in(root, target):
            if tree in seen or _too_wide(tree, target.home):
                continue
            seen.add(tree)
            trees.append(tree)
    return _collapse(tuple(trees))


def bind_mounts(sources: tuple[Path, ...], dest: Path) -> tuple[BindMount, ...]:
    """Map real trees onto dest. Nested sources bind once via the outer tree."""
    sources = _collapse(sources)
    if not sources:
        return ()
    if len(sources) == 1:
        return (BindMount(sources[0], dest),)
    return tuple(
        BindMount(src, dest / "chat" / f"src{i}") for i, src in enumerate(sources)
    )


def is_owned_lemonade_where(dest: Path, where: Path) -> bool:
    """True for dest itself or dest/chat/srcN. Those are the only bind targets we create."""
    dest = _resolve(dest)
    where = _resolve(where)
    if where == dest:
        return True
    try:
        rel = where.relative_to(dest)
    except ValueError:
        return False
    return len(rel.parts) == 2 and rel.parts[0] == "chat" and _is_srcn(rel.parts[1])


def leftover_owned_binds(
    dest: Path,
    plan: tuple[BindMount, ...],
    *,
    unit_dir: Path,
    mounted: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """ubuntuai-owned dest / dest/chat/srcN binds that the collapsed plan no longer uses."""
    planned = {_resolve(mount.where) for mount in plan}
    leftovers = [
        where
        for where in owned_bind_wheres(dest, unit_dir, mounted)
        if _resolve(where) not in planned
    ]
    leftovers.sort(key=lambda path: (-len(_resolve(path).parts), str(_resolve(path))))
    return tuple(leftovers)


def owned_bind_wheres(
    dest: Path,
    unit_dir: Path,
    mounted: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Discover dest and dest/chat/srcN locations we created. Skip foreign units."""
    dest = _resolve(dest)
    found: list[Path] = []
    seen: set[Path] = set()
    foreign: set[Path] = set()

    def add(where: Path) -> None:
        where = _resolve(where)
        if where in seen or where in foreign:
            return
        if not is_owned_lemonade_where(dest, where):
            return
        seen.add(where)
        found.append(where)

    if unit_dir.is_dir():
        for path in sorted(unit_dir.glob("*.mount")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            where = _where_from_unit(text)
            if where is None:
                continue
            if _unit_is_ours(path):
                add(where)
            elif is_owned_lemonade_where(dest, where):
                foreign.add(_resolve(where))
    for where in mounted:
        add(where)
    return tuple(found)


def quote_unit_path(path: Path) -> str:
    # Spaces in ~/AI models must survive the systemd unit parser.
    text = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def mount_unit_text(what: Path, where: Path) -> str:
    return (
        "[Unit]\n"
        f"Description={OWNED_UNIT_DESC}\n"
        "After=local-fs.target\n"
        "Before=snap.lemonade-server.daemon.service\n"
        "\n"
        "[Mount]\n"
        f"What={quote_unit_path(what)}\n"
        f"Where={quote_unit_path(where)}\n"
        "Type=none\n"
        "Options=bind\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _search_roots(target: UserTarget) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(raw: Path) -> None:
        try:
            path = raw.resolve()
        except OSError:
            return
        if not path.is_dir() or path in seen:
            return
        seen.add(path)
        roots.append(path)

    for extra in target.extra_model_paths:
        add(extra)
    add(target.model_root / "gguf")
    gguf = target.model_root / "gguf"
    if not _any_gguf(gguf):
        add(target.model_root)
    return tuple(roots)


def _any_gguf(root: Path) -> bool:
    try:
        path = root.resolve()
    except OSError:
        return False
    if not path.is_dir():
        return False
    try:
        found = path.rglob("*.gguf")
    except OSError:
        return False
    for p in found:
        if p.is_file() or p.is_symlink():
            return True
    return False


def _trees_in(root: Path, target: UserTarget) -> tuple[Path, ...]:
    try:
        files = list(root.rglob("*.gguf"))
    except OSError:
        return ()
    has_real_here = False
    escaped: list[Path] = []
    for p in files:
        try:
            if p.is_symlink():
                real = p.resolve()
                if real.is_file():
                    escaped.append(real)
            elif p.is_file():
                has_real_here = True
        except OSError:
            continue
    found: list[Path] = []
    if has_real_here:
        found.append(root)
    seen: set[Path] = set()
    for real in escaped:
        tree = _lift(real, root, target)
        if tree in seen:
            continue
        seen.add(tree)
        found.append(tree)
    return tuple(found)


def _lift(real_file: Path, search_root: Path, target: UserTarget) -> Path:
    candidates: list[Path] = []
    for extra in target.extra_model_paths:
        try:
            candidates.append(extra.resolve())
        except OSError:
            continue
    try:
        candidates.append(search_root.resolve())
    except OSError:
        pass
    for cand in candidates:
        try:
            real_file.relative_to(cand)
            return cand
        except ValueError:
            continue
    return real_file.parent


def _too_wide(path: Path, home: Path) -> bool:
    try:
        path = path.resolve()
        home = home.resolve()
    except OSError:
        return True
    banned = {Path("/"), Path("/home"), Path("/var"), Path("/usr"), Path("/etc")}
    return path in banned or path == home


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _nested_under(path: Path, parent: Path) -> bool:
    return parent in path.parents


def _collapse(trees: tuple[Path, ...]) -> tuple[Path, ...]:
    """Keep outer trees only. A source nested under another selected source is dropped."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for tree in trees:
        path = _resolve(tree)
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    kept: list[Path] = []
    for tree in resolved:
        if any(tree != other and _nested_under(tree, other) for other in resolved):
            continue
        kept.append(tree)
    return tuple(kept)


def _is_srcn(name: str) -> bool:
    return name.startswith("src") and name[3:].isdigit()


def _unit_is_ours(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return f"Description={OWNED_UNIT_DESC}" in text


def _where_from_unit(text: str) -> Path | None:
    for line in text.splitlines():
        if not line.startswith("Where="):
            continue
        raw = line.split("=", 1)[1].strip()
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            raw = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return Path(raw) if raw else None
    return None


def _owned_unit_path(where: Path) -> Path | None:
    where = _resolve(where)
    if not SYSTEM_UNIT_DIR.is_dir():
        return None
    for path in SYSTEM_UNIT_DIR.glob("*.mount"):
        if not _unit_is_ours(path):
            continue
        parsed = _where_from_unit(path.read_text(encoding="utf-8"))
        if parsed is not None and _resolve(parsed) == where:
            return path
    return None


def _mounted_wheres() -> tuple[Path, ...]:
    info = Path("/proc/self/mountinfo")
    try:
        text = info.read_text(encoding="utf-8")
    except OSError:
        return ()
    found: list[Path] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        mountpoint = parts[4].replace("\\040", " ").replace("\\011", "\t")
        found.append(Path(mountpoint))
    return tuple(found)


def _drop_bind_unit(where: Path, dest: Path) -> None:
    if not is_owned_lemonade_where(dest, where):
        return
    path = _owned_unit_path(where)
    if path is not None:
        _run(["systemctl", "disable", "--now", path.name])
        path.unlink(missing_ok=True)
        _run(["systemctl", "daemon-reload"])
    else:
        try:
            unit = _escape_mount(where)
        except RuntimeError:
            unit = ""
        if unit:
            _run(["systemctl", "disable", "--now", unit])
    _run(["umount", str(where)])


def _drop_obsolete_owned_binds(
    dest: Path, plan: tuple[BindMount, ...]
) -> tuple[Path, ...]:
    leftovers = leftover_owned_binds(
        dest, plan, unit_dir=SYSTEM_UNIT_DIR, mounted=_mounted_wheres()
    )
    for where in leftovers:
        _drop_bind_unit(where, dest)
    return leftovers


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
    text = mount_unit_text(what, where)
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


def largest_gguf_bytes(target: UserTarget) -> int:
    biggest = 0
    for root in gguf_sources(target):
        shard_dirs: dict[Path, int] = {}
        for p in root.rglob("*.gguf"):
            if p.is_symlink() or not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            biggest = max(biggest, size)
            if "-of-" in p.name.lower():
                shard_dirs[p.parent] = shard_dirs.get(p.parent, 0) + size
        if shard_dirs:
            biggest = max(biggest, max(shard_dirs.values()))
    return biggest


def load_tuning(hw, largest_bytes: int) -> dict[str, object]:
    ram = max(int(getattr(hw, "ram_bytes", 0) or 0), 1)
    frac = largest_bytes / ram
    backends = hw.backends() if hasattr(hw, "backends") else set()
    # Strix Halo / gfx115x chat stays on Vulkan. See lemonade#3610, llama.cpp#28211.
    if hasattr(hw, "strix_halo_class") and hw.strix_halo_class():
        backend = "vulkan"
    elif "rocm" in backends:
        backend = "rocm"
    elif "vulkan" in backends:
        backend = "vulkan"
    else:
        backend = "auto"
    timeout = 600
    ctx = -1
    if frac >= 0.70:
        ctx = 2048
        timeout = 2400
    elif frac >= 0.50:
        ctx = 4096
        timeout = 1800
    elif frac >= 0.35:
        ctx = 8192
        timeout = 1200
    return {
        "ctx_size": ctx,
        "global_timeout": timeout,
        "max_loaded_models": 1,
        "llamacpp_backend": backend,
    }


def apply_tuning(settings: dict[str, object]) -> str:
    body = json.dumps(settings).encode("utf-8")
    req = urllib.request.Request(
        f"{LEMONADE_API}/internal/set",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read().decode("utf-8"))
        return "lemonade load settings updated"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    exe = shutil.which("lemonade-server") or shutil.which("lemonade")
    if not exe:
        raise RuntimeError("lemonade-server is not on PATH")
    parts = [f"{key}={value}" for key, value in settings.items()]
    # CLI wants dotted backend key.
    parts = [
        p.replace("llamacpp_backend=", "llamacpp.backend=") for p in parts
    ]
    p = _run([exe, "config", "set", *parts])
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "lemonade config set failed").strip())
    return "lemonade load settings updated"


def check_model_updates() -> str:
    exe = shutil.which("lemonade-server") or shutil.which("lemonade")
    if not exe:
        return ""
    p = _run([exe, "check-updates"])
    return ((p.stdout or "") + (p.stderr or "")).strip()


def publish(target: UserTarget) -> str:
    """Bind real GGUF trees and set extra_models_dir. Snap dest is never /home."""
    kind = detect()
    if not kind:
        return "lemonade not installed"
    sources = gguf_sources(target)
    if not sources:
        return "no real GGUF files to publish (Lemonade cannot follow store symlinks)"
    if kind == "snap":
        dest = extra_dir(kind)
        if dest == Path() or is_home_path(dest):
            raise RuntimeError(
                "snap extra_models_dir must be under /var/snap/lemonade-server/common"
            )
        mounts = bind_mounts(sources, dest)
        # Drop dest before writing dest/chat/srcN. A leftover dest bind would write chat/ into the user's tree.
        _drop_obsolete_owned_binds(dest, mounts)
        unit = ""
        for mount in mounts:
            unit = _write_bind_unit(mount.what, mount.where)
        _set_extra_models_dir(dest)
        _restart_snap()
        if len(mounts) == 1:
            return f"lemonade extra_models_dir={dest} via {unit}"
        return f"lemonade extra_models_dir={dest} ({len(mounts)} trees)"
    dest = sources[0]
    _set_extra_models_dir(dest)
    return f"lemonade extra_models_dir={dest}"
