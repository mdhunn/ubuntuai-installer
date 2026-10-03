"""Publish installer GGUF trees to Lemonade.

The snap cannot follow symlinks and cannot read /home as
extra_models_dir. Bind the real trees into SNAP_LEMONADE_MODELS and set
extra_models_dir to that snap-common path.

A mixed model folder is not one bind. A chat-only directory is mounted
at dest/chat/<folder name>. A chat GGUF in a mixed folder, or loose in
the model folder, is a file bind at dest/chat/<file stem>. Lemonade
labels a model from that directory name. Embedding GGUFs also bind at
dest/embeddings. A chat directory that contains an embeddings folder
gets an empty read-only tmpfs on a private staging bind under
/var/lib/ubuntuai. A source with no embeddings folder stays a direct
bind. A cover on the shared dest mount would propagate back onto the
user's embeddings folder.

Apply runs helper verb lemonade-publish after weights. That verb calls
publish(target_for(USER)). CLI --publish-lemonade is the same verb.
Weights owns source resolution and bind targets. Apply owns privilege
and order.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from domain import UserTarget
from probe import probe
from paths import (
    SNAP_LEMONADE_COMMON,
    SNAP_LEMONADE_MODELS,
    is_home_path,
    lemonade_extra_models_dir,
)
from weights import (
    UA,
    SHARD_RE,
    gguf_publish_role,
    human_bytes,
    is_foreign_mount,
    read_gguf_info,
)

SNAP_COMMON = SNAP_LEMONADE_COMMON
SNAP_EXTRA = SNAP_LEMONADE_MODELS
LEMONADE_API = "http://127.0.0.1:13305"
APPLY_PUBLISH_VERB = "lemonade-publish"
APPLY_TUNE_VERB = "lemonade-tune"
OWNED_UNIT_DESC = "Ubuntu AI models for Lemonade"
SYSTEM_UNIT_DIR = Path("/etc/systemd/system")
DAEMON_UNIT = "snap.lemonade-server.daemon"
# Private staging lives next to the tuning ledger. It is outside /var/snap
# and outside /home, so a cover there cannot join the user's peer group.
STAGE_DIR = DEFAULT_STAGE_DIR = Path("/var/lib/ubuntuai/stage")
UNIT_BACKUP_DIR = Path("/var/lib/ubuntuai/unit-backups")
UNIT_BACKUP_KEEP = 10

# Mark HITL. This PR warns only. Flip the string if a later change should refuse.
LOAD_RISK_POLICY = "warn_only"
WARN_FRAC = 0.35
STRONG_FRAC = 0.50
# 105G-class GGUF/shard trees. Big-RAM boxes still flag these.
HUGE_GGUF_BYTES = 80 * 1024**3
# Lemonade has no load-mode key. /internal/set uses llamacpp_args.
LLAMACPP_MMAP_ARGS = "--load-mode mmap"
# Shipped lemonade defaults.json. /internal/config merges these in, so a
# fresh install looks fully set until it is compared with /internal/config/defaults.
SHIPPED_DEFAULTS: dict[str, object] = {
    "ctx_size": -1,
    "global_timeout": 600,
    "max_loaded_models": 1,
    "llamacpp": {
        "backend": "auto",
        "args": "",
        "vulkan_args": "",
    },
}
CLI_TUNING_KEYS = {
    "llamacpp_backend": "llamacpp.backend",
    "llamacpp_args": "llamacpp.args",
    "llamacpp_vulkan_args": "llamacpp.vulkan_args",
}


@dataclass(frozen=True)
class BindMount:
    what: Path
    where: Path
    options: str = "bind,nofail"


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
    """Chat directories Lemonade can see. A mixed model folder is not one source."""
    trees: list[Path] = []
    seen: set[Path] = set()
    for root in _search_roots(target):
        for tree in _trees_in(root, target):
            if tree in seen or _too_wide(tree, target.home) or is_foreign_mount(tree):
                continue
            seen.add(tree)
            trees.append(tree)
    return _order_sources(_collapse(tuple(trees)), target)


def bind_mounts(sources: tuple[Path, ...], dest: Path) -> tuple[BindMount, ...]:
    """Map each chat directory onto dest/chat/<folder name>.

    The folder name is the Lemonade model id. Two directories with the
    same name get a stable numeric suffix. Nested sources bind once via
    the outer tree. One source does not land on dest itself.
    """
    sources = _collapse(sources)
    if not sources:
        return ()
    names = _assign_chat_names(
        [(source.name, str(_resolve(source))) for source in sources]
    )
    return tuple(
        BindMount(_resolve(source), dest / "chat" / names[str(_resolve(source))])
        for source in sources
    )


def embeddings_mount(target: UserTarget, dest: Path) -> BindMount | None:
    """Bind model_root/embeddings at dest/embeddings when that folder has a GGUF.

    Lemonade reads the embeddings label from the first directory under
    extra_models_dir. This bind is that directory. The chat srcN bind is a
    second copy and is covered separately.
    """
    source = _embeddings_source(target)
    if source is None:
        return None
    return BindMount(source, dest / "embeddings")


def embeddings_cover(
    model_root: Path, mounts: tuple[BindMount, ...]
) -> BindMount | None:
    """Empty tmpfs over embeddings on the private stage, not on the user's tree.

    The stage slot follows the covered source whose files are model_root.
    A symlink is skipped. tmpfs on that path would follow the link and hide
    the real folder for the whole host. The cover is not placed on the
    snap dest. That dest lives on shared /var, and a shared bind of a
    /home tree would copy the cover back onto the source.
    """
    if not _source_needs_cover(model_root):
        return None
    root = _resolve(model_root)
    slot = 0
    for mount in mounts:
        if _is_cover(mount) or _is_file_what(mount.what):
            continue
        where = _resolve(mount.where)
        if where == _stage_root() or _is_stage_src(where) or where.name == "embeddings":
            continue
        if not _source_needs_cover(mount.what):
            continue
        if _resolve(mount.what) == root:
            cover = _stage_src(f"src{slot}") / "embeddings"
            return BindMount(Path("tmpfs"), cover, _mount_options(Path("tmpfs"), cover))
        slot += 1
    return None


def is_owned_lemonade_where(dest: Path, where: Path) -> bool:
    """True for dest, dest/embeddings, a chat model dir, a chat file bind, or stage.

    dest itself is a legacy single-source bind. Cleanup still recognizes it.
    dest/chat/srcN remains owned so an older layout can be removed. A file
    bind ends in .gguf. Anything nested beside those paths is left alone.
    """
    if _is_owned_stage_where(where):
        return True
    dest = _resolve(dest)
    where = _resolve(where)
    if where == dest:
        return True
    try:
        rel = where.relative_to(dest)
    except ValueError:
        return False
    if rel.parts == ("embeddings",):
        return True
    if len(rel.parts) == 2 and rel.parts[0] == "chat" and rel.parts[1] not in {"", ".", ".."}:
        return True
    if (
        len(rel.parts) == 3
        and rel.parts[0] == "chat"
        and _is_srcn(rel.parts[1])
        and rel.parts[2] == "embeddings"
    ):
        return True
    return (
        len(rel.parts) == 3
        and rel.parts[0] == "chat"
        and rel.parts[2].lower().endswith(".gguf")
    )


def leftover_owned_binds(
    dest: Path,
    plan: tuple[BindMount, ...],
    *,
    unit_dir: Path,
    mounted: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """ubuntuai-owned dest / dest/chat/srcN binds that the collapsed plan no longer uses.

    An rbind onto dest/chat/srcN carries the stage cover along. That copy is
    not a mount of its own, and unmounting it strips the cover in every namespace.
    """
    planned = {_resolve(mount.where) for mount in plan}
    leftovers = []
    for where in owned_bind_wheres(dest, unit_dir, mounted):
        resolved = _resolve(where)
        if resolved in planned:
            continue
        if _is_carried_rbind_mount(resolved, plan):
            continue
        leftovers.append(where)
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
    # RequiresMountsFor= splits on spaces. Quotes keep ~/AI models as one path.
    # % starts a specifier inside the quoted value too.
    text = str(path).replace("\\", "\\\\").replace("%", "%%").replace('"', '\\"')
    return f'"{text}"'


def mount_unit_path(path: Path) -> str:
    # systemd does not unquote What= or Where=. Quotes would be part of the path.
    # % starts a specifier, so a literal percent is written %%.
    return str(path).replace("%", "%%")


def mount_unit_text(
    what: Path,
    where: Path,
    *,
    options: str = "bind,nofail",
    after: tuple[str, ...] = (),
    before: tuple[str, ...] = ("snap.lemonade-server.daemon.service",),
) -> str:
    # DefaultDependencies orders a /var mount Before=local-fs.target.
    # After=local-fs.target cycles with that. RequiresMountsFor waits for the source path.
    # nofail avoids a cycle on NFS or SMB and a long stall when a nofail disk is absent.
    after_lines = "".join(f"After={name}\n" for name in after)
    before_lines = "".join(f"Before={name}\n" for name in before)
    return (
        "[Unit]\n"
        f"Description={OWNED_UNIT_DESC}\n"
        f"RequiresMountsFor={quote_unit_path(what)}\n"
        f"{after_lines}"
        f"{before_lines}"
        "\n"
        "[Mount]\n"
        f"What={mount_unit_path(what)}\n"
        f"Where={mount_unit_path(where)}\n"
        "Type=none\n"
        f"Options={options}\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def cover_unit_text(where: Path, parent_unit: str) -> str:
    # Mounts after the model_root srcN bind. Hides embeddings that Lemonade would label chat.
    return (
        "[Unit]\n"
        f"Description={OWNED_UNIT_DESC}\n"
        f"RequiresMountsFor={quote_unit_path(where.parent)}\n"
        f"After={parent_unit}\n"
        "Before=snap.lemonade-server.daemon.service\n"
        "\n"
        "[Mount]\n"
        "What=tmpfs\n"
        f"Where={mount_unit_path(where)}\n"
        "Type=tmpfs\n"
        "Options=ro,nosuid,nodev,noexec,size=64k,mode=0555,nofail\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _search_roots(target: UserTarget) -> tuple[Path, ...]:
    """Extra paths, then model_root or its gguf store.

    A GGUF outside model_root/gguf still searches the whole model folder.
    Chat-only directories inside that folder become the sources. The gguf
    subdir is not added again, so it cannot become a second bind.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(raw: Path) -> None:
        try:
            path = raw.resolve()
        except OSError:
            return
        if not path.is_dir() or path in seen or is_foreign_mount(path):
            return
        seen.add(path)
        roots.append(path)

    for extra in target.extra_model_paths:
        add(extra)
    gguf = target.model_root / "gguf"
    if _has_gguf_outside_store(target.model_root):
        add(target.model_root)
    else:
        add(gguf)
        if not _any_gguf(gguf):
            add(target.model_root)
    return tuple(roots)


def _has_gguf_outside_store(model_root: Path) -> bool:
    """True when a GGUF directory entry lives in model_root but not under gguf/."""
    try:
        root = model_root.resolve()
    except OSError:
        return False
    if not root.is_dir():
        return False
    try:
        store = (model_root / "gguf").resolve()
    except OSError:
        store = root / "gguf"
    try:
        found = root.rglob("*.gguf")
    except OSError:
        return False
    for path in found:
        try:
            if not (path.is_file() or path.is_symlink()):
                continue
            # The directory entry decides. A symlink inside gguf/ stays in the store.
            located = path.parent.resolve() / path.name
        except OSError:
            continue
        try:
            located.relative_to(store)
        except ValueError:
            return True
    return False


def _embeddings_source(target: UserTarget) -> Path | None:
    try:
        folder = (target.model_root / "embeddings").resolve()
    except OSError:
        return None
    if not folder.is_dir() or _too_wide(folder, target.home) or is_foreign_mount(folder):
        return None
    if not _any_gguf(folder):
        return None
    return folder


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
    escaped: list[Path] = []
    for path in files:
        try:
            if path.is_symlink():
                real = path.resolve()
                if real.is_file():
                    escaped.append(real)
        except OSError:
            continue
    found = list(_chat_dirs(root))
    seen = set(found)
    for real in escaped:
        for picked in _chat_dirs(_lift(real, root, target)):
            if picked in seen:
                continue
            seen.add(picked)
            found.append(picked)
    return tuple(found)


def _chat_dirs(root: Path) -> tuple[Path, ...]:
    """Outermost directories whose visible GGUFs are all chat models."""
    if _is_chat_source(root):
        return (_resolve(root),)
    found: list[Path] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return ()
    children.sort(key=lambda path: path.name)
    for child in children:
        if child.name == "embeddings":
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                continue
        except OSError:
            continue
        found.extend(_chat_dirs(child))
    return tuple(found)


def _is_chat_source(root: Path) -> bool:
    """True when every GGUF Lemonade would see here is a text chat model.

    Files under the top-level embeddings directory are hidden by the cover,
    so they do not count and they do not block. mmproj and clip companions
    do not count and do not block. Later shards inherit shard 1. A diffusers
    tree is not a chat bind.
    """
    if _contains_diffusers(root):
        return False
    files = [path for path in _real_ggufs(root) if not _under_top_embeddings(path, root)]
    if not files:
        return False
    saw_chat = False
    for group in _groups_under(files):
        if group.role == "companion":
            continue
        if group.role != "chat":
            return False
        saw_chat = True
    return saw_chat


def _real_ggufs(root: Path) -> tuple[Path, ...]:
    try:
        files = list(root.rglob("*.gguf"))
    except OSError:
        return ()
    found: list[Path] = []
    for path in files:
        try:
            if path.is_symlink() or not path.is_file():
                continue
        except OSError:
            continue
        found.append(path)
    return tuple(found)


def _under_top_embeddings(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) >= 2 and rel.parts[0] == "embeddings"


def _contains_diffusers(root: Path) -> bool:
    try:
        found = root.rglob("model_index.json")
    except OSError:
        return False
    for path in found:
        try:
            if path.is_file():
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class _GgufGroup:
    directory: Path
    stem: str
    files: tuple[Path, ...]
    role: str
    primary: Path | None

    @property
    def key(self) -> str:
        path = self.primary or self.files[0]
        return str(_resolve(path))


def _shard_parts(path: Path) -> tuple[str, int | None]:
    match = SHARD_RE.search(path.name)
    if match is None:
        return path.stem, None
    return path.name[: match.start()], int(match.group(1))


def _split_no(path: Path) -> int | None:
    info = read_gguf_info(path)
    if info is None:
        return None
    return info.split_no


def _is_companion_file(path: Path) -> bool:
    if "mmproj" in path.name.lower():
        return True
    return gguf_publish_role(path) == "companion"


def _shard_group_role(members: list[Path]) -> tuple[str, Path | None]:
    """Role of shard 1. Later shards do not carry general.architecture."""
    numbered: list[tuple[int, str, Path]] = []
    for path in members:
        number = _shard_parts(path)[1]
        split = _split_no(path)
        if split == 1 or number == 1:
            numbered.append((0 if split == 1 else 1, str(_resolve(path)), path))
    if not numbered:
        for path in members:
            info = read_gguf_info(path)
            if info is None or not info.architecture:
                continue
            number = _shard_parts(path)[1]
            if number is None:
                number = _split_no(path) or 0
            numbered.append((number, str(_resolve(path)), path))
        if not numbered:
            return "missing-shard", None
    numbered.sort()
    primary = numbered[0][2]
    return gguf_publish_role(primary), primary


def _with_file(group: _GgufGroup, path: Path) -> _GgufGroup:
    files = tuple(sorted((*group.files, path), key=lambda item: item.name))
    return _GgufGroup(group.directory, group.stem, files, group.role, group.primary)


def _single_group(path: Path, role: str) -> _GgufGroup:
    return _GgufGroup(path.parent, path.stem, (path,), role, path)


def _best_companion_host(path: Path, groups: list[_GgufGroup]) -> _GgufGroup:
    if len(groups) == 1:
        return groups[0]
    name = path.name.lower()
    best = groups[0]
    best_key: tuple[int, str] | None = None
    for group in groups:
        stem = group.stem.lower()
        score = len(stem) if stem and stem in name else 0
        key = (-score, group.key)
        if best_key is None or key < best_key:
            best_key = key
            best = group
    return best


def _groups_in_directory(files: list[Path]) -> tuple[_GgufGroup, ...]:
    """One chat target per model. Shards and mmproj share that target."""
    companions: list[Path] = []
    buckets: dict[str, list[Path]] = {}
    deferred: list[Path] = []
    singles: list[Path] = []
    ordered = sorted(files, key=lambda path: str(_resolve(path)))
    for path in ordered:
        if _is_companion_file(path):
            companions.append(path)
            continue
        prefix, number = _shard_parts(path)
        split = _split_no(path)
        if number is not None:
            buckets.setdefault(prefix, []).append(path)
            continue
        if split is not None and split > 0:
            deferred.append(path)
            continue
        singles.append(path)
    if len(buckets) == 1 and deferred:
        only = next(iter(buckets))
        buckets[only].extend(deferred)
        deferred = []
    groups: list[_GgufGroup] = []
    for prefix, members in sorted(buckets.items()):
        role, primary = _shard_group_role(members)
        files_sorted = tuple(sorted(members, key=lambda item: item.name))
        groups.append(
            _GgufGroup(members[0].parent, prefix, files_sorted, role, primary)
        )
    for path in deferred:
        role, primary = _shard_group_role([path])
        groups.append(_GgufGroup(path.parent, path.stem, (path,), role, primary))
    for path in singles:
        groups.append(_single_group(path, gguf_publish_role(path)))
    if not companions:
        return tuple(groups)
    chat_groups = [group for group in groups if group.role == "chat"]
    if not chat_groups:
        for path in companions:
            groups.append(_single_group(path, "companion"))
        return tuple(groups)
    for path in companions:
        host = _best_companion_host(path, chat_groups)
        index = groups.index(host)
        groups[index] = _with_file(host, path)
        chat_groups = [group for group in groups if group.role == "chat"]
    return tuple(groups)


def _groups_under(files: list[Path]) -> tuple[_GgufGroup, ...]:
    by_dir: dict[Path, list[Path]] = {}
    for path in files:
        by_dir.setdefault(_resolve(path.parent), []).append(path)
    found: list[_GgufGroup] = []
    for members in by_dir.values():
        found.extend(_groups_in_directory(members))
    return tuple(found)


def _assign_chat_names(items: list[tuple[str, str]]) -> dict[str, str]:
    """Map a stable key to a Lemonade directory name.

    Sort by the key, not by discovery order. The first claimant keeps the
    bare name. Later claimants of that name get name-2, name-3, and so on.
    """
    used: set[str] = set()
    assigned: dict[str, str] = {}
    for desired, key in sorted(items, key=lambda item: item[1]):
        base = desired.strip()
        if not base or base in {".", ".."} or "/" in base or "\\" in base:
            base = "chat-model"
        if base not in used:
            chosen = base
        else:
            number = 2
            while f"{base}-{number}" in used:
                number += 1
            chosen = f"{base}-{number}"
        used.add(chosen)
        assigned[key] = chosen
    return assigned


def _chat_file_groups(
    target: UserTarget, sources: tuple[Path, ...]
) -> tuple[_GgufGroup, ...]:
    """Chat files that are not already inside a published chat directory."""
    try:
        root_embeddings = (target.model_root / "embeddings").resolve()
    except OSError:
        root_embeddings = target.model_root / "embeddings"
    loose: list[Path] = []
    for path in _candidate_ggufs(target):
        if _under_dir(path, root_embeddings):
            continue
        if _published_source(path, sources) is not None:
            continue
        parent = _resolve(path.parent)
        if _too_wide(parent, target.home) or is_foreign_mount(parent):
            continue
        loose.append(path)
    found = [group for group in _groups_under(loose) if group.role == "chat"]
    found.sort(key=lambda group: group.key)
    return tuple(found)


def _loose_groups(target: UserTarget, sources: tuple[Path, ...]) -> dict[Path, _GgufGroup]:
    try:
        root_embeddings = (target.model_root / "embeddings").resolve()
    except OSError:
        root_embeddings = target.model_root / "embeddings"
    loose: list[Path] = []
    for path in _candidate_ggufs(target):
        if _under_dir(path, root_embeddings):
            continue
        if _published_source(path, sources) is not None:
            continue
        loose.append(path)
    index: dict[Path, _GgufGroup] = {}
    for group in _groups_under(loose):
        for path in group.files:
            index[_resolve(path)] = group
    return index


def _has_gguf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _source_needs_cover(source: Path) -> bool:
    """True when source/embeddings is a real directory that holds a GGUF."""
    folder = source / "embeddings"
    try:
        if not folder.exists() or folder.is_symlink() or not folder.is_dir():
            return False
    except OSError:
        return False
    return _any_gguf(folder)


def _under_dir(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
    except ValueError:
        return False
    return path != folder


def _candidate_ggufs(target: UserTarget) -> tuple[Path, ...]:
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            if path.is_symlink():
                path = path.resolve()
            if not path.is_file():
                return
            key = path.resolve()
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        found.append(key)

    for root in _search_roots(target):
        try:
            files = root.rglob("*.gguf")
        except OSError:
            continue
        for path in files:
            add(path)
    found.sort(key=str)
    return tuple(found)


def _published_source(path: Path, sources: tuple[Path, ...]) -> Path | None:
    for source in sources:
        if _under_dir(path, source):
            return source
    return None


def _left_out_reason(path: Path, *, covered: bool, group_role: str | None = None) -> str:
    if covered:
        return (
            "It is in an embeddings folder that is hidden from chat. "
            "It is not the embeddings folder Lemonade is given."
        )
    if group_role == "missing-shard":
        return "It is a later shard and shard 1 is missing."
    role = group_role or gguf_publish_role(path)
    info = read_gguf_info(path)
    arch = info.architecture if info is not None and info.architecture else "unknown"
    if role == "tts":
        return (
            f"The architecture is {arch}. It is a speech model. "
            "Lemonade has no speech folder for extra models."
        )
    if role == "diffusion":
        return (
            f"The architecture is {arch}. It is an image model. "
            "Lemonade has no image folder for extra models."
        )
    if role == "rerank":
        return (
            "It is a reranker. "
            "Lemonade is not given a reranking folder from this header."
        )
    if role == "truncated":
        if not _has_gguf_magic(path):
            return "It is not a GGUF file. It is not a chat ggml model."
        return "The GGUF header is unreadable."
    if role == "companion":
        return "It is a vision companion with no chat model beside it."
    if role == "chat":
        return (
            "It shares a folder with a file that is not chat, "
            "so that folder is not published."
        )
    return f"The architecture is {arch}. It is not recognised as chat by this version."


def left_out_lines(target: UserTarget) -> tuple[str, ...]:
    """Plain-English lines for every GGUF that publish does not expose.

    Files inside a published chat tree stay quiet. A chat file kept as its
    own file bind stays quiet too. Files inside model_root/embeddings stay
    quiet because dest/embeddings still binds that folder. A covered
    embeddings tree on some other chat source is named.
    """
    sources = gguf_sources(target)
    published = {
        _resolve(path)
        for group in _chat_file_groups(target, sources)
        for path in group.files
    }
    reasons = _loose_groups(target, sources)
    try:
        root_embeddings = (target.model_root / "embeddings").resolve()
    except OSError:
        root_embeddings = target.model_root / "embeddings"
    lines: list[str] = []
    seen: set[Path] = set()
    for path in _candidate_ggufs(target):
        resolved = _resolve(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if _under_dir(path, root_embeddings):
            continue
        container = _published_source(path, sources)
        covered = container is not None and _under_top_embeddings(path, container)
        if container is not None and not covered:
            continue
        if resolved in published and not covered:
            continue
        group = reasons.get(resolved)
        described = path
        group_role = None if group is None else group.role
        if (
            group is not None
            and group.primary is not None
            and group_role not in {"missing-shard", "companion"}
        ):
            described = group.primary
        reason = _left_out_reason(described, covered=covered, group_role=group_role)
        lines.append(f"Left out {resolved}. {reason}")
    return tuple(lines)


def _emit_left_out(lines: tuple[str, ...]) -> None:
    # Helper stdout is the progress channel. The same lines stay in the result.
    for line in lines:
        print(line, flush=True)


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


def _order_sources(trees: tuple[Path, ...], target: UserTarget) -> tuple[Path, ...]:
    """model_root first so vulkan_bin stays on src0. Remaining trees sort by path."""
    if not trees:
        return ()
    primary = _primary_tree(trees, target)
    rest = [tree for tree in trees if tree != primary]
    rest.sort(key=str)
    if primary is None:
        return tuple(rest)
    return (primary, *rest)


def _primary_tree(trees: tuple[Path, ...], target: UserTarget) -> Path | None:
    model_root = _resolve(target.model_root)
    store = _resolve(target.model_root / "gguf")
    for tree in trees:
        if tree == model_root:
            return tree
    for tree in trees:
        if tree == store:
            return tree
    parents = [tree for tree in trees if _nested_under(model_root, tree)]
    if not parents:
        return None
    parents.sort(key=lambda path: len(path.parts))
    return parents[-1]


def _is_srcn(name: str) -> bool:
    return name.startswith("src") and name[3:].isdigit()


def _is_cover(mount: BindMount) -> bool:
    return mount.what == Path("tmpfs")


def _is_file_what(what: Path) -> bool:
    try:
        return what.is_file() and not what.is_symlink()
    except OSError:
        return False


def _stage_root() -> Path:
    return _resolve(STAGE_DIR)


def _stage_src(name: str) -> Path:
    return _stage_root() / name


def _is_stage_src(path: Path) -> bool:
    path = _resolve(path)
    return path.parent == _stage_root() and _is_srcn(path.name)


def _is_owned_stage_where(where: Path) -> bool:
    where = _resolve(where)
    if where == _stage_root() or _is_stage_src(where):
        return True
    return _is_stage_src(where.parent) and where.name == "embeddings"


def _mount_options(what: Path, where: Path) -> str:
    if what == Path("tmpfs"):
        return "ro,nosuid,nodev,noexec,size=64k,mode=0555,nofail"
    where_r = _resolve(where)
    what_r = _resolve(what)
    # A new bind of a shared source joins that peer group until it is made private.
    # rprivate on the stage directory alone is not enough.
    if where_r == _stage_root() or _is_stage_src(where_r):
        return "bind,rprivate,nofail"
    if _is_stage_src(what_r):
        return "rbind,nofail"
    return "bind,nofail"


def _counts_as_tree(mount: BindMount) -> bool:
    if _is_cover(mount):
        return False
    where = _resolve(mount.where)
    if where == _stage_root() or _is_stage_src(where):
        return False
    return True


def _unit_is_ours(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return f"Description={OWNED_UNIT_DESC}" in text


def _field_from_unit(text: str, field: str) -> str | None:
    prefix = f"{field}="
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        raw = line.split("=", 1)[1].strip()
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            raw = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        else:
            # New units write a literal percent as %%. Legacy quoted units do not.
            raw = raw.replace("%%", "%")
        return raw or None
    return None


def _where_from_unit(text: str) -> Path | None:
    raw = _field_from_unit(text, "Where")
    return Path(raw) if raw else None


def _what_from_unit(text: str) -> str | None:
    return _field_from_unit(text, "What")


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
    return tuple(row.mountpoint for row in _mount_rows())


def _umount_mountpoint(where: Path) -> None:
    child = where / "embeddings"
    removed_cover = False
    if child != where and _is_mountpoint(child):
        row = _row_at(child)
        removed_cover = row is not None and _is_cover_signature(row)
        _run(["umount", str(child)])
        if _is_mountpoint(child):
            raise RuntimeError(_still_mounted_message(child))
    _run(["umount", str(where)])
    if not removed_cover or not _is_mountpoint(where) or _is_mountpoint(child):
        return
    restored = _run(list(propagation_command(BindMount(Path("tmpfs"), child))))
    if restored.returncode != 0 or not _is_mountpoint(child):
        raise RuntimeError(
            _still_mounted_message(where) + " " + _cover_removed_message(child)
        )


def _drop_bind_unit(where: Path, dest: Path) -> None:
    if not is_owned_lemonade_where(dest, where):
        return
    path = _owned_unit_path(where)
    if path is not None:
        _run(["systemctl", "disable", "--now", path.name])
    else:
        try:
            unit = _escape_mount(where)
        except RuntimeError:
            unit = ""
        if unit:
            _run(["systemctl", "disable", "--now", unit])
    _umount_mountpoint(where)
    if _is_mountpoint(where):
        # Keep the unit. A later publish has to be able to stop this mount.
        raise RuntimeError(_still_mounted_message(where))
    if path is not None and path.exists():
        # Backup first. A failed copy must not delete the unit file.
        _backup_unit_file(path)
        path.unlink(missing_ok=True)
        _run(["systemctl", "daemon-reload"])


def _drop_obsolete_owned_binds(
    dest: Path, plan: tuple[BindMount, ...]
) -> tuple[Path, ...]:
    _release_carried_units(plan)
    leftovers = leftover_owned_binds(
        dest, plan, unit_dir=SYSTEM_UNIT_DIR, mounted=_mounted_wheres()
    )
    for where in leftovers:
        _drop_bind_unit(where, dest)
    _remove_empty_stage_dirs(plan)
    return leftovers


def _release_carried_units(plan: tuple[BindMount, ...]) -> None:
    """Disable a unit whose Where is only the cover an rbind already carries.

    disable --now would unmount that copy and strip the cover everywhere.
    """
    if not SYSTEM_UNIT_DIR.is_dir():
        return
    for path in sorted(SYSTEM_UNIT_DIR.glob("*.mount")):
        if not _unit_is_ours(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        where = _where_from_unit(text)
        if where is None or not _is_carried_rbind_mount(where, plan):
            continue
        _run(["systemctl", "disable", path.name])
        if not path.exists():
            continue
        _backup_unit_file(path)
        path.unlink(missing_ok=True)
        _run(["systemctl", "daemon-reload"])


def _missing_tool_message(tool: str) -> str:
    return (
        f"{tool} is not installed. "
        "Publish stopped so your model files stay where they are."
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        tool = cmd[0] if cmd else ""
        if tool in {"systemctl", "mount", "umount"}:
            raise RuntimeError(_missing_tool_message(tool)) from None
        raise


def _escape_mount(where: Path) -> str:
    """systemd-escape --path --suffix=mount, without calling the binary.

    Repeated slashes, a trailing slash, and "." segments collapse. A path
    with no leading slash is taken from the root. A leading dot on the first
    component is \\x2e. Later dots stay. Slash-replacement dashes stay
    literal. Every other byte outside ASCII letters, digits, and :_. is
    \\xHH of its UTF-8 encoding. ".." is rejected.
    """
    raw = str(where)
    if not raw.startswith("/"):
        raw = "/" + raw
    parts: list[str] = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise RuntimeError(f"systemd-escape failed for {where}")
        parts.append(part)
    if not parts:
        return "-.mount"
    chunks: list[str] = []
    for index, part in enumerate(parts):
        if index:
            chunks.append("-")
        body = part
        if index == 0 and body.startswith("."):
            chunks.append("\\x2e")
            body = body[1:]
        for ch in body:
            if ch.isascii() and (ch.isalnum() or ch in ":_."):
                chunks.append(ch)
                continue
            for byte in ch.encode("utf-8"):
                chunks.append(f"\\x{byte:02x}")
    return "".join(chunks) + ".mount"


def _is_mountpoint(where: Path) -> bool:
    where = _resolve(where)
    return any(_resolve(item) == where for item in _mounted_wheres())


def _bind_dest(where: Path) -> Path:
    where = _resolve(where)
    if (
        where.name == "embeddings"
        and _is_srcn(where.parent.name)
        and where.parent.parent.name == "chat"
    ):
        return where.parent.parent.parent
    if where.parent.name == "chat":
        return where.parent.parent
    if where.parent.parent.name == "chat" and where.name.lower().endswith(".gguf"):
        return where.parent.parent.parent
    if where.name == "embeddings":
        return where.parent
    return where


def _owned_ancestor_mount(where: Path, dest: Path) -> Path | None:
    where = _resolve(where)
    mounted = {_resolve(item) for item in _mounted_wheres()}
    for parent in where.parents:
        if parent in mounted and is_owned_lemonade_where(dest, parent):
            return parent
    return None


def _still_mounted_message(where: Path) -> str:
    return (
        f"The old Lemonade folder is still mounted at {where}. "
        "Publish stopped so your model files stay where they are."
    )


def _planned_what(mount: BindMount) -> str:
    if _is_cover(mount):
        return "tmpfs"
    return str(_resolve(mount.what))


def _unit_what(text: str) -> str | None:
    raw = _what_from_unit(text)
    if raw is None:
        return None
    if raw == "tmpfs":
        return raw
    return str(_resolve(Path(raw)))


def _refresh_changed_whats(plan: tuple[BindMount, ...]) -> None:
    """Remount an owned unit whose source moved but whose Where stayed put."""
    for mount in plan:
        path = _owned_unit_path(mount.where)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        current = _unit_what(text)
        if current is None or current == _planned_what(mount):
            continue
        _run(["systemctl", "stop", path.name])
        _umount_mountpoint(mount.where)
        if _is_mountpoint(mount.where):
            raise RuntimeError(_still_mounted_message(mount.where))


# /internal/config has to answer before extra_models_dir or tuning. Both need a live server.
# A cold Lemonade has answered after 15s. The wait has to outlast that or tuning is skipped.
CONFIG_READY_TIMEOUT = 90.0
CONFIG_READY_INTERVAL = 0.25


def _daemon_was_running() -> bool:
    return _run(["systemctl", "is-active", "--quiet", DAEMON_UNIT]).returncode == 0


def _stop_daemon() -> None:
    _run(["systemctl", "stop", DAEMON_UNIT])


def _start_daemon() -> None:
    _run(["systemctl", "start", DAEMON_UNIT])


def _lemonade_not_ready_message() -> str:
    return (
        "The folder mounts are in place. "
        "Lemonade did not answer, so the models folder was not set. "
        "Run publish again."
    )


def _config_answers() -> bool:
    req = urllib.request.Request(
        f"{LEMONADE_API}/internal/config",
        headers={"User-Agent": UA},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=1) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return isinstance(data, dict)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def _wait_for_config() -> None:
    # Helper stdout is the progress channel. Apply and the CLI stream it.
    print("Waiting for Lemonade to answer.", flush=True)
    deadline = time.monotonic() + CONFIG_READY_TIMEOUT
    while True:
        if _config_answers():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(_lemonade_not_ready_message())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(_lemonade_not_ready_message())
        time.sleep(min(CONFIG_READY_INTERVAL, remaining))


def _mkdir_nofollow(path: Path) -> None:
    """Create path and its parents. A symlink on the way is refused.

    pathlib mkdir follows links. Publish runs as root, so a link under the
    snap dest would create folders inside the user's tree.
    """
    if path.parent == path:
        return
    chain: list[Path] = []
    current = path
    while True:
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            chain.append(current)
            if current.parent == current:
                raise RuntimeError(
                    f"The folder {path} cannot be created. "
                    "Publish stopped so your files stay where they are."
                )
            current = current.parent
            continue
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise RuntimeError(
                f"A folder on the way to {path} is a symlink. "
                "Publish stopped so your files stay where they are."
            )
        break
    for directory in reversed(chain):
        parent = directory.parent
        try:
            parent_fd = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise RuntimeError(
                f"A folder on the way to {path} is a symlink. "
                "Publish stopped so your files stay where they are."
            ) from exc
        try:
            os.mkdir(directory.name, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            try:
                st = os.lstat(directory.name, dir_fd=parent_fd)
            except OSError as exc:
                raise RuntimeError(
                    f"A folder on the way to {path} is a symlink. "
                    "Publish stopped so your files stay where they are."
                ) from exc
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise RuntimeError(
                    f"A folder on the way to {path} is a symlink. "
                    "Publish stopped so your files stay where they are."
                )
        finally:
            os.close(parent_fd)


def _prepare_file_mountpoint(where: Path) -> None:
    """Create an empty file under the snap dest. The user's tree is not touched.

    mount(8) needs a regular file as the target of a file bind. A symlink
    is refused. Publish never creates a symlink.
    """
    _mkdir_nofollow(where.parent)
    parent = where.parent
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise RuntimeError(
            f"A folder on the way to {where} is a symlink. "
            "Publish stopped so your files stay where they are."
        ) from exc
    try:
        try:
            st = os.lstat(where.name, dir_fd=parent_fd)
        except FileNotFoundError:
            fd = os.open(
                where.name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o644,
                dir_fd=parent_fd,
            )
            os.close(fd)
            return
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise RuntimeError(
                f"The mount point {where} is not a regular file. "
                "Publish stopped so your files stay where they are."
            )
    finally:
        os.close(parent_fd)


_action_notes: list[str] = []
_before_units: dict[Path, tuple[str, ...]] = {}
_after_units: dict[Path, tuple[str, ...]] = {}


def _note(text: str) -> None:
    if text:
        _action_notes.append(text)


def _state_dir_unsafe() -> str:
    return (
        "The Ubuntu AI state folder is not safe to use. "
        "Publish stopped so your files stay unchanged."
    )


def _ensure_state_dir(directory: Path) -> None:
    fd = _ensure_ledger_dir(directory, refuse=_state_dir_unsafe)
    os.close(fd)


def _prepare_state_dirs() -> None:
    """Create the state parent, the stage, and the unit-backup folder.

    Each level is the same no-follow root-owned check as the tuning ledger.
    Callers that are about to unmount must do this first, and must stop if
    it fails.
    """
    parents: list[Path] = []
    for directory in (STAGE_DIR, UNIT_BACKUP_DIR):
        parent = directory.parent
        if parent not in parents:
            parents.append(parent)
    for directory in (*parents, STAGE_DIR, UNIT_BACKUP_DIR):
        _ensure_state_dir(directory)


def _prepare_stage_dir() -> None:
    _prepare_state_dirs()


def _set_unit_order(mounts: tuple[BindMount, ...]) -> dict[Path, str]:
    names = {_resolve(mount.where): _escape_mount(mount.where) for mount in mounts}
    _before_units.clear()
    _after_units.clear()
    cover_names = tuple(
        names[_resolve(mount.where)] for mount in mounts if _is_cover(mount)
    )
    for mount in mounts:
        key = _resolve(mount.where)
        if _is_cover(mount):
            parent = _resolve(mount.where.parent)
            if parent in names:
                _after_units[key] = (names[parent],)
            continue
        if mount.options.startswith("rbind"):
            rel: list[str] = []
            src = _resolve(mount.what)
            if src in names:
                rel.append(names[src])
            cover_where = _resolve(src / "embeddings")
            if cover_where in names:
                rel.append(names[cover_where])
            if rel:
                _after_units[key] = tuple(rel)
            continue
        if "rprivate" in mount.options and _resolve(mount.what) != _resolve(mount.where):
            parent = _resolve(mount.where.parent)
            if parent in names:
                _after_units[key] = (names[parent],)
            continue
        if mount.where.name == "embeddings" and cover_names:
            # The embeddings bind is never ordered after a cover.
            _before_units[key] = cover_names
    return names


def _render_unit(mount: BindMount) -> str:
    key = _resolve(mount.where)
    if _is_cover(mount):
        parent = _after_units.get(key, ("",))[0]
        return cover_unit_text(mount.where, parent)
    before = (*_before_units.get(key, ()), "snap.lemonade-server.daemon.service")
    return mount_unit_text(
        mount.what,
        mount.where,
        options=mount.options,
        after=_after_units.get(key, ()),
        before=before,
    )


def propagation_command(mount: BindMount) -> tuple[str, ...]:
    """mount(8) argv for one planned step. systemd Options= uses the same flags."""
    if _is_cover(mount):
        return (
            "mount",
            "-t",
            "tmpfs",
            "-o",
            "ro,nosuid,nodev,noexec,size=64k,mode=0555",
            "tmpfs",
            str(mount.where),
        )
    what = str(mount.what)
    where = str(mount.where)
    if mount.options.startswith("rbind"):
        return ("mount", "--rbind", what, where)
    if "rprivate" in mount.options:
        return ("mount", "-o", "bind,rprivate", what, where)
    return ("mount", "--bind", what, where)


def propagation_commands(
    mounts: tuple[BindMount, ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(propagation_command(mount) for mount in mounts)


def unmount_commands(mounts: tuple[BindMount, ...]) -> tuple[tuple[str, ...], ...]:
    """Reverse order. The rbind carries the cover, so that submount goes first."""
    cmds: list[tuple[str, ...]] = []
    for mount in reversed(mounts):
        if mount.options.startswith("rbind"):
            cmds.append(("umount", str(mount.where / "embeddings")))
        cmds.append(("umount", str(mount.where)))
    return tuple(cmds)


def apply_propagation(mounts: tuple[BindMount, ...]) -> None:
    """Run the mount sequence with mount(8). Publish uses the same commands as a fallback."""
    for mount in mounts:
        if not _is_cover(mount):
            if _resolve(mount.where) == _stage_root():
                _prepare_stage_dir()
            if _is_file_what(mount.what):
                _prepare_file_mountpoint(mount.where)
            else:
                _mkdir_nofollow(mount.where)
        elif not mount.where.is_dir():
            raise RuntimeError(
                f"The embeddings cover has no folder at {mount.where}. "
                "Publish stopped so your model files stay where they are."
            )
        mounted = _mount_with_command(mount.what, mount.where, mount.options)
        if mounted is None:
            continue
        if mounted.returncode != 0:
            detail = (mounted.stderr or mounted.stdout or "").strip()
            raise RuntimeError(
                f"Could not mount {mount.where}. "
                "Publish stopped so your model files stay where they are. "
                f"{detail}".rstrip()
            )


def _read_mountinfo_text() -> str:
    try:
        return Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return ""


@dataclass(frozen=True)
class _MountRow:
    mount_id: int
    parent_id: int
    dev: str
    root: str
    mountpoint: Path
    options: tuple[str, ...]
    optional: tuple[str, ...]
    fstype: str
    source: str
    super_options: tuple[str, ...]


def _unescape_mount(raw: str) -> str:
    return (
        raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _parse_mountinfo(text: str) -> tuple[_MountRow, ...]:
    found: list[_MountRow] = []
    for line in text.splitlines():
        parts = line.split()
        if "-" not in parts:
            continue
        sep = parts.index("-")
        if sep < 6 or len(parts) < sep + 2:
            continue
        try:
            mount_id = int(parts[0])
            parent_id = int(parts[1])
        except ValueError:
            continue
        super_raw = parts[sep + 3] if len(parts) > sep + 3 else ""
        found.append(
            _MountRow(
                mount_id=mount_id,
                parent_id=parent_id,
                dev=parts[2],
                root=_unescape_mount(parts[3]),
                mountpoint=Path(_unescape_mount(parts[4])),
                options=tuple(parts[5].split(",")),
                optional=tuple(parts[6:sep]),
                fstype=parts[sep + 1],
                source=_unescape_mount(parts[sep + 2] if len(parts) > sep + 2 else ""),
                super_options=tuple(super_raw.split(",")) if super_raw else (),
            )
        )
    return tuple(found)


def _mount_rows() -> tuple[_MountRow, ...]:
    return _parse_mountinfo(_read_mountinfo_text())


def _row_at(where: Path, rows: tuple[_MountRow, ...] | None = None) -> _MountRow | None:
    rows = _mount_rows() if rows is None else rows
    wanted = {_resolve(where)}
    found: _MountRow | None = None
    for row in rows:
        point = row.mountpoint
        if point in wanted or _resolve(point) in wanted:
            found = row
    return found


def _option_tokens(row: _MountRow) -> set[str]:
    return {token for token in (*row.options, *row.super_options) if token}


def _is_cover_signature(row: _MountRow) -> bool:
    """The empty cover an older publish mounted. mode=0555 is stored as mode=555."""
    if row.fstype != "tmpfs":
        return False
    tokens = _option_tokens(row)
    if "ro" not in tokens or "rw" in tokens:
        return False
    if "size=64k" not in tokens:
        return False
    return "mode=0555" in tokens or "mode=555" in tokens


def _peer_ids(row: _MountRow) -> set[str]:
    found: set[str] = set()
    for token in row.optional:
        if token.startswith("shared:") or token.startswith("master:"):
            found.add(token.split(":", 1)[1])
    return found


def _is_installer_cover_where(dest: Path, where: Path) -> bool:
    """True for dest/chat/srcN/embeddings or the stage cover stage/srcN/embeddings.

    dest/embeddings is the real embeddings bind. A peer there is not a cover.
    """
    where = _resolve(where)
    if where.name != "embeddings":
        return False
    if _is_stage_src(where.parent):
        return True
    dest = _resolve(dest)
    try:
        rel = where.relative_to(dest)
    except ValueError:
        return False
    return len(rel.parts) == 3 and rel.parts[0] == "chat" and _is_srcn(rel.parts[1])


def _cover_peer_is_ours(row: _MountRow, dest: Path, rows: tuple[_MountRow, ...]) -> bool:
    """True when this tmpfs shares a peer group with an installer cover.

    The peer has to sit on a cover location and carry the cover options.
    An exact cover signature with no such peer is the user's own tmpfs.
    """
    ids = _peer_ids(row)
    if not ids:
        return False
    for other in rows:
        if other.mount_id == row.mount_id or not (_peer_ids(other) & ids):
            continue
        if not _is_cover_signature(other):
            continue
        if _is_installer_cover_where(dest, other.mountpoint):
            return True
    return False


def _foreign_tmpfs_message(folder: Path) -> str:
    return (
        f"A temporary filesystem is mounted on {folder}. "
        "It is not the empty cover from an older publish. "
        "Publish stopped so that folder stays as it is."
    )


def _cover_removed_message(folder: Path) -> str:
    return (
        f"The embeddings cover at {folder} was removed. "
        "Run publish again to put it back."
    )


def _embeddings_tmpfs_row(folder: Path) -> _MountRow | None:
    try:
        if folder.is_symlink():
            return None
    except OSError:
        return None
    row = _row_at(folder)
    if row is None or row.fstype != "tmpfs":
        return None
    return row


def _clear_embeddings_leak(model_root: Path, dest: Path) -> str:
    """Drop the empty cover an older publish left on the user's embeddings folder.

    Any other tmpfs stays mounted. Removing it would destroy the files inside.
    """
    folder = model_root / "embeddings"
    row = _embeddings_tmpfs_row(folder)
    if row is None:
        return ""
    rows = _mount_rows()
    if not _is_cover_signature(row) or not _cover_peer_is_ours(row, dest, rows):
        raise RuntimeError(_foreign_tmpfs_message(folder))
    _run(["umount", str(folder)])
    if _embeddings_tmpfs_row(folder) is not None:
        raise RuntimeError(
            f"A temporary filesystem is still hiding {folder}. "
            "Publish stopped so your model files stay where they are."
        )
    return f"Removed a temporary filesystem that was hiding {folder}."


def _leak_plan_line(model_root: Path, dest: Path) -> str:
    folder = model_root / "embeddings"
    row = _embeddings_tmpfs_row(folder)
    if row is None:
        return ""
    if _is_cover_signature(row) and _cover_peer_is_ours(row, dest, _mount_rows()):
        return f"Remove the empty cover that is hiding {folder}."
    return _foreign_tmpfs_message(folder)


def _same_directory(left: Path, right: Path) -> bool | None:
    try:
        a = os.stat(left)
        b = os.stat(right)
    except OSError:
        return None
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _is_carried_rbind_mount(where: Path, plan: tuple[BindMount, ...]) -> bool:
    """True when where is the replica an rbind in plan already carries.

    The replica and the stage cover share a superblock. A cover mounted on
    dest/chat/srcN before that rbind has a different device and is still removed.
    """
    resolved = _resolve(where)
    rows = _mount_rows()
    here = _row_at(resolved, rows)
    if here is None:
        return False
    for mount in plan:
        if not str(mount.options).startswith("rbind"):
            continue
        target = _resolve(mount.where)
        if resolved == target or target not in resolved.parents:
            continue
        if _row_at(target, rows) is None:
            continue
        try:
            rel = resolved.relative_to(target)
        except ValueError:
            continue
        origin = _row_at(_resolve(mount.what) / rel, rows)
        if origin is not None and origin.dev == here.dev and origin.fstype == here.fstype:
            return True
    return False


def _mount_is_current(what: Path, where: Path, options: str) -> bool:
    if not _is_mountpoint(where):
        return False
    if what == Path("tmpfs"):
        row = _row_at(where)
        return row is not None and _is_cover_signature(row)
    if _same_directory(where, what) is not True:
        return False
    if not options.startswith("rbind"):
        return True
    origin = _resolve(what) / "embeddings"
    cover = _resolve(where) / "embeddings"
    if _is_mountpoint(origin):
        return _is_carried_rbind_mount(cover, (BindMount(what, where, options),))
    return not _is_mountpoint(cover)


def _mount_with_command(
    what: Path, where: Path, options: str
) -> subprocess.CompletedProcess | None:
    """Mount what onto where. None when that mount is already in place.

    A second mount(8) on a live mountpoint stacks. A wrong mount is removed
    first so the new one is the only one.
    """
    if _mount_is_current(what, where, options):
        return None
    if _is_mountpoint(where):
        child = where / "embeddings"
        if _is_mountpoint(child):
            planned = (BindMount(what, where, options),)
            # A carried replica shares the stage cover. Unmounting it strips
            # that cover in every namespace. A child left by an older rbind
            # still has to go before the parent can be replaced.
            carried = options.startswith("rbind") and _is_carried_rbind_mount(
                child, planned
            )
            if not carried:
                _run(["umount", str(child)])
                if _is_mountpoint(child):
                    raise RuntimeError(_still_mounted_message(child))
        _run(["umount", str(where)])
        if _is_mountpoint(where):
            raise RuntimeError(_still_mounted_message(where))
    return _run(list(propagation_command(BindMount(what, where, options))))


def _covering_mount(path: Path) -> _MountRow | None:
    resolved = _resolve(path)
    best: _MountRow | None = None
    best_len = -1
    for row in _mount_rows():
        point = _resolve(row.mountpoint)
        if resolved != point and point not in resolved.parents:
            continue
        length = len(point.parts)
        if length > best_len:
            best = row
            best_len = length
    return best


def _path_on_mount(row: _MountRow, path: Path) -> str:
    """mountinfo root of path, using the mount that contains it."""
    point = _resolve(row.mountpoint)
    resolved = _resolve(path)
    rel = "" if resolved == point else resolved.relative_to(point).as_posix()
    base = row.root.rstrip("/")
    if not rel:
        return base or "/"
    if base in ("", "/"):
        return "/" + rel
    return f"{base}/{rel}"


def _mountinfo_is_real_source(where: Path, what: Path) -> bool:
    """True when where's mount is the filesystem object at what.

    A tmpfs model folder and the installer cover both show up as tmpfs.
    The device id plus the root inside that filesystem tell them apart.
    """
    here = _row_at(where)
    there = _covering_mount(what)
    if here is None or there is None:
        return False
    if here.dev != there.dev or here.source != there.source:
        return False
    return here.root == _path_on_mount(there, what)


def _embeddings_bind_is_stale(mount: BindMount) -> bool:
    if _is_cover(mount) or mount.where.name != "embeddings":
        return False
    if _is_owned_stage_where(mount.where):
        return False
    if not _is_mountpoint(mount.where):
        return False
    row = _row_at(mount.where)
    # A bind of a tmpfs model folder keeps that folder's st_dev and st_ino.
    # The installer cover is a different tmpfs. When the directories cannot
    # be stated, the mountinfo device, source, and root are the same check.
    same = _same_directory(mount.where, mount.what)
    if same is True:
        return False
    if row is not None and row.fstype == "tmpfs":
        if same is None and _mountinfo_is_real_source(mount.where, mount.what):
            return False
        return True
    if same is not None:
        return not same
    if row is None:
        return False
    there = _row_at(mount.what)
    if there is None:
        return False
    return (row.dev, row.root, row.source) != (there.dev, there.root, there.source)


def _stop_mount_unit(where: Path) -> None:
    path = _owned_unit_path(where)
    if path is not None:
        _run(["systemctl", "stop", path.name])
        return
    try:
        unit = _escape_mount(where)
    except RuntimeError:
        return
    _run(["systemctl", "stop", unit])


def _remount_stale_embeddings(plan: tuple[BindMount, ...]) -> None:
    """Stop and unmount dest/embeddings when it is not the real embeddings folder.

    enable --now does not replace an active mount, and a unit whose What= is
    already the real path is not rewritten. The later enable mounts it again.
    """
    for mount in plan:
        if not _embeddings_bind_is_stale(mount):
            continue
        _stop_mount_unit(mount.where)
        _run(["umount", str(mount.where)])
        if _is_mountpoint(mount.where):
            raise RuntimeError(_still_mounted_message(mount.where))


def _backup_stamp() -> str:
    seconds = time.time()
    micros = int((seconds - int(seconds)) * 1_000_000)
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime(seconds)) + f"{micros:06d}Z"


def _backup_unit_file(path: Path) -> Path:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(_state_dir_unsafe()) from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise RuntimeError(
            f"The unit file {path} is a symlink. "
            "Publish stopped so that file stays unchanged."
        )
    try:
        src = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise RuntimeError(_state_dir_unsafe()) from exc
    try:
        data = _read_fd(src)
    finally:
        os.close(src)
    dir_fd = _ensure_ledger_dir(UNIT_BACKUP_DIR, refuse=_state_dir_unsafe)
    name = f"{path.name}.{_backup_stamp()}"
    try:
        out = os.open(
            name,
            os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY | os.O_CLOEXEC,
            0o644,
            dir_fd=dir_fd,
        )
        try:
            os.write(out, data)
            os.fsync(out)
        finally:
            os.close(out)
        _prune_unit_backups(dir_fd, path.name)
    except OSError as exc:
        raise RuntimeError(_state_dir_unsafe()) from exc
    finally:
        os.close(dir_fd)
    saved = UNIT_BACKUP_DIR / name
    _note(f"Saved the previous unit file at {saved}.")
    return saved


def _prune_unit_backups(dir_fd: int, unit_name: str) -> None:
    prefix = unit_name + "."
    names: list[str] = []
    for entry in os.scandir(dir_fd):
        try:
            if entry.name.startswith(prefix) and entry.is_file(follow_symlinks=False):
                names.append(entry.name)
        except OSError:
            continue
    names.sort()
    for old in names[:-UNIT_BACKUP_KEEP]:
        try:
            os.unlink(old, dir_fd=dir_fd)
        except OSError:
            continue


def _read_unit_nofollow(path: Path) -> str | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        return _read_fd(fd).decode("utf-8")
    except UnicodeError:
        return None
    finally:
        os.close(fd)


def _daemon_wants_dir() -> Path:
    return SYSTEM_UNIT_DIR / f"{DAEMON_UNIT}.service.wants"


def _wants_target(link: Path) -> Path | None:
    try:
        raw = os.readlink(link)
    except OSError:
        return None
    target = Path(raw)
    if not target.is_absolute():
        target = link.parent / target
    try:
        if not target.is_file() or target.is_symlink():
            return None
    except OSError:
        return None
    return target


def _clean_daemon_wants(plan: tuple[BindMount, ...]) -> None:
    wants = _daemon_wants_dir()
    if not wants.is_dir():
        return
    planned = {_resolve(mount.where) for mount in plan}
    for link in sorted(wants.iterdir(), key=lambda item: item.name):
        try:
            if not link.is_symlink():
                continue
        except OSError:
            continue
        target = _wants_target(link)
        if target is None:
            _note(
                f"Left a broken Lemonade service link {link.name} "
                "because the unit file is missing."
            )
            continue
        text = _read_unit_nofollow(target)
        if text is None or f"Description={OWNED_UNIT_DESC}" not in text:
            _note(
                f"Left a Lemonade service link this installer does not own. {link.name}"
            )
            continue
        where = _where_from_unit(text)
        if where is not None and _resolve(where) in planned:
            continue
        try:
            link.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Could not remove the old Lemonade mount link {link.name}. "
                "Publish stopped so that link stays in place."
            ) from exc
        _note(f"Removed an old Lemonade mount link {link.name}.")


def _unit_file(unit: str) -> Path:
    return SYSTEM_UNIT_DIR / unit


def _install_unit(path: Path, text: str, unit: str) -> subprocess.CompletedProcess | None:
    """None when systemd started the mount. Otherwise the failed enable result."""
    previous = None
    try:
        if path.is_symlink():
            raise RuntimeError(
                f"The unit file {path} is a symlink. "
                "Publish stopped so that file stays unchanged."
            )
        if path.is_file():
            previous = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(_state_dir_unsafe()) from exc
    if previous is not None and previous != text:
        _backup_unit_file(path)
    if previous != text:
        path.write_text(text, encoding="utf-8")
    _run(["systemctl", "daemon-reload"])
    enabled = _run(["systemctl", "enable", "--now", unit])
    if enabled.returncode != 0:
        return enabled
    return None


def _write_bind_unit(what: Path, where: Path) -> str:
    if _is_owned_stage_where(where):
        if _resolve(where) == _stage_root():
            _prepare_stage_dir()
    else:
        dest = _bind_dest(where)
        ancestor = _owned_ancestor_mount(where, dest)
        if ancestor is not None:
            raise RuntimeError(_still_mounted_message(ancestor))
    if _is_file_what(what):
        _prepare_file_mountpoint(where)
    else:
        _mkdir_nofollow(where)
    unit = _escape_mount(where)
    options = _mount_options(what, where)
    before = (*_before_units.get(_resolve(where), ()), "snap.lemonade-server.daemon.service")
    text = mount_unit_text(
        what,
        where,
        options=options,
        after=_after_units.get(_resolve(where), ()),
        before=before,
    )
    path = _unit_file(unit)
    enabled = _install_unit(path, text, unit)
    if enabled is not None:
        mounted = _mount_with_command(what, where, options)
        if mounted is not None and mounted.returncode != 0:
            raise RuntimeError(
                (enabled.stderr or enabled.stdout or "")
                + (mounted.stderr or mounted.stdout or "")
            )
    return unit


def _write_cover_unit(where: Path, parent_unit: str) -> str:
    # The staged bind already exposes this directory. Never mkdir it.
    try:
        if where.is_symlink():
            return ""
    except OSError:
        return ""
    if not where.is_dir():
        raise RuntimeError(
            f"The embeddings cover has no folder at {where}. "
            "Publish stopped so your model files stay where they are."
        )
    unit = _escape_mount(where)
    text = cover_unit_text(where, parent_unit)
    path = _unit_file(unit)
    enabled = _install_unit(path, text, unit)
    if enabled is not None:
        mounted = _mount_with_command(
            Path("tmpfs"), where, _mount_options(Path("tmpfs"), where)
        )
        if mounted is not None and mounted.returncode != 0:
            raise RuntimeError(
                (enabled.stderr or enabled.stdout or "")
                + (mounted.stderr or mounted.stdout or "")
            )
    return unit


def _directory_and_file_mounts(
    target: UserTarget, dest: Path, sources: tuple[Path, ...]
) -> tuple[tuple[BindMount, ...], tuple[BindMount, ...]]:
    sources = _collapse(sources)
    groups = _chat_file_groups(target, sources)
    names = _assign_chat_names(
        [(source.name, str(_resolve(source))) for source in sources]
        + [(group.stem, group.key) for group in groups]
    )
    directories = tuple(
        BindMount(_resolve(source), dest / "chat" / names[str(_resolve(source))])
        for source in sources
    )
    files: list[BindMount] = []
    for group in groups:
        folder = dest / "chat" / names[group.key]
        for path in group.files:
            files.append(BindMount(_resolve(path), folder / path.name))
    return directories, tuple(files)


def _snap_mount_plan(
    target: UserTarget, dest: Path, sources: tuple[Path, ...]
) -> tuple[BindMount, ...]:
    directories, files = _directory_and_file_mounts(target, dest, sources)
    mounts = list(directories)
    covers: dict[int, BindMount] = {}
    slot = 0
    for index, mount in enumerate(mounts):
        if not _source_needs_cover(mount.what):
            continue
        where = _stage_src(f"src{slot}") / "embeddings"
        slot += 1
        covers[index] = BindMount(Path("tmpfs"), where, _mount_options(Path("tmpfs"), where))
    emb = embeddings_mount(target, dest)
    if not covers:
        ordered = list(mounts)
        if emb is not None:
            ordered.append(emb)
        ordered.extend(files)
        return tuple(ordered)
    stage = _stage_root()
    ordered = [BindMount(stage, stage, _mount_options(stage, stage))]
    emb_done = False
    for index, mount in enumerate(mounts):
        cover = covers.get(index)
        if cover is None:
            ordered.append(mount)
            continue
        stage_src = cover.where.parent
        ordered.append(
            BindMount(mount.what, stage_src, _mount_options(mount.what, stage_src))
        )
        if emb is not None and not emb_done:
            ordered.append(emb)
            emb_done = True
        ordered.append(cover)
        ordered.append(
            BindMount(stage_src, mount.where, _mount_options(stage_src, mount.where))
        )
    if emb is not None and not emb_done:
        ordered.append(emb)
    ordered.extend(files)
    return tuple(ordered)


def _cover_path_is_ours(where: Path) -> bool:
    where = _resolve(where)
    if _is_owned_stage_where(where):
        return True
    return where.name == "embeddings" and where.parent.parent.name == "chat"


def _snapshot_cover_mounts() -> tuple[BindMount, ...]:
    """Cover mounts that are up now. systemd stop drops the carried copy."""
    found: list[BindMount] = []
    for row in _mount_rows():
        if not _is_cover_signature(row) or not _cover_path_is_ours(row.mountpoint):
            continue
        where = row.mountpoint
        found.append(BindMount(Path("tmpfs"), where, _mount_options(Path("tmpfs"), where)))
    return tuple(found)


def _restore_cover_mounts(covers: tuple[BindMount, ...]) -> None:
    for cover in covers:
        if _is_mountpoint(cover.where):
            continue
        try:
            if not cover.where.parent.is_dir():
                continue
        except OSError:
            continue
        _run(list(propagation_command(cover)))


def _remove_empty_stage_dirs(plan: tuple[BindMount, ...]) -> None:
    root = _stage_root()
    try:
        if not root.is_dir() or root.is_symlink():
            return
        children = list(root.iterdir())
    except OSError:
        return
    planned = {_resolve(mount.where) for mount in plan}
    for child in children:
        try:
            if child.is_symlink() or not child.is_dir() or not _is_srcn(child.name):
                continue
            if _resolve(child) in planned or _is_mountpoint(child):
                continue
            emb = child / "embeddings"
            if not emb.is_symlink() and emb.is_dir() and not _is_mountpoint(emb):
                try:
                    emb.rmdir()
                except OSError:
                    pass
            child.rmdir()
        except OSError:
            continue


def _apply_snap_mounts(
    dest: Path,
    mounts: tuple[BindMount, ...],
    *,
    on_ready: Callable[[], None] | None = None,
) -> str:
    was_running = _daemon_was_running()
    covers = _snapshot_cover_mounts() if was_running else ()
    label = ""
    started = False
    try:
        if was_running:
            _stop_daemon()
        if any(_is_cover(mount) or mount.options != "bind,nofail" for mount in mounts):
            _set_unit_order(mounts)
        _remount_stale_embeddings(mounts)
        _refresh_changed_whats(mounts)
        _drop_obsolete_owned_binds(dest, mounts)
        written: dict[Path, str] = {}
        for mount in mounts:
            if _is_cover(mount):
                parent = _resolve(mount.where.parent)
                parent_unit = written.get(parent) or _escape_mount(mount.where.parent)
                _write_cover_unit(mount.where, parent_unit)
            else:
                label = _write_bind_unit(mount.what, mount.where)
                written[_resolve(mount.where)] = label
        _clean_daemon_wants(mounts)
        # Binds are in place. The server has to be up before config writes.
        _start_daemon()
        started = True
        _wait_for_config()
        _set_extra_models_dir(dest)
        if on_ready is not None:
            on_ready()
    finally:
        if started and not was_running:
            _stop_daemon()
        elif was_running and not started:
            _restore_cover_mounts(covers)
            _start_daemon()
    return label


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


def gguf_tree_bytes(root: Path) -> int:
    """Largest real GGUF in root. Numbered shards in one folder count as one file."""
    try:
        if not root.is_dir():
            return 0
        files = root.rglob("*.gguf")
    except OSError:
        return 0
    biggest = 0
    shard_dirs: dict[Path, int] = {}
    for p in files:
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


def model_gguf_bytes(path: Path) -> int:
    """Bytes Lemonade would see for one selected file or shard folder."""
    try:
        if path.is_symlink() or path.is_file():
            real = path.resolve()
            if real.is_dir():
                return gguf_tree_bytes(real)
            if not real.is_file():
                return 0
            if "-of-" in real.name.lower():
                return gguf_tree_bytes(real.parent)
            return real.stat().st_size
        if path.is_dir():
            return gguf_tree_bytes(path)
    except OSError:
        return 0
    return 0


def _visible_tree_count(mounts: tuple[BindMount, ...]) -> int:
    seen: set[Path] = set()
    count = 0
    for mount in mounts:
        if _is_cover(mount):
            continue
        where = _resolve(mount.where)
        if where == _stage_root() or _is_stage_src(where):
            continue
        key = where.parent if _is_file_what(mount.what) else where
        if key in seen:
            continue
        seen.add(key)
        count += 1
    return count


def largest_gguf_bytes(target: UserTarget) -> int:
    biggest = 0
    sources = gguf_sources(target)
    for root in sources:
        biggest = max(biggest, gguf_tree_bytes(root))
    for group in _chat_file_groups(target, sources):
        total = 0
        best = 0
        sharded = False
        for path in group.files:
            if "mmproj" in path.name.lower():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            best = max(best, size)
            if _shard_parts(path)[1] is not None:
                sharded = True
                total += size
        biggest = max(biggest, total if sharded else best)
    return biggest


def _is_large(frac: float, largest_bytes: int) -> bool:
    return frac >= WARN_FRAC or int(largest_bytes) >= HUGE_GGUF_BYTES


def _igpu_vulkan_path(hw) -> bool:
    backends = hw.backends() if hasattr(hw, "backends") else set()
    if "vulkan" in backends:
        return True
    if hasattr(hw, "strix_halo_class") and hw.strix_halo_class():
        return True
    for device in getattr(hw, "devices", ()):
        if getattr(device, "kind", "") != "igpu":
            continue
        if getattr(device, "backend", "") == "vulkan":
            return True
        if getattr(device, "vendor", "") in {"amd", "intel"}:
            return True
    return False


def load_risk(
    frac: float, bytes: int, backend: str | None = None
) -> dict[str, object]:
    """Detect huge loads. Policy is warn-only. Callers must not refuse."""
    level = "ok"
    if frac >= STRONG_FRAC or int(bytes) >= HUGE_GGUF_BYTES:
        level = "strong"
    elif frac >= WARN_FRAC:
        level = "warn"
    return {
        "level": level,
        "policy": LOAD_RISK_POLICY,
        "action": "warn",
        "frac": frac,
        "bytes": int(bytes),
        "backend": backend or "",
    }


def _models_phrase(count: int) -> str:
    if count == 1:
        return "one model"
    if count < 0:
        return "as many models as memory allows"
    return f"{count} models"


def risk_english(
    risk: dict[str, object], ram_bytes: int = 0, max_loaded_models: int = 1
) -> str:
    level = str(risk.get("level") or "ok")
    if level == "ok":
        return ""
    size = human_bytes(int(risk.get("bytes") or 0))
    ram = human_bytes(ram_bytes) if ram_bytes else "this machine's RAM"
    try:
        count = int(max_loaded_models)
    except (TypeError, ValueError):
        count = 1
    kept = _models_phrase(count)
    if level == "strong":
        return (
            f"Strong warning. Largest GGUF is {size} on {ram}. "
            "Vulkan can lose the GPU when a file this large is loaded. "
            f"Lemonade will keep {kept}, shrink context, and memory-map the file. "
            "Publish and updates continue."
        )
    return (
        f"Warning. Largest GGUF is {size} on {ram}. "
        "That is a large share of RAM. "
        f"Lemonade will keep {kept} and shrink context."
    )


# Values load_tuning can write, plus any earlier release. With no ledger, a
# non-factory value is ours only when it appears here. max_loaded_models is
# listed for the drift guard. Any non-factory count stays the user's.
INSTALLER_WRITTEN_VALUES: dict[str, frozenset[object]] = {
    "ctx_size": frozenset({-1, 2048, 4096, 8192}),
    "global_timeout": frozenset({600, 1200, 1800, 2400}),
    "llamacpp_backend": frozenset({"auto", "vulkan", "rocm"}),
    "llamacpp_args": frozenset({"", LLAMACPP_MMAP_ARGS}),
    "llamacpp_vulkan_args": frozenset({"", LLAMACPP_MMAP_ARGS}),
    "max_loaded_models": frozenset({1}),
}


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
    # mmap is the primary large+vulkan/iGPU mitigation. llama.cpp#27360.
    mmap = _is_large(frac, largest_bytes) and _igpu_vulkan_path(hw)
    settings: dict[str, object] = {
        "ctx_size": ctx,
        "global_timeout": timeout,
        "max_loaded_models": 1,
        "llamacpp_backend": backend,
    }
    if mmap:
        settings["llamacpp_args"] = LLAMACPP_MMAP_ARGS
    return settings


def cli_tuning_parts(settings: dict[str, object]) -> list[str]:
    parts = []
    for key, value in settings.items():
        parts.append(f"{CLI_TUNING_KEYS.get(key, key)}={value}")
    args = str(settings.get("llamacpp_args") or "")
    if LLAMACPP_MMAP_ARGS in args and not any(
        item.startswith("llamacpp.vulkan_args=") for item in parts
    ):
        parts.append(f"llamacpp.vulkan_args={args}")
    return parts


def _flatten_config(data: dict) -> dict[str, object]:
    flat: dict[str, object] = dict(data)
    nested = data.get("llamacpp")
    if isinstance(nested, dict):
        if "backend" in nested and "llamacpp_backend" not in flat:
            flat["llamacpp_backend"] = nested["backend"]
        if "args" in nested and "llamacpp_args" not in flat:
            flat["llamacpp_args"] = nested["args"]
        if "vulkan_args" in nested:
            flat["llamacpp_vulkan_args"] = nested["vulkan_args"]
    return flat


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mmap_in_config(actual: dict) -> bool | None:
    flat = _flatten_config(actual)
    seen = False
    for key in ("llamacpp_args", "llamacpp_vulkan_args"):
        if key not in flat:
            continue
        seen = True
        if LLAMACPP_MMAP_ARGS in str(flat.get(key) or ""):
            return True
    if not seen:
        return None
    return False


def read_config() -> dict:
    req = urllib.request.Request(
        f"{LEMONADE_API}/internal/config",
        headers={"User-Agent": UA},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict):
            return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    exe = shutil.which("lemonade-server") or shutil.which("lemonade")
    if not exe:
        return {}
    p = _run([exe, "config"])
    text = (p.stdout or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    out: dict[str, object] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _is_set(flat: dict, key: str) -> bool:
    if key not in flat:
        return False
    value = flat.get(key)
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _values_equal(left: object, right: object) -> bool:
    left_int = _as_int(left)
    right_int = _as_int(right)
    if left_int is not None and right_int is not None:
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        return left_int == right_int
    return str(left if left is not None else "").strip() == str(
        right if right is not None else ""
    ).strip()


def _user_overrode(current: dict, factory: dict, key: str) -> bool:
    if not _is_set(current, key):
        return False
    if key not in factory:
        return True
    return not _values_equal(current.get(key), factory.get(key))


def _config_known(current: object) -> bool:
    return isinstance(current, dict) and bool(current)


# Keys the installer may write. The ledger holds these values and nothing else.
TUNING_LEDGER_KEYS = frozenset(
    {
        "max_loaded_models",
        "ctx_size",
        "global_timeout",
        "llamacpp_backend",
        "llamacpp_args",
        "llamacpp_vulkan_args",
    }
)


# Lemonade config is system-wide. The ledger lives with it, not in a home directory.
DEFAULT_TUNING_LEDGER_DIR = Path("/var/lib/ubuntuai")
TUNING_LEDGER_DIR = DEFAULT_TUNING_LEDGER_DIR
TUNING_LEDGER_NAME = "lemonade-tuning.json"
_LEDGER_WRITE_FLAGS = (
    os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY | os.O_CLOEXEC
)


def tuning_ledger_path(directory: Path | None = None) -> Path:
    return (directory if directory is not None else TUNING_LEDGER_DIR) / TUNING_LEDGER_NAME


def _ledger_unreadable_message() -> str:
    return (
        "The saved Lemonade load settings could not be read. "
        "Models are still published."
    )


def _ledger_write_failed_message() -> str:
    return (
        "The Lemonade settings record could not be saved. "
        "Models are still published."
    )


def _ledger_dir_unsafe() -> str:
    return (
        "The Lemonade settings folder is not safe to use. "
        "Publish stopped so your files stay unchanged."
    )


def _ledger_symlink_message() -> str:
    return (
        "The Lemonade settings record is a symlink. "
        "Publish stopped so that file stays unchanged."
    )


def _corrupt_ledger_warning() -> str:
    return (
        "The saved Lemonade load settings could not be read. "
        "Your model count is kept when it is not the Lemonade default. "
        "Other load settings are set again only when they match a value "
        "this installer writes."
    )


def _ledger_dir_is_trusted(st: os.stat_result) -> bool:
    return st.st_uid == 0 and st.st_gid == 0


def _ensure_ledger_dir(
    directory: Path, *, refuse: Callable[[], str] | None = None
) -> int:
    """Return a dir fd for a real root-owned directory. Caller closes it."""
    message = refuse or _ledger_dir_unsafe
    parent = directory.parent
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise RuntimeError(message()) from exc
    try:
        try:
            os.mkdir(directory.name, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
    finally:
        os.close(parent_fd)
    try:
        dir_fd = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise RuntimeError(message()) from exc
    try:
        st = os.fstat(dir_fd)
        if not stat.S_ISDIR(st.st_mode) or not _ledger_dir_is_trusted(st):
            raise RuntimeError(message())
        os.fchmod(dir_fd, 0o755)
    except Exception:
        os.close(dir_fd)
        raise
    return dir_fd


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 65536)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def _load_ledger(directory: Path | None = None) -> tuple[dict[str, object] | None, str]:
    """Return the ledger and a warning. None means there is no usable ledger."""
    path = tuning_ledger_path(directory)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, ""
    except OSError as exc:
        raise RuntimeError(_ledger_dir_unsafe()) from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise RuntimeError(_ledger_symlink_message())
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise RuntimeError(_ledger_dir_unsafe()) from exc
    try:
        raw = _read_fd(fd)
    finally:
        os.close(fd)
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None, _corrupt_ledger_warning()
    if not isinstance(data, dict):
        return None, _corrupt_ledger_warning()
    return (
        {
            str(key): value
            for key, value in data.items()
            if str(key) in TUNING_LEDGER_KEYS
        },
        "",
    )


def read_tuning_ledger(directory: Path | None = None) -> dict[str, object] | None:
    values, _warning = _load_ledger(directory)
    return values


def _ledger_payload(values: dict[str, object]) -> bytes:
    kept = {
        str(key): value
        for key, value in values.items()
        if str(key) in TUNING_LEDGER_KEYS
    }
    return (json.dumps(kept, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_tuning_ledger(
    values: dict[str, object],
    directory: Path | None = None,
) -> None:
    directory = directory if directory is not None else TUNING_LEDGER_DIR
    payload = _ledger_payload(values)
    dir_fd = _ensure_ledger_dir(directory)
    tmp_name = f".{TUNING_LEDGER_NAME}.{secrets.token_hex(8)}.tmp"
    fd = -1
    renamed = False
    try:
        try:
            final_st = os.lstat(TUNING_LEDGER_NAME, dir_fd=dir_fd)
        except FileNotFoundError:
            final_st = None
        if final_st is not None and stat.S_ISLNK(final_st.st_mode):
            raise RuntimeError(_ledger_symlink_message())
        try:
            fd = os.open(tmp_name, _LEDGER_WRITE_FLAGS, 0o644, dir_fd=dir_fd)
        except OSError as exc:
            raise RuntimeError(_ledger_dir_unsafe()) from exc
        os.fchmod(fd, 0o644)
        view = memoryview(payload)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                raise OSError("The Lemonade settings record could not be saved.")
            view = view[wrote:]
        os.fsync(fd)
        try:
            final_st = os.lstat(TUNING_LEDGER_NAME, dir_fd=dir_fd)
        except FileNotFoundError:
            final_st = None
        if final_st is not None and stat.S_ISLNK(final_st.st_mode):
            raise RuntimeError(_ledger_symlink_message())
        os.rename(
            tmp_name,
            TUNING_LEDGER_NAME,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
        )
        renamed = True
        os.fsync(dir_fd)
    finally:
        opened = fd >= 0
        if opened:
            os.close(fd)
        if opened and not renamed and dir_fd >= 0:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except OSError:
                pass
        if dir_fd >= 0:
            os.close(dir_fd)


def _remember_tuning(
    chosen: dict[str, object],
    *,
    directory: Path | None = None,
    cli: bool = False,
) -> None:
    prior, _warning = _load_ledger(directory)
    merged = dict(prior or {})
    merged.update(
        {key: value for key, value in chosen.items() if key in TUNING_LEDGER_KEYS}
    )
    if cli:
        names = {
            "llamacpp.backend": "llamacpp_backend",
            "llamacpp.args": "llamacpp_args",
            "llamacpp.vulkan_args": "llamacpp_vulkan_args",
        }
        numeric = {"ctx_size", "global_timeout", "max_loaded_models"}
        for part in cli_tuning_parts(chosen):
            raw_key, sep, raw_value = part.partition("=")
            if not sep:
                continue
            key = names.get(raw_key, raw_key)
            if key not in TUNING_LEDGER_KEYS:
                continue
            parsed = _as_int(raw_value)
            merged[key] = parsed if parsed is not None and key in numeric else raw_value
    write_tuning_ledger(merged, directory)


def _installer_wrote_value(key: str, value: object) -> bool:
    allowed = INSTALLER_WRITTEN_VALUES.get(key)
    if not allowed:
        return False
    return any(_values_equal(value, item) for item in allowed)


def _user_owns_key(
    current: dict,
    factory: dict,
    ledger: dict[str, object] | None,
    key: str,
) -> bool:
    """True when the live value belongs to the user.

    With a ledger, the value differs from the factory default and from
    the last value this installer wrote.

    With no ledger, a non-factory max_loaded_models belongs to the user.
    Any other non-factory value belongs to the user when this installer
    cannot write that value.
    """
    if not _is_set(current, key):
        return False
    differs = _user_overrode(current, factory, key)
    if ledger is None:
        if not differs:
            return False
        if key == "max_loaded_models":
            return True
        return not _installer_wrote_value(key, current.get(key))
    if not differs:
        return False
    if key not in ledger:
        return True
    return not _values_equal(current.get(key), ledger.get(key))


def _load_mode_user_owns(
    current: dict, factory: dict, ledger: dict[str, object] | None
) -> bool:
    return _user_owns_key(current, factory, ledger, "llamacpp_args") or _user_owns_key(
        current, factory, ledger, "llamacpp_vulkan_args"
    )


def read_factory_defaults() -> dict:
    req = urllib.request.Request(
        f"{LEMONADE_API}/internal/config/defaults",
        headers={"User-Agent": UA},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict) and data:
            return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    return json.loads(json.dumps(SHIPPED_DEFAULTS))


def _tuning_plan(
    defaults: dict[str, object],
    current: dict,
    factory: dict | None = None,
    ledger: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Split installer-owned keys into values to post and values that already match.

    A matching value is not posted. It is still recorded when another key
    is written, so the next ledger run does not treat it as user-set.
    """
    if not _config_known(current):
        return {}, {}
    if factory is None:
        factory = read_factory_defaults()
    flat = _flatten_config(current)
    flat_factory = _flatten_config(factory) if factory else {}
    post: dict[str, object] = {}
    matched: dict[str, object] = {}
    for key, value in defaults.items():
        if key in {"llamacpp_args", "llamacpp_vulkan_args"}:
            user_owns = _load_mode_user_owns(flat, flat_factory, ledger)
        else:
            user_owns = _user_owns_key(flat, flat_factory, ledger, key)
        if user_owns:
            continue
        if _is_set(flat, key) and _values_equal(flat.get(key), value):
            matched[key] = value
            continue
        post[key] = value
    return post, matched


def tuning_to_apply(
    defaults: dict[str, object],
    current: dict,
    factory: dict | None = None,
    ledger: dict[str, object] | None = None,
) -> dict[str, object]:
    """Keys to write. Skip user-owned keys and keys that already match."""
    post, _matched = _tuning_plan(defaults, current, factory, ledger)
    return post


def effective_max_loaded(
    settings: dict[str, object],
    current: dict,
    factory: dict,
    ledger: dict[str, object] | None = None,
) -> int:
    if not _config_known(current):
        value = _as_int(settings.get("max_loaded_models"))
        return 1 if value is None else value
    flat = _flatten_config(current)
    flat_factory = _flatten_config(factory) if factory else {}
    if _user_owns_key(flat, flat_factory, ledger, "max_loaded_models"):
        value = _as_int(flat.get("max_loaded_models"))
        if value is not None:
            return value
    value = _as_int(settings.get("max_loaded_models"))
    return 1 if value is None else value


def verify_tuning(
    expected: dict[str, object], actual: dict
) -> tuple[bool, str]:
    if not expected:
        return True, ""
    if not actual:
        return False, (
            "Lemonade did not return the load settings the installer wrote. "
            "Models are still published."
        )
    flat = _flatten_config(actual)
    misses: list[str] = []
    if "max_loaded_models" in expected and _as_int(
        flat.get("max_loaded_models")
    ) != _as_int(expected.get("max_loaded_models")):
        misses.append("max_loaded_models")
    want_ctx = _as_int(expected.get("ctx_size"))
    if want_ctx is not None and _as_int(flat.get("ctx_size")) != want_ctx:
        misses.append("ctx_size")
    want_timeout = _as_int(expected.get("global_timeout"))
    if (
        want_timeout is not None
        and _as_int(flat.get("global_timeout")) != want_timeout
    ):
        misses.append("global_timeout")
    want_backend = expected.get("llamacpp_backend")
    if want_backend is not None and flat.get("llamacpp_backend") != want_backend:
        misses.append("llamacpp_backend")
    if expected.get("llamacpp_args") and _mmap_in_config(actual) is False:
        misses.append("llamacpp_args")
    if misses:
        return False, (
            "Lemonade did not keep the load settings the installer wrote "
            f"({', '.join(misses)}). Models are still published."
        )
    return True, ""


def _with_ledger_warning(warning: str, text: str) -> str:
    if warning and text:
        return f"{warning}\n{text}"
    return warning or text


def apply_tuning(
    settings: dict[str, object],
    current: dict | None = None,
    factory: dict | None = None,
    ledger_dir: Path | None = None,
) -> str:
    # Compare with factory defaults and the ledger. /internal/config merges both.
    if current is None:
        current = read_config()
    if not _config_known(current):
        return (
            "Lemonade did not report its load settings. "
            "The installer left them unchanged."
        )
    if factory is None:
        factory = read_factory_defaults()
    ledger, ledger_warning = _load_ledger(ledger_dir)
    chosen, matched = _tuning_plan(settings, current, factory, ledger)
    if not chosen:
        return _with_ledger_warning(ledger_warning, "lemonade load settings kept")
    body = json.dumps(chosen).encode("utf-8")
    req = urllib.request.Request(
        f"{LEMONADE_API}/internal/set",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    wrote = False
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            json.loads(resp.read().decode("utf-8"))
        wrote = True
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    cli = False
    if not wrote:
        exe = shutil.which("lemonade-server") or shutil.which("lemonade")
        if not exe:
            raise RuntimeError("lemonade-server is not on PATH")
        p = _run([exe, "config", "set", *cli_tuning_parts(chosen)])
        if p.returncode != 0:
            raise RuntimeError(
                (p.stderr or p.stdout or "lemonade config set failed").strip()
            )
        cli = True
    try:
        _remember_tuning({**matched, **chosen}, directory=ledger_dir, cli=cli)
    except (OSError, RuntimeError):
        return _with_ledger_warning(ledger_warning, _ledger_write_failed_message())
    ok, miss = verify_tuning(chosen, read_config())
    if not ok:
        return _with_ledger_warning(ledger_warning, miss)
    return _with_ledger_warning(ledger_warning, "lemonade load settings updated")


def report_load_tuning(
    target: UserTarget, hw=None, ledger_dir: Path | None = None
) -> str:
    hw = hw or probe()
    largest = largest_gguf_bytes(target)
    settings = load_tuning(hw, largest)
    ram = int(getattr(hw, "ram_bytes", 0) or 0)
    frac = largest / max(ram, 1)
    risk = load_risk(frac, largest, str(settings.get("llamacpp_backend") or ""))
    lines: list[str] = []
    current = read_config()
    factory = read_factory_defaults()
    ledger = None
    try:
        ledger, _ledger_warning = _load_ledger(ledger_dir)
    except RuntimeError:
        lines.append(_ledger_unreadable_message())
    else:
        try:
            lines.append(apply_tuning(settings, current, factory, ledger_dir))
        except RuntimeError:
            lines.append(
                "Lemonade did not accept the load settings. "
                "Models are still published."
            )
    warn = risk_english(
        risk,
        ram_bytes=ram,
        max_loaded_models=effective_max_loaded(settings, current, factory, ledger),
    )
    if warn:
        lines.append(warn)
    return "\n".join(line for line in lines if line)


def check_model_updates() -> str:
    exe = shutil.which("lemonade-server") or shutil.which("lemonade")
    if not exe:
        return ""
    p = _run([exe, "check-updates"])
    return ((p.stdout or "") + (p.stderr or "")).strip()


def _tuning_lines(target: UserTarget) -> tuple[str, ...]:
    settings = load_tuning(probe(), largest_gguf_bytes(target))
    lines: list[str] = []
    for key in (
        "ctx_size",
        "global_timeout",
        "max_loaded_models",
        "llamacpp_backend",
        "llamacpp_args",
        "llamacpp_vulkan_args",
    ):
        if key in settings:
            lines.append(f"{key}={settings[key]}")
    return tuple(lines)


def _format_plan_mounts(mounts: tuple[BindMount, ...]) -> tuple[str, ...]:
    if not mounts:
        return ()
    _set_unit_order(mounts)
    lines: list[str] = []
    for mount in mounts:
        lines.append(_render_unit(mount).rstrip("\n"))
        lines.append(f"What={_planned_what(mount)}")
        lines.append(f"Where={mount.where}")
        lines.append(" ".join(propagation_command(mount)))
    return tuple(lines)


def publish_plan(target: UserTarget) -> str:
    """Planned units, mounts, and load settings. Does not write or call systemctl."""
    kind = detect()
    if not kind:
        return "lemonade not installed"
    sources = gguf_sources(target)
    lines: list[str] = []
    if kind == "snap":
        dest = extra_dir(kind)
        if dest == Path() or is_home_path(dest):
            raise RuntimeError(
                "snap extra_models_dir must be under /var/snap/lemonade-server/common"
            )
        mounts = _snap_mount_plan(target, dest, sources)
        lines.append(f"lemonade extra_models_dir={dest}")
        leak = _leak_plan_line(target.model_root, dest)
        if leak:
            lines.append(leak)
        if not mounts:
            lines.append(
                "no real GGUF files to publish (Lemonade cannot follow store symlinks)"
            )
        else:
            lines.extend(_format_plan_mounts(mounts))
    elif not sources:
        lines.append(
            "no real GGUF files to publish (Lemonade cannot follow store symlinks)"
        )
    else:
        lines.append(f"lemonade extra_models_dir={sources[0]}")
    lines.extend(left_out_lines(target))
    lines.extend(_tuning_lines(target))
    return "\n".join(lines)


def publish(target: UserTarget) -> str:
    """Bind real GGUF trees and set extra_models_dir. Snap dest is never /home."""
    kind = detect()
    if not kind:
        return "lemonade not installed"
    sources = gguf_sources(target)
    if kind == "snap":
        dest = extra_dir(kind)
        if dest == Path() or is_home_path(dest):
            raise RuntimeError(
                "snap extra_models_dir must be under /var/snap/lemonade-server/common"
            )
        _action_notes.clear()
        _before_units.clear()
        _after_units.clear()
        # The backup folder has to exist before any unmount. A failure here
        # leaves every mount where it is.
        _prepare_state_dirs()
        repair = _clear_embeddings_leak(target.model_root, dest)
        if repair:
            _note(repair)
        mounts = _snap_mount_plan(target, dest, sources)
        omitted = left_out_lines(target)
        _emit_left_out(omitted)
        if not mounts:
            prefix = "no real GGUF files to publish (Lemonade cannot follow store symlinks)"
            parts = [prefix, *omitted, *_action_notes]
            return "\n".join(part for part in parts if part)
        # Stop the daemon before replacing binds. A busy dest mount would
        # mkdir chat/ inside the user's model tree.
        tuning: list[str] = []

        def _ready() -> None:
            text = report_load_tuning(target)
            if text:
                tuning.append(text)

        unit = _apply_snap_mounts(dest, mounts, on_ready=_ready)
        visible = _visible_tree_count(mounts)
        if visible == 1:
            prefix = f"lemonade extra_models_dir={dest} via {unit}"
        else:
            prefix = f"lemonade extra_models_dir={dest} ({visible} trees)"
        parts = [prefix, *tuning, *_action_notes, *omitted]
        return "\n".join(part for part in parts if part)
    omitted = left_out_lines(target)
    _emit_left_out(omitted)
    if not sources:
        prefix = "no real GGUF files to publish (Lemonade cannot follow store symlinks)"
        parts = [prefix, *omitted]
        return "\n".join(part for part in parts if part)
    dest = sources[0]
    _set_extra_models_dir(dest)
    prefix = f"lemonade extra_models_dir={dest}"
    extra = report_load_tuning(target)
    parts = [prefix, extra, *omitted]
    return "\n".join(part for part in parts if part)
