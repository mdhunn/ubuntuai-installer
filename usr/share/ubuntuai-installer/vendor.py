"""Install pinned vendor runtimes into ~/.local."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

from domain import Hardware, UserTarget
from paths import VENDORS_FILE
from weights import UA, hash_file, human_bytes, lookup_published_hash

WRAPPER = """#!/bin/sh
DIR="{libdir}"
export LD_LIBRARY_PATH="$DIR${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
exec "$DIR/{binary}" "$@"
"""

LAUNCHER = """#!/bin/sh
DIR="{libdir}"
MODELS="${{UBUNTUAI_MODELS:-$HOME/Models}}/openmoss"
MODEL=""
for f in "$MODELS"/moss-tts-local*.gguf "$MODELS"/*.gguf; do
  case "$f" in
    *.extras.gguf) continue ;;
    *.gguf) MODEL="$f"; break ;;
  esac
done
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "ubuntuai-openmoss: no GGUF in $MODELS" >&2
  exit 1
fi
HOST="${{UBUNTUAI_BIND:-127.0.0.1}}"
export LD_LIBRARY_PATH="$DIR${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
exec "$DIR/moss-tts-server" --model "$MODEL" --host "$HOST" --port {port} --webui-dir "$DIR/webui" "$@"
"""


def load_vendors(path: Path | None = None) -> dict:
    src = path or VENDORS_FILE
    return json.loads(src.read_text(encoding="utf-8"))


def pick_archive(spec: dict, hw: Hardware) -> dict:
    archives = spec["archives"]
    if "rocm" in hw.backends() and "rocm" in archives:
        return archives["rocm"]
    if "vulkan" in archives:
        return archives["vulkan"]
    return next(iter(archives.values()))


def vendor_libdir(home: Path, vendor_id: str) -> Path:
    return home / ".local" / "lib" / "ubuntuai" / vendor_id


def vendor_bindir(home: Path) -> Path:
    return home / ".local" / "bin"


def vendor_cachedir(home: Path) -> Path:
    return home / ".cache" / "ubuntuai" / "vendor"


def vendor_present(spec: dict, home: Path) -> bool:
    lib = vendor_libdir(home, _id_from_spec(spec))
    for name in spec.get("binaries") or ():
        if (lib / name).is_file():
            return True
        if shutil.which(name):
            return True
    wrap = spec.get("wrapper")
    if wrap and (vendor_bindir(home) / wrap).is_file():
        return True
    return False


def _id_from_spec(spec: dict) -> str:
    return spec.get("id") or "openmoss"


def vendor_versions_path(home: Path) -> Path:
    return home / ".config" / "ubuntuai" / "vendor-versions.json"


def recorded_vendor_version(home: Path, vendor_id: str) -> str:
    path = vendor_versions_path(home)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get(vendor_id) or "")


def record_vendor_version(home: Path, vendor_id: str, version: str, uid: int, gid: int) -> None:
    path = vendor_versions_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    data[vendor_id] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.chown(path, uid, gid)


def install_vendor(
    vendor_id: str,
    hw: Hardware,
    target: UserTarget,
    *,
    on_progress: object | None = None,
    dry_run: bool = False,
    vendors: dict | None = None,
    force: bool = False,
    archive_url: str = "",
    version: str = "",
) -> str:
    table = vendors or load_vendors()
    spec = table.get(vendor_id)
    if spec is None:
        raise KeyError(f"unknown vendor {vendor_id}")
    spec = dict(spec)
    spec["id"] = vendor_id
    libdir = vendor_libdir(target.home, vendor_id)
    if vendor_present(spec, target.home) and not force:
        return f"already {vendor_id}"
    if force and libdir.exists():
        shutil.rmtree(libdir)
    archive = dict(pick_archive(spec, hw))
    if archive_url:
        archive["url"] = archive_url
        name = Path(urllib.parse.urlparse(archive_url).path).name
        if name:
            archive["filename"] = name
    bindir = vendor_bindir(target.home)
    cache = vendor_cachedir(target.home)
    tarball = cache / archive["filename"]
    if dry_run:
        return (
            f"download {archive['url']} -> {libdir} "
            f"({human_bytes(int(archive.get('bytes') or 0))})"
        )
    cache.mkdir(parents=True, exist_ok=True)
    if not tarball.is_file():
        _download(archive["url"], tarball, int(archive.get("bytes") or 0), on_progress)
    published = lookup_published_hash(str(archive.get("url") or ""))
    if published and tarball.is_file():
        actual = hash_file(tarball, published.algo)
        if actual != published.hexdigest:
            tarball.unlink(missing_ok=True)
            raise RuntimeError(
                f"{vendor_id} {published.algo} mismatch for {tarball.name}"
            )
    libdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tar:
        tar.extractall(libdir, filter="data")
    _chown_tree(libdir, target.uid, target.gid)
    bindir.mkdir(parents=True, exist_ok=True)
    for binary in spec.get("binaries") or ():
        src = libdir / binary
        if not src.is_file():
            raise RuntimeError(f"{vendor_id} archive missing {binary}")
        wrap = bindir / binary
        wrap.write_text(
            WRAPPER.format(libdir=str(libdir), binary=binary),
            encoding="utf-8",
        )
        os.chmod(wrap, 0o755)
        os.chown(wrap, target.uid, target.gid)
    launcher = spec.get("launcher")
    if launcher:
        script = bindir / launcher
        script.write_text(
            LAUNCHER.format(libdir=str(libdir), port=int(spec.get("port") or 8081)),
            encoding="utf-8",
        )
        os.chmod(script, 0o755)
        os.chown(script, target.uid, target.gid)
    os.chown(bindir, target.uid, target.gid)
    record_vendor_version(
        target.home,
        vendor_id,
        version or str(spec.get("version") or ""),
        target.uid,
        target.gid,
    )
    return f"installed {vendor_id} to {libdir}"


def _download(url: str, dest: Path, total_hint: int, on_progress: object | None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    part = dest.with_name(dest.name + ".part")
    written = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0) or total_hint
        with open(part, "wb") as fh:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                if on_progress:
                    on_progress(f"Downloading {dest.name}: {human_bytes(written)} / {human_bytes(total)}")
    part.replace(dest)


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        os.chown(dirpath, uid, gid)
        for name in filenames:
            os.chown(Path(dirpath) / name, uid, gid)
