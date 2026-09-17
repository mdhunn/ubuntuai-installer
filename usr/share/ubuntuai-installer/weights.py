"""Find, classify, link or move, and download model weights."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from domain import CatalogWeight, FileHash, FoundWeight, UserTarget
from paths import WEIGHTS_FILE

MIN_BYTES = 64 * 1024
SKIP_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
SKIP_WALK_DIRS = {".git", "__pycache__", "blobs", ".cache"}
WEIGHT_SUFFIXES = {
    ".gguf",
    ".ggml",
    ".safetensors",
    ".sft",
    ".onnx",
    ".ort",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".q4nx",
    ".exl2",
    ".npz",
}
FOLDER_SUBDIR = {
    "loras": "loras",
    "lora": "loras",
    "vae": "vae",
    "clip": "clip",
    "text_encoders": "clip",
    "clip_vision": "clip",
    "controlnet": "controlnet",
    "unet": "unet",
    "diffusion_models": "safetensors",
    "checkpoints": "safetensors",
    "upscale_models": "pytorch",
    "upscale": "pytorch",
    "ipadapter": "safetensors",
    "whisper": "whisper",
    "embeddings": "embeddings",
    "openmoss": "openmoss",
    "moss": "openmoss",
    "flm": "flm",
    "q4nx_files": "flm",
    "mmproj": "mmproj",
    "onnx": "onnx",
    "hf": "hf",
    "diffusers": "diffusers",
    "exl2": "exl2",
}
KNOWN_SUBDIRS = {
    "gguf",
    "safetensors",
    "mmproj",
    "loras",
    "vae",
    "clip",
    "whisper",
    "embeddings",
    "flm",
    "openmoss",
    "hf",
    "onnx",
    "pytorch",
    "diffusers",
    "controlnet",
    "unet",
    "exl2",
}
UA = "ubuntuai-installer/0.1"
HEX_LEN = {
    "md5": 32,
    "sha1": 40,
    "sha224": 56,
    "sha256": 64,
    "sha384": 96,
    "sha512": 128,
    "sha3_256": 64,
    "blake2s": 64,
    "blake2b": 128,
}
ALGO_ALIASES = {
    "sha-256": "sha256",
    "sha256": "sha256",
    "sha-1": "sha1",
    "sha1": "sha1",
    "sha-512": "sha512",
    "sha512": "sha512",
    "sha-384": "sha384",
    "sha384": "sha384",
    "sha-224": "sha224",
    "sha224": "sha224",
    "md5": "md5",
    "blake2b": "blake2b",
    "blake2s": "blake2s",
    "sha3-256": "sha3_256",
    "sha3_256": "sha3_256",
}
OID_RE = re.compile(
    r"(?i)\b(sha-?512|sha-?384|sha-?256|sha-?224|sha-?1|sha1|md5|blake2b|blake2s|sha3-256)\s*[:=]\s*([0-9a-f]{32,128})"
)
LABEL_RE = re.compile(
    r"(?is)(sha-?512|sha-?384|sha-?256|sha-?224|sha-?1|sha1|md5|blake2b|blake2s)"
    r"(?:\s*checksum)?\s*(?:[:：=]|is)\s*`?([0-9a-fA-F]{32,128})`?"
)


def normalize_algo(name: str) -> str | None:
    raw = (name or "").strip().lower().replace("_", "-")
    algo = ALGO_ALIASES.get(raw)
    if algo is None:
        candidate = raw.replace("-", "_")
        if candidate in hashlib.algorithms_available:
            algo = candidate
    if algo and algo in hashlib.algorithms_available:
        return algo
    return None


def _hex_ok(algo: str, digest: str) -> bool:
    digest = digest.lower()
    if not re.fullmatch(r"[0-9a-f]+", digest):
        return False
    want = HEX_LEN.get(algo)
    if want and len(digest) != want:
        return False
    return len(digest) >= 32


def parse_named_hash(text: str) -> FileHash | None:
    if not text:
        return None
    compact = text.strip()
    if ":" in compact and " " not in compact.split(":", 1)[0]:
        algo_raw, hexdigest = compact.split(":", 1)
        algo = normalize_algo(algo_raw)
        hexdigest = hexdigest.strip().lower()
        if algo and _hex_ok(algo, hexdigest):
            return FileHash(algo, hexdigest)
    match = OID_RE.search(text) or LABEL_RE.search(text)
    if not match:
        return None
    algo = normalize_algo(match.group(1))
    hexdigest = match.group(2).lower()
    if algo and _hex_ok(algo, hexdigest):
        return FileHash(algo, hexdigest)
    return None


def parse_hf_url(url: str) -> tuple[str, str] | None:
    try:
        parts = urllib.parse.urlparse(url)
    except ValueError:
        return None
    if parts.netloc not in {"huggingface.co", "hf.co"}:
        return None
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) < 5 or segs[2] not in {"resolve", "blob"}:
        return None
    repo = f"{segs[0]}/{segs[1]}"
    filename = "/".join(segs[4:])
    return repo, filename


def parse_github_release_url(url: str) -> tuple[str, str, str, str] | None:
    try:
        parts = urllib.parse.urlparse(url)
    except ValueError:
        return None
    if parts.netloc not in {"github.com", "www.github.com"}:
        return None
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) < 7 or segs[2] != "releases" or segs[3] != "download":
        return None
    return segs[0], segs[1], segs[4], "/".join(segs[5:])


def _http_json(url: str, *, data: bytes | None = None, timeout: int = 20) -> object | None:
    headers = {"User-Agent": UA}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _http_text(url: str, timeout: int = 15) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(1024 * 256)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return raw.decode("utf-8", errors="replace")


def _hash_from_hf_row(row: object, filename: str) -> FileHash | None:
    if not isinstance(row, dict):
        return None
    path = str(row.get("path") or row.get("rfilename") or "")
    if path not in {filename, Path(filename).name}:
        return None
    lfs = row.get("lfs") or {}
    if isinstance(lfs, dict):
        parsed = parse_named_hash(str(lfs.get("oid") or ""))
        if parsed:
            return parsed
        # git-lfs default oid is SHA-256. Hugging Face often omits the "sha256:" prefix.
        hexdigest = str(lfs.get("sha256") or lfs.get("oid") or "").lower()
        if hexdigest and _hex_ok("sha256", hexdigest):
            return FileHash("sha256", hexdigest)
    return None


def _hash_from_hf(url: str) -> FileHash | None:
    parsed = parse_hf_url(url)
    if parsed is None:
        return None
    repo, filename = parsed
    api = f"https://huggingface.co/api/models/{repo}/paths-info/main"
    body = json.dumps({"paths": [filename], "expand": True}).encode("utf-8")
    rows = _http_json(api, data=body)
    if isinstance(rows, list):
        for row in rows:
            found = _hash_from_hf_row(row, filename)
            if found:
                return found
    parent = Path(filename).parent.as_posix()
    tree_tail = "" if parent in {".", ""} else f"/{parent}"
    tree = f"https://huggingface.co/api/models/{repo}/tree/main{tree_tail}"
    rows = _http_json(tree)
    if isinstance(rows, list):
        for row in rows:
            found = _hash_from_hf_row(row, filename)
            if found:
                return found
    blob = f"https://huggingface.co/{repo}/blob/main/{filename}"
    html = _http_text(blob)
    if html:
        return parse_named_hash(html)
    return None


def _hash_from_github(url: str) -> FileHash | None:
    parsed = parse_github_release_url(url)
    if parsed is None:
        return None
    owner, repo, tag, filename = parsed
    api = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    data = _http_json(api)
    if not isinstance(data, dict):
        return None
    for asset in data.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("name") or "") != Path(filename).name:
            continue
        found = parse_named_hash(str(asset.get("digest") or ""))
        if found:
            return found
    html = _http_text(f"https://github.com/{owner}/{repo}/releases/tag/{tag}")
    if html:
        return parse_named_hash(html)
    return None


def _hash_from_sidecar(url: str) -> FileHash | None:
    suffixes = (".sha256", ".sha512", ".sha1", ".md5", ".checksum")
    for suffix in suffixes:
        text = _http_text(url + suffix)
        if not text:
            continue
        found = parse_named_hash(text)
        if found:
            return found
        hexdigest = text.split()[0].strip().lower() if text.split() else ""
        algo = normalize_algo(suffix.lstrip("."))
        if algo and _hex_ok(algo, hexdigest):
            return FileHash(algo, hexdigest)
    return None


def lookup_published_hash(url: str) -> FileHash | None:
    if not url:
        return None
    for finder in (_hash_from_hf, _hash_from_github, _hash_from_sidecar):
        found = finder(url)
        if found:
            return found
    html = _http_text(url)
    if html:
        return parse_named_hash(html)
    return None


def catalog_checksum(row: dict) -> tuple[str, str]:
    raw = str(row.get("checksum") or "").strip()
    parsed = parse_named_hash(raw) if raw else None
    if parsed:
        return parsed.algo, parsed.hexdigest
    legacy = str(row.get("sha256") or "").strip().lower()
    if legacy and _hex_ok("sha256", legacy):
        return "sha256", legacy
    return "", ""


def load_catalog(path: Path | None = None) -> tuple[CatalogWeight, ...]:
    src = path or WEIGHTS_FILE
    raw = json.loads(src.read_text(encoding="utf-8"))
    items = []
    seen: set[str] = set()
    for row in raw["models"]:
        algo, hexdigest = catalog_checksum(row)
        w = CatalogWeight(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            subdir=row["subdir"],
            filename=row["filename"],
            url=row["url"],
            bytes=int(row["bytes"]),
            workflows=tuple(row.get("workflows") or ()),
            hash_algo=algo,
            hash_hex=hexdigest,
            default=bool(row.get("default")),
            fmt=row.get("fmt") or _fmt_from_name(row["filename"]),
        )
        if w.id in seen:
            raise ValueError(f"duplicate weight id {w.id}")
        if w.subdir not in KNOWN_SUBDIRS:
            raise ValueError(f"unknown subdir {w.subdir} for {w.id}")
        seen.add(w.id)
        items.append(w)
    return tuple(items)


def human_bytes(n: int) -> str:
    for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= size:
            val = n / size
            return f"{val:.1f} {unit}" if val < 10 else f"{val:.0f} {unit}"
    return f"{n} B"


def _fmt_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".gguf": "gguf",
        ".ggml": "ggml",
        ".safetensors": "safetensors",
        ".sft": "safetensors",
        ".onnx": "onnx",
        ".ort": "onnx",
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".ckpt": "pytorch",
        ".bin": "pytorch",
        ".q4nx": "q4nx",
        ".exl2": "exl2",
        ".npz": "numpy",
    }.get(suffix, suffix.lstrip(".") or "unknown")


def classify(path: Path) -> str | None:
    name = path.name.lower()
    if name in SKIP_NAMES or name.startswith("put_") and name.endswith("_here"):
        return None
    if name.endswith(".index.json") or name.endswith(".json"):
        return None
    suffix = path.suffix.lower()
    if suffix not in WEIGHT_SUFFIXES:
        return None
    if "mmproj" in name:
        return "mmproj"
    if suffix == ".q4nx":
        return "flm"
    if "whisper" in name or (suffix == ".bin" and name.startswith("ggml-")):
        return "whisper"
    parts = {p.lower() for p in path.parts}
    for key, subdir in FOLDER_SUBDIR.items():
        if key in parts:
            return subdir
    if any(tok in name for tok in ("lora", ".lora")):
        return "loras"
    if "vae" in name:
        return "vae"
    if name.startswith("clip") or "text_encoder" in name:
        return "clip"
    if "embed" in name:
        return "embeddings"
    if "moss" in name:
        return "openmoss"
    if "controlnet" in name:
        return "controlnet"
    if suffix in {".gguf", ".ggml"}:
        return "gguf"
    if suffix in {".onnx", ".ort"}:
        return "onnx"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "pytorch"
    if suffix == ".exl2":
        return "exl2"
    if suffix in {".safetensors", ".sft"}:
        return "safetensors"
    if suffix == ".bin":
        return "pytorch"
    return None


def detect_bundle(path: Path) -> tuple[str, str] | None:
    """Return (subdir, fmt) for a Hugging Face, Diffusers, or EXL2 directory."""
    if not path.is_dir():
        return None
    if (path / "model_index.json").is_file():
        return "diffusers", "diffusers"
    has_config = (path / "config.json").is_file()
    if not has_config:
        return None
    try:
        names = [p.name.lower() for p in path.iterdir() if p.is_file() or p.is_symlink()]
    except OSError:
        return None
    safetensors = [n for n in names if n.endswith(".safetensors")]
    onnx = [n for n in names if n.endswith(".onnx")]
    bins = [n for n in names if n.startswith("pytorch_model") and n.endswith(".bin")]
    if (path / "measurement.json").is_file() or any("exl2" in n for n in names):
        if safetensors or bins:
            return "exl2", "exl2"
    if safetensors or bins:
        return "hf", "safetensors"
    if onnx:
        return "onnx", "onnx"
    return None


def bundle_dest_name(path: Path) -> str:
    name = path.name
    if path.parent.name == "snapshots":
        repo = path.parent.parent.name
        if repo.startswith("models--"):
            return repo.removeprefix("models--")
        return repo
    if name.startswith("models--"):
        return name.removeprefix("models--")
    return name


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def builtin_scan_roots(home: Path, model_root: Path) -> tuple[Path, ...]:
    return (
        home / "AI models",
        home / "Downloads",
        home / ".cache" / "huggingface" / "hub",
        home / "Projects" / "AI Apps" / "ComfyUI" / "models",
        home / "Projects" / "AI Apps" / "q4nx_files",
        home / "Projects" / "AI Apps" / "gguf_files",
        model_root,
    )


def normalize_scan_folder(raw: str | Path, home: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = home / p
    try:
        p = p.resolve()
    except OSError as exc:
        raise ValueError(f"cannot resolve {raw}") from exc
    if p == Path("/"):
        raise ValueError("refusing to scan /")
    if not p.is_dir():
        raise ValueError(f"not a directory: {p}")
    return p


def scan_roots(
    home: Path,
    model_root: Path,
    extra: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    candidates = [*builtin_scan_roots(home, model_root), *extra]
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in candidates:
        try:
            p = raw.resolve()
        except OSError:
            continue
        if not p.is_dir():
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return tuple(out)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _already_in_store(path: Path, model_root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(model_root.resolve())
    except (ValueError, OSError):
        return False
    parts = rel.parts
    return bool(parts) and parts[0] in KNOWN_SUBDIRS


def _dest_for(
    src: Path,
    resolved: Path,
    subdir: str,
    model_root: Path,
    used: dict[tuple[str, str], Path],
    preferred: str | None = None,
) -> tuple[str, str]:
    base = preferred or src.name
    if _already_in_store(src, model_root):
        return base, "already"
    names = [base, f"{src.parent.name}-{base}"]
    n = 2
    while True:
        for dest_name in names:
            key = (subdir, dest_name)
            dest = model_root / subdir / dest_name
            if dest.exists() or dest.is_symlink():
                try:
                    if dest.resolve() == resolved:
                        return dest_name, "already"
                except OSError:
                    pass
                continue
            if key in used:
                continue
            return dest_name, "new"
        names = [f"{src.parent.name}-{n}-{base}"]
        n += 1


def _append(
    found: list[FoundWeight],
    used: dict[tuple[str, str], Path],
    seen_src: set[Path],
    src: Path,
    resolved: Path,
    subdir: str,
    model_root: Path,
    *,
    size: int,
    fmt: str,
    kind: str,
    preferred: str | None = None,
) -> None:
    if resolved in seen_src:
        return
    seen_src.add(resolved)
    dest_name, state = _dest_for(
        src, resolved, subdir, model_root, used, preferred=preferred
    )
    used[(subdir, dest_name)] = resolved
    found.append(
        FoundWeight(
            path=src,
            size=size,
            subdir=subdir,
            dest_name=dest_name,
            state=state,
            fmt=fmt,
            kind=kind,
        )
    )


def scan(roots: tuple[Path, ...], model_root: Path) -> tuple[FoundWeight, ...]:
    found: list[FoundWeight] = []
    used: dict[tuple[str, str], Path] = {}
    seen_src: set[Path] = set()
    try:
        store = model_root.resolve()
    except OSError:
        store = model_root
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in SKIP_WALK_DIRS]
            current = Path(dirpath)
            try:
                resolved_dir = current.resolve()
            except OSError:
                continue
            if resolved_dir != store:
                bundle = detect_bundle(current)
                if bundle:
                    subdir, fmt = bundle
                    size = _dir_size(current)
                    if size >= MIN_BYTES:
                        _append(
                            found,
                            used,
                            seen_src,
                            current,
                            resolved_dir,
                            subdir,
                            model_root,
                            size=size,
                            fmt=fmt,
                            kind="dir",
                            preferred=bundle_dest_name(current),
                        )
                    dirnames[:] = []
                    continue
            for fname in filenames:
                src = Path(dirpath) / fname
                try:
                    resolved = src.resolve()
                    size = src.stat().st_size
                except OSError:
                    continue
                if size < MIN_BYTES:
                    continue
                subdir = classify(src)
                if subdir is None:
                    continue
                _append(
                    found,
                    used,
                    seen_src,
                    src,
                    resolved,
                    subdir,
                    model_root,
                    size=size,
                    fmt=_fmt_from_name(src.name),
                    kind="file",
                )
    found.sort(key=lambda w: (w.subdir, w.dest_name.lower()))
    return tuple(found)


def store_inventory(model_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not model_root.is_dir():
        return counts
    for sub in sorted(KNOWN_SUBDIRS):
        folder = model_root / sub
        if not folder.is_dir():
            continue
        n = sum(1 for p in folder.iterdir() if p.name not in SKIP_NAMES)
        if n:
            counts[sub] = n
    return counts


def hash_file(path: Path, algo: str) -> str:
    name = normalize_algo(algo)
    if name is None:
        raise ValueError(f"unsupported hash algorithm {algo!r}")
    digest = hashlib.new(name)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path, algo: str) -> str:
    name = normalize_algo(algo)
    if name is None:
        raise ValueError(f"unsupported hash algorithm {algo!r}")
    if path.is_dir() and not path.is_symlink():
        digest = hashlib.new(name)
        files = sorted(p for p in path.rglob("*") if p.is_file())
        for p in files:
            rel = p.relative_to(path).as_posix().encode()
            digest.update(rel)
            digest.update(b"\0")
            digest.update(bytes.fromhex(hash_file(p, name)))
        return digest.hexdigest()
    return hash_file(path, name)


def integrity_algo_for(item: FoundWeight) -> str:
    try:
        catalog = load_catalog()
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        catalog = ()
    for model in catalog:
        if model.filename not in {item.path.name, item.dest_name}:
            continue
        published = model.published_hash()
        if published:
            return published.algo
    return "sha256"


def _remove_source(path: Path, kind: str) -> None:
    if kind == "dir" and path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _copy_into_store(item: FoundWeight, dest: Path, uid: int | None, gid: int | None) -> None:
    if item.kind == "dir":
        shutil.copytree(item.path, dest, symlinks=True)
        if uid is not None and gid is not None:
            for dirpath, _dirnames, filenames in os.walk(dest):
                os.chown(dirpath, uid, gid)
                for name in filenames:
                    os.chown(Path(dirpath) / name, uid, gid)
        return
    shutil.copy2(item.path, dest)
    if uid is not None and gid is not None:
        os.chown(dest, uid, gid)


def organize(
    items: tuple[FoundWeight, ...],
    model_root: Path,
    *,
    mode: str = "link",
    remove_source: bool = False,
    uid: int | None = None,
    gid: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    if mode not in {"link", "copy", "move"}:
        raise ValueError("mode must be link, copy, or move")
    want_remove = False
    if mode == "copy":
        want_remove = remove_source
    elif mode == "move":
        want_remove = True
    log: list[str] = []
    try:
        store = model_root.resolve()
    except OSError:
        store = model_root
    for item in items:
        dest = model_root / item.subdir / item.dest_name
        if item.state == "already":
            log.append(f"already {dest}")
            continue
        if item.state == "exists":
            log.append(f"skip {item.path} -> {dest} (name taken)")
            continue
        extra = " then remove source" if want_remove else ""
        if dry_run:
            log.append(f"{mode} {item.path} -> {dest}{extra}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if uid is not None and gid is not None:
            os.chown(dest.parent, uid, gid)
        if dest.exists() or dest.is_symlink():
            log.append(f"skip {dest} (appeared)")
            continue
        parent_writable = os.access(item.path.parent, os.W_OK)
        if mode in {"copy", "move"} and (mode == "copy" or parent_writable):
            algo = integrity_algo_for(item)
            src_hash = hash_path(item.path, algo)
            _copy_into_store(item, dest, uid, gid)
            dest_hash = hash_path(dest, algo)
            if dest_hash != src_hash:
                _remove_source(dest, item.kind)
                log.append(f"checksum mismatch after {mode} {item.path} -> {dest} ({algo})")
                continue
            verb = "copied" if mode == "copy" else "moved"
            log.append(f"{verb} {item.path} -> {dest} {algo}={dest_hash[:12]}")
            if want_remove:
                if not parent_writable:
                    log.append(f"kept source {item.path} (not writable)")
                elif _under(item.path, store):
                    log.append(f"kept source {item.path} (inside store)")
                else:
                    _remove_source(item.path, item.kind)
                    log.append(f"removed source {item.path} after checksum match")
            continue
        dest.symlink_to(item.path)
        if uid is not None and gid is not None:
            os.lchown(dest, uid, gid)
        why = "link" if mode == "link" else "link (source not writable)"
        log.append(f"{why} {item.path} -> {dest}")
    return log


def catalog_dest(model: CatalogWeight, model_root: Path) -> Path:
    return model_root / model.subdir / model.filename


def download(
    model: CatalogWeight,
    model_root: Path,
    *,
    on_progress: object | None = None,
    uid: int | None = None,
    gid: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    expected: FileHash | None = None,
) -> str:
    dest = catalog_dest(model, model_root)
    if (dest.exists() or dest.is_symlink()) and not force:
        return f"already {dest}"
    if dry_run:
        return f"download {model.id} -> {dest} ({human_bytes(model.bytes)})"
    if force and (dest.exists() or dest.is_symlink()):
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uid is not None and gid is not None:
        os.chown(dest.parent, uid, gid)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(model.url, headers={"User-Agent": UA})
    written = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0) or model.bytes
        with open(part, "wb") as fh:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                if on_progress:
                    on_progress(written, total)
    want = expected or model.published_hash() or lookup_published_hash(model.url)
    if want:
        digest = hash_file(part, want.algo)
        if digest != want.hexdigest.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{want.algo} mismatch for {model.id}")
    part.replace(dest)
    if uid is not None and gid is not None:
        os.chown(dest, uid, gid)
    return f"downloaded {dest} ({human_bytes(written)})"


def ensure_weight(
    model: CatalogWeight,
    target: UserTarget,
    extra: tuple[Path, ...] = (),
    *,
    on_progress: object | None = None,
    dry_run: bool = False,
) -> str:
    dest = catalog_dest(model, target.model_root)
    if dest.exists() or dest.is_symlink():
        return f"already {dest}"
    roots = scan_roots(target.home, target.model_root, extra)
    for item in scan(roots, target.model_root):
        if item.path.name != model.filename:
            continue
        try:
            if item.path.resolve() == dest.resolve():
                return f"already {dest}"
        except OSError:
            pass
        if dry_run:
            return f"link {item.path} -> {dest}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if target.uid is not None:
            os.chown(dest.parent, target.uid, target.gid)
        dest.symlink_to(item.path)
        if target.uid is not None:
            os.lchown(dest, target.uid, target.gid)
        return f"linked {item.path} -> {dest}"

    def prog(done: int, total: int) -> None:
        if on_progress:
            on_progress(
                f"Downloading {model.filename}: {human_bytes(done)} / {human_bytes(total)}"
            )

    return download(
        model,
        target.model_root,
        on_progress=prog,
        uid=target.uid,
        gid=target.gid,
        dry_run=dry_run,
    )
