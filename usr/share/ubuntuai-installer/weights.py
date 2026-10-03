"""Find, classify, copy or move, and download model weights."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import struct
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from domain import CatalogWeight, FileHash, FoundWeight, UserTarget
from paths import DEFAULT_MODEL_DIRNAME, LEGACY_MODEL_DIRNAME, WEIGHTS_FILE

MIN_BYTES = 64 * 1024
# One percent, and at least 1 MiB, so a copy that barely fits still has room to finish.
COPY_FREE_MARGIN_MIN = 1024 * 1024
COPY_PART_SUFFIX = ".part"
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
FOREIGN_FSTYPES = frozenset({"ntfs", "fuseblk", "vfat", "exfat"})
_FOREIGN_PREFIXES = (Path("/media"), Path("/mnt"))
ANOTHER_DISK = "another disk"
UBUNTU_DISK = "this computer's Ubuntu disk"
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


def _size_from_hf_row(row: object, filename: str) -> int:
    if not isinstance(row, dict):
        return 0
    path = str(row.get("path") or row.get("rfilename") or "")
    if path not in {filename, Path(filename).name}:
        return 0
    lfs = row.get("lfs") or {}
    if isinstance(lfs, dict):
        try:
            n = int(lfs.get("size") or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            return n
    try:
        return int(row.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def _meta_from_hf(url: str) -> tuple[FileHash | None, int]:
    parsed = parse_hf_url(url)
    if parsed is None:
        return None, 0
    repo, filename = parsed
    found = None
    size = 0
    api = f"https://huggingface.co/api/models/{repo}/paths-info/main"
    body = json.dumps({"paths": [filename], "expand": True}).encode("utf-8")
    rows = _http_json(api, data=body)
    if isinstance(rows, list):
        for row in rows:
            if found is None:
                found = _hash_from_hf_row(row, filename)
            if size <= 0:
                size = _size_from_hf_row(row, filename)
            if found and size:
                return found, size
    parent = Path(filename).parent.as_posix()
    tree_tail = "" if parent in {".", ""} else f"/{parent}"
    tree = f"https://huggingface.co/api/models/{repo}/tree/main{tree_tail}"
    rows = _http_json(tree)
    if isinstance(rows, list):
        for row in rows:
            if found is None:
                found = _hash_from_hf_row(row, filename)
            if size <= 0:
                size = _size_from_hf_row(row, filename)
            if found and size:
                return found, size
    if found is None:
        blob = f"https://huggingface.co/{repo}/blob/main/{filename}"
        html = _http_text(blob)
        if html:
            found = parse_named_hash(html)
    return found, size


def _hash_from_hf(url: str) -> FileHash | None:
    found, _size = _meta_from_hf(url)
    return found


def _meta_from_github(url: str) -> tuple[FileHash | None, int]:
    parsed = parse_github_release_url(url)
    if parsed is None:
        return None, 0
    owner, repo, tag, filename = parsed
    api = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    data = _http_json(api)
    if isinstance(data, dict):
        for asset in data.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            if str(asset.get("name") or "") != Path(filename).name:
                continue
            found = parse_named_hash(str(asset.get("digest") or ""))
            try:
                size = int(asset.get("size") or 0)
            except (TypeError, ValueError):
                size = 0
            return found, size
    html = _http_text(f"https://github.com/{owner}/{repo}/releases/tag/{tag}")
    if html:
        return parse_named_hash(html), 0
    return None, 0


def _hash_from_github(url: str) -> FileHash | None:
    found, _size = _meta_from_github(url)
    return found


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


def lookup_published_meta(url: str) -> tuple[FileHash | None, int]:
    """Hash and byte size published on the download page or API. Either may be missing."""
    if not url:
        return None, 0
    found, size = _meta_from_hf(url)
    if found is None or size <= 0:
        gh_found, gh_size = _meta_from_github(url)
        found = found or gh_found
        if size <= 0:
            size = gh_size
    if found is None:
        found = _hash_from_sidecar(url)
    if found is None:
        html = _http_text(url)
        if html:
            found = parse_named_hash(html)
    return found, size


def lookup_published_hash(url: str) -> FileHash | None:
    found, _size = lookup_published_meta(url)
    return found


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


# llama.cpp general.architecture strings for text generation. A name that is
# not in this set is not published as chat. Speech, image, and reranker
# strings are listed on their own so the publish report can name them.
_LLAMA_CPP_ARCHITECTURES = frozenset({
    "afmoe", "apertus", "arctic", "arcee", "arwkv7", "baichuan", "bailingmoe",
    "bailingmoe2", "bailingmoe3", "bert", "bitnet", "bloom", "chameleon",
    "chatglm", "clef", "clip", "codeshell", "cogvlm", "cohere2", "cohere2moe",
    "command-r", "dbrx", "deci", "deepseek", "deepseek2", "deepseek2-ocr",
    "deepseek32", "deepseek4", "dflash", "dots1", "dots3note", "dream",
    "eagle3", "ernie4_5", "ernie4_5-moe", "eurobert", "exaone", "exaone-moe",
    "exaone4", "falcon", "falcon-h1", "gemma", "gemma-embedding", "gemma2",
    "gemma3", "gemma3n", "gemma4", "gemma4-assistant", "glm-dsa", "glm4",
    "glm4moe", "glm5-next", "gpt-oss", "gpt2", "gptj", "gptneox", "granite",
    "granite_swa", "granitehybrid", "granitemoe", "graniteswitch", "grok",
    "grovemoe", "hrm_text", "hunyuan-dense", "hunyuan-moe", "hunyuan_vl",
    "hy_v3", "hy_v4", "internlm2", "jais", "jais2", "jamba", "jina-bert-v2",
    "jina-bert-v3", "kimi-k3", "kimi-linear", "laguna", "lfm2", "lfm2moe",
    "llama", "llama-embed", "llama4", "llada", "llada-moe", "maincoder",
    "mamba", "mamba2", "maple", "mellum", "mimo2", "minicpm", "minicpm3",
    "minimax-01", "minimax-m2", "minimax-m3", "mistral3", "mistral4",
    "modern-bert", "mpt", "muse-glimmer", "nanbeige", "nemotron", "nemotron_h",
    "nemotron_h_moe", "neo-bert", "nomic-bert", "nomic-bert-moe", "olmo",
    "olmo2", "olmoe", "openelm", "orion", "paddleocr", "pangu-embedded",
    "phi2", "phi3", "phimoe", "plamo", "plamo2", "plamo3", "plm", "pockettts",
    "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen35", "qwen35moe",
    "qwen3moe", "qwen3next", "qwen3tts", "qwen3vl", "qwen3vlmoe", "qwen4exp",
    "refact", "rnd1", "rwkv6", "rwkv6qwen2", "rwkv7", "seed_oss", "smallthinker",
    "smollm3", "spark2_5", "stablelm", "starcoder", "starcoder2", "step35",
    "t5", "t5encoder", "talkie", "wavtokenizer-dec", "xverse",
})
_COMPANION_ARCHITECTURES = frozenset({"clip", "eagle3", "dflash"})
_NOT_CHAT_ARCHITECTURES = frozenset({
    "bert", "deepseek2-ocr", "eurobert", "gemma-embedding", "jina-bert-v2",
    "jina-bert-v3", "llama-embed", "modern-bert", "neo-bert", "nomic-bert",
    "nomic-bert-moe", "paddleocr", "pangu-embedded", "pockettts", "qwen3tts",
    "t5encoder", "talkie", "wavtokenizer-dec",
})
_TTS_ARCHITECTURES = frozenset({
    "magpie", "magpie-tts", "moss-tts-delay", "nano-codec", "pockettts",
    "qwen3tts", "talkie", "wavtokenizer-dec",
})
_DIFFUSION_ARCHITECTURES = frozenset({
    "aura", "chroma", "cosmos", "flux", "flux2", "hidream", "hyvid", "ltxv",
    "lumina2", "qwen_image", "sd1", "sd2", "sd3", "sdxl", "svd", "wan",
})
CHAT_GGUF_ARCHITECTURES = (
    _LLAMA_CPP_ARCHITECTURES
    - _COMPANION_ARCHITECTURES
    - _NOT_CHAT_ARCHITECTURES
    - _TTS_ARCHITECTURES
)
_RERANK_POOLING = 4
_RERANK_TYPES = frozenset({"rerank", "reranking"})
_GGUF_MAGIC = b"GGUF"
_GGUF_HEADER_CAP = 2 * 1024 * 1024
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


@dataclass(frozen=True)
class GgufInfo:
    architecture: str | None
    model_type: str | None
    pooling: int | None


def gguf_architecture_header(
    architecture: str,
    *,
    pooling: int | None = None,
    model_type: str | None = None,
) -> bytes:
    """Minimal GGUF header. Tests use it. It is not a model file."""
    fields: list[tuple[str, object]] = [("general.architecture", architecture)]
    if model_type is not None:
        fields.append(("general.type", model_type))
    if pooling is not None:
        fields.append((f"{architecture}.pooling_type", pooling))
    parts = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0), struct.pack("<Q", len(fields))]
    for key, value in fields:
        key_b = key.encode("utf-8")
        parts.append(struct.pack("<Q", len(key_b)))
        parts.append(key_b)
        if isinstance(value, int):
            parts.append(struct.pack("<I", 4))
            parts.append(struct.pack("<I", value))
        else:
            val_b = str(value).encode("utf-8")
            parts.append(struct.pack("<I", _GGUF_STRING))
            parts.append(struct.pack("<Q", len(val_b)))
            parts.append(val_b)
    return b"".join(parts)


def _gguf_string(blob: bytes, off: int) -> tuple[str | None, int | None]:
    if off + 8 > len(blob):
        return None, None
    n = struct.unpack_from("<Q", blob, off)[0]
    off += 8
    if n > len(blob) or off + n > len(blob):
        return None, None
    try:
        text = blob[off : off + n].decode("utf-8")
    except UnicodeDecodeError:
        return None, None
    return text, off + n


def _gguf_skip(blob: bytes, off: int, vtype: int) -> int | None:
    if vtype == _GGUF_STRING:
        _, nxt = _gguf_string(blob, off)
        return nxt
    if vtype == _GGUF_ARRAY:
        if off + 12 > len(blob):
            return None
        et = struct.unpack_from("<I", blob, off)[0]
        count = struct.unpack_from("<Q", blob, off + 4)[0]
        off += 12
        if count > 1_000_000:
            return None
        for _ in range(count):
            nxt = _gguf_skip(blob, off, et)
            if nxt is None:
                return None
            off = nxt
        return off
    size = _GGUF_SIZES.get(vtype)
    if size is None or off + size > len(blob):
        return None
    return off + size


def read_gguf_info(path: Path) -> GgufInfo | None:
    """Read general.architecture. None when the header is missing or cut off.

    The scan stops at 2 MiB. A header that hides the architecture past that
    cap is treated as unreadable. Unknown architectures are not chat.
    """
    try:
        with path.open("rb") as handle:
            blob = handle.read(_GGUF_HEADER_CAP)
    except OSError:
        return None
    if len(blob) < 24 or blob[:4] != _GGUF_MAGIC:
        return None
    version = struct.unpack_from("<I", blob, 4)[0]
    if version < 2 or version > 3:
        return None
    n_kv = struct.unpack_from("<Q", blob, 16)[0]
    if n_kv > 1_000_000:
        return None
    off = 24
    architecture: str | None = None
    model_type: str | None = None
    pooling: int | None = None
    for _ in range(n_kv):
        key, off = _gguf_string(blob, off)
        if key is None or off is None or off + 4 > len(blob):
            break
        vtype = struct.unpack_from("<I", blob, off)[0]
        off += 4
        if key == "general.architecture" and vtype == _GGUF_STRING:
            value, off = _gguf_string(blob, off)
            if value is None or off is None:
                break
            architecture = value.strip().lower() or None
            continue
        if key == "general.type" and vtype == _GGUF_STRING:
            value, off = _gguf_string(blob, off)
            if value is None or off is None:
                break
            model_type = value.strip().lower() or None
            continue
        if (
            architecture
            and key == f"{architecture}.pooling_type"
            and vtype == 4
            and off + 4 <= len(blob)
        ):
            pooling = struct.unpack_from("<I", blob, off)[0]
            off += 4
            continue
        nxt = _gguf_skip(blob, off, vtype)
        if nxt is None:
            break
        off = nxt
        if architecture and model_type is not None and pooling is not None:
            break
    if not architecture:
        return None
    return GgufInfo(architecture, model_type, pooling)


def gguf_architecture(path: Path) -> str | None:
    info = read_gguf_info(path)
    if info is None:
        return None
    return info.architecture


def gguf_publish_role(path: Path) -> str:
    """chat, companion, tts, diffusion, rerank, unknown, or truncated.

    mmproj files and clip, eagle3, and dflash architectures are companions.
    They do not count as chat and they do not block a chat directory.
    A reranker is pooling type rank or general.type rerank, even when the
    architecture is also a text model. Anything else this table does not
    know stays out of chat.
    """
    if "mmproj" in path.name.lower():
        return "companion"
    info = read_gguf_info(path)
    if info is None or not info.architecture:
        return "truncated"
    arch = info.architecture
    if info.model_type in _RERANK_TYPES or info.pooling == _RERANK_POOLING:
        return "rerank"
    if arch in _TTS_ARCHITECTURES or "tts" in arch:
        return "tts"
    if arch in _DIFFUSION_ARCHITECTURES:
        return "diffusion"
    if arch in _COMPANION_ARCHITECTURES:
        return "companion"
    if arch in CHAT_GGUF_ARCHITECTURES:
        return "chat"
    return "unknown"


SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def detect_shard_bundle(path: Path) -> bool:
    """True when this folder holds a numbered GGUF shard set (keep it intact)."""
    if not path.is_dir():
        return False
    try:
        names = [p.name for p in path.iterdir() if p.is_file() or p.is_symlink()]
    except OSError:
        return False
    totals: dict[int, int] = {}
    for name in names:
        match = SHARD_RE.search(name)
        if not match:
            continue
        total = int(match.group(2))
        totals[total] = totals.get(total, 0) + 1
    return any(count >= 2 and total >= 2 for total, count in totals.items())


def detect_bundle(path: Path) -> tuple[str, str] | None:
    """Return (subdir, fmt) for a Hugging Face, Diffusers, shard, or EXL2 directory."""
    if not path.is_dir():
        return None
    if detect_shard_bundle(path):
        return "gguf", "gguf"
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
        home / DEFAULT_MODEL_DIRNAME,
        home / LEGACY_MODEL_DIRNAME,
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
            foreign=is_foreign_mount(src),
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


class ForeignMountError(ValueError):
    """Move and source removal are refused on a foreign mount."""


def _unescape_mount(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 3 < n and text[i + 1 : i + 4].isdigit():
            try:
                out.append(chr(int(text[i + 1 : i + 4], 8)))
            except ValueError:
                out.append(text[i])
                i += 1
                continue
            i += 4
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _parse_mountinfo(text: str) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        try:
            sep = fields.index("-")
        except ValueError:
            continue
        if sep < 5 or sep + 1 >= len(fields):
            continue
        mountpoint = _unescape_mount(fields[4])
        if not mountpoint.startswith("/"):
            continue
        rows.append((mountpoint, fields[sep + 1].lower()))
    return tuple(rows)


def _mount_rows(mountinfo: str | None) -> tuple[tuple[str, str], ...]:
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return ()
    return _parse_mountinfo(mountinfo)


def _path_forms(path: Path) -> tuple[Path, ...]:
    forms: list[Path] = []
    raw = path if path.is_absolute() else path.absolute()
    forms.append(raw)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = None
    if resolved is not None and resolved not in forms:
        forms.append(resolved)
    return tuple(forms)


def _on_foreign_prefix(path: Path) -> bool:
    for cand in _path_forms(path):
        for root in _FOREIGN_PREFIXES:
            if cand == root or cand.is_relative_to(root):
                return True
    return False


def _fstype(path: Path, rows: tuple[tuple[str, str], ...]) -> str:
    best = ""
    best_len = -1
    for cand in _path_forms(path):
        for mountpoint, fstype in rows:
            mp = Path(mountpoint)
            try:
                inside = cand == mp or cand.is_relative_to(mp)
            except (TypeError, ValueError):
                inside = False
            if not inside:
                continue
            score = len(mp.parts)
            if score > best_len:
                best = fstype
                best_len = score
    return best


def is_foreign_mount(path: Path, mountinfo: str | None = None) -> bool:
    """True for /media, /mnt, or fstype ntfs, fuseblk, vfat, or exfat."""
    if _on_foreign_prefix(path):
        return True
    return _fstype(path, _mount_rows(mountinfo)) in FOREIGN_FSTYPES


def foreign_source(item: FoundWeight) -> bool:
    """True when this weight is copy only. Move stays off."""
    return bool(item.foreign) or is_foreign_mount(item.path)


def disk_words(foreign: bool) -> str:
    """Place name a Weights tab can bind. True is another disk."""
    return ANOTHER_DISK if foreign else UBUNTU_DISK


def move_off(items: Iterable[FoundWeight]) -> bool:
    """True when any selected weight is on another disk. Bind Move to the inverse."""
    return any(foreign_source(item) for item in items)


def _refuse_foreign_mutate(
    items: tuple[FoundWeight, ...],
    mode: str,
    remove_source: bool,
) -> None:
    if mode != "move" and not remove_source:
        return
    for item in items:
        if item.state in {"already", "exists"}:
            continue
        if not foreign_source(item):
            continue
        if mode == "move":
            raise ForeignMountError(
                f"Move is refused for {item.path}. The file is on {ANOTHER_DISK}. Copy only."
            )
        raise ForeignMountError(
            f"Removing the original is refused for {item.path}. The file is on {ANOTHER_DISK}. Copy only."
        )


def _remove_source(path: Path, kind: str) -> None:
    if kind == "dir" and path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _copy_part(dest: Path) -> Path:
    return dest.with_name(dest.name + COPY_PART_SUFFIX)


def _clear_copy_temp(dest: Path) -> None:
    """Drop a leftover temp. A killed copy must not be treated as the model."""
    part = _copy_part(dest)
    try:
        if part.is_dir() and not part.is_symlink():
            shutil.rmtree(part)
        elif part.exists() or part.is_symlink():
            part.unlink()
    except OSError:
        pass


def _copy_bytes_needed(size: int) -> int:
    margin = max(COPY_FREE_MARGIN_MIN, size // 100)
    return size + margin


def _require_copy_space(directory: Path, nbytes: int) -> None:
    need = _copy_bytes_needed(nbytes)
    try:
        st = os.statvfs(directory)
        free = int(st.f_bavail) * int(st.f_frsize)
    except OSError as exc:
        raise RuntimeError(
            "Could not copy this model. Free space could not be checked."
        ) from exc
    if free < need:
        raise RuntimeError(
            "Could not copy this model. The disk does not have enough free space. "
            f"The copy needs {human_bytes(need)} and {human_bytes(free)} is free."
        )


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _finish_store_file(path: Path, uid: int | None, gid: int | None) -> None:
    # The store file is 0644. The source mode is not kept.
    os.chmod(path, 0o644)
    if uid is not None and gid is not None and not path.is_symlink():
        os.chown(path, uid, gid)


def _open_part(part: Path):
    """Create the temp file without following a symlink or truncating a name that exists."""
    fd = os.open(
        part,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
    )
    try:
        os.fchmod(fd, 0o644)
    except OSError:
        os.close(fd)
        raise
    return os.fdopen(fd, "wb")


def _transfer_file(src: Path, part: Path) -> None:
    with open(src, "rb") as rf, _open_part(part) as wf:
        while True:
            chunk = rf.read(1024 * 1024)
            if not chunk:
                break
            wf.write(chunk)
        wf.flush()
        os.fsync(wf.fileno())


def _raise_copy_error(exc: Exception) -> None:
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        raise RuntimeError(
            "Could not copy this model. The disk filled up during the copy. "
            "The partial file was removed."
        ) from exc
    raise exc


def _copy_file_into_store(
    src: Path,
    dest: Path,
    uid: int | None,
    gid: int | None,
    algo: str,
    src_hash: str,
) -> None:
    # Temp sits next to the final name so the rename stays on one filesystem.
    # The final name appears only after the checksum matches.
    _require_copy_space(dest.parent, src.stat().st_size)
    part = _copy_part(dest)
    _clear_copy_temp(dest)
    try:
        _transfer_file(src, part)
        _finish_store_file(part, uid, gid)
        if hash_path(part, algo) != src_hash:
            raise RuntimeError(
                f"checksum mismatch copying {src}. The store file was not kept."
            )
        os.replace(part, dest)
        _fsync_dir(dest.parent)
    except Exception as exc:
        _clear_copy_temp(dest)
        _raise_copy_error(exc)


def _copy_tree_into_store(
    item: FoundWeight,
    dest: Path,
    uid: int | None,
    gid: int | None,
    algo: str,
    src_hash: str,
) -> None:
    _require_copy_space(dest.parent, _dir_size(item.path))
    part = _copy_part(dest)
    _clear_copy_temp(dest)
    try:
        # A foreign tree must land as real files. Links back to that mount are refused.
        shutil.copytree(item.path, part, symlinks=not foreign_source(item))
        for dirpath, _dirnames, filenames in os.walk(part):
            if uid is not None and gid is not None:
                os.chown(dirpath, uid, gid)
            for name in filenames:
                fp = Path(dirpath) / name
                if fp.is_symlink():
                    continue
                _finish_store_file(fp, uid, gid)
                with open(fp, "rb") as fh:
                    os.fsync(fh.fileno())
        if hash_path(part, algo) != src_hash:
            raise RuntimeError(
                f"checksum mismatch copying {item.path}. The store file was not kept."
            )
        os.replace(part, dest)
        _fsync_dir(dest.parent)
    except Exception as exc:
        _clear_copy_temp(dest)
        _raise_copy_error(exc)


def _copy_into_store(
    item: FoundWeight,
    dest: Path,
    uid: int | None,
    gid: int | None,
    *,
    algo: str,
    src_hash: str,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uid is not None and gid is not None:
        os.chown(dest.parent, uid, gid)
    if item.kind == "dir":
        _copy_tree_into_store(item, dest, uid, gid, algo, src_hash)
        return
    _copy_file_into_store(item.path, dest, uid, gid, algo, src_hash)


def organize(
    items: tuple[FoundWeight, ...],
    model_root: Path,
    *,
    mode: str = "copy",
    remove_source: bool = False,
    uid: int | None = None,
    gid: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    if mode not in {"copy", "move"}:
        raise ValueError("mode must be copy or move")
    _refuse_foreign_mutate(items, mode, remove_source)
    want_remove = mode == "move" or remove_source
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
        algo = integrity_algo_for(item)
        src_hash = hash_path(item.path, algo)
        try:
            _copy_into_store(item, dest, uid, gid, algo=algo, src_hash=src_hash)
        except RuntimeError as exc:
            if "checksum mismatch" in str(exc):
                log.append(
                    f"checksum mismatch after {mode} {item.path} -> {dest} ({algo})"
                )
                continue
            raise
        verb = "copied" if mode == "copy" else "moved"
        log.append(f"{verb} {item.path} -> {dest} {algo}={src_hash[:12]}")
        if want_remove:
            if not parent_writable:
                log.append(f"kept source {item.path} (not writable)")
            elif _under(item.path, store):
                log.append(f"kept source {item.path} (inside store)")
            else:
                _remove_source(item.path, item.kind)
                log.append(f"removed source {item.path} after checksum match")
    return log


def catalog_dest(model: CatalogWeight, model_root: Path) -> Path:
    return model_root / model.subdir / model.filename


def _catalog_skip_reason(path: Path, model: CatalogWeight) -> str | None:
    """Why this file cannot fill the catalog entry. None when it matches.

    Size is checked first. A published checksum is checked when the catalog has one.
    A short file left by a killed copy does not match.
    """
    if path.is_symlink() or not path.is_file():
        return f"Skipped {path}. It is not a regular file."
    try:
        size = path.stat().st_size
    except OSError:
        return f"Skipped {path}. It could not be read."
    if size <= 0:
        return f"Skipped {path}. It is empty."
    if model.bytes > 0 and size != model.bytes:
        return (
            f"Skipped {path}. It is {human_bytes(size)} and the catalog "
            f"wants {human_bytes(model.bytes)}."
        )
    published = model.published_hash()
    if published:
        try:
            digest = hash_file(path, published.algo)
        except (OSError, ValueError):
            return f"Skipped {path}. Its checksum could not be read."
        if digest != published.hexdigest.lower():
            return f"Skipped {path}. Its checksum does not match the catalog."
    if model.bytes <= 0 and published is None:
        return f"Skipped {path}. The catalog has no size or checksum to trust."
    return None


def _catalog_file_complete(path: Path, model: CatalogWeight) -> bool:
    """True when the file matches the catalog size and published checksum."""
    return _catalog_skip_reason(path, model) is None


def _acceptable_source(
    model: CatalogWeight,
    target: UserTarget,
    extra: tuple[Path, ...],
    dest: Path,
) -> tuple[FoundWeight | None, str]:
    """A found file that matches the catalog, plus why the first miss was skipped."""
    roots = scan_roots(target.home, target.model_root, extra)
    try:
        dest_real = dest.resolve() if dest.exists() or dest.is_symlink() else None
    except OSError:
        dest_real = None
    skipped = ""
    for item in scan(roots, target.model_root):
        if item.kind != "file" or item.path.name != model.filename:
            continue
        try:
            if dest_real is not None and item.path.resolve() == dest_real:
                continue
        except OSError:
            continue
        reason = _catalog_skip_reason(item.path, model)
        if reason is None:
            return item, ""
        if not skipped:
            skipped = reason
    return None, skipped


def verify_download(
    path: Path,
    model: CatalogWeight,
    *,
    expected: FileHash | None = None,
    expected_size: int = 0,
) -> None:
    """Reject an exit-0 download that is missing, empty, short, or the wrong hash.

    Hugging Face Xet and some CDNs close the body early and still report success.
    Catalog checksums are often empty, so size is the gate when no digest exists.
    """
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"download missing for {model.id}")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"empty download for {model.id}")
    want = expected or model.published_hash()
    if want:
        digest = hash_file(path, want.algo)
        if digest != want.hexdigest.lower():
            raise RuntimeError(f"{want.algo} mismatch for {model.id}")
        return
    want_size = expected_size or model.bytes
    if want_size > 0 and size != want_size:
        raise RuntimeError(
            f"incomplete download for {model.id} ({size} B of {want_size} B)"
        )


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
    _clear_copy_temp(dest)
    if (dest.exists() or dest.is_symlink()) and not force:
        if _catalog_file_complete(dest, model):
            return f"already {dest}"
    if dry_run:
        return f"download {model.id} -> {dest} ({human_bytes(model.bytes)})"
    # The final name stays until the temp file has been verified.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uid is not None and gid is not None:
        os.chown(dest.parent, uid, gid)
    part = _copy_part(dest)
    req = urllib.request.Request(model.url, headers={"User-Agent": UA})
    written = 0
    content_length = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_length = int(resp.headers.get("Content-Length") or 0)
            total = content_length or model.bytes
            with _open_part(part) as fh:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if on_progress:
                        on_progress(written, total)
                fh.flush()
                os.fsync(fh.fileno())
        want = expected or model.published_hash()
        published_size = 0
        if want is None:
            want, published_size = lookup_published_meta(model.url)
        # Prefer the size published on the download page. Catalog bytes next.
        # Content-Length last. A truncated Xet body can advertise its own short length.
        want_size = published_size or model.bytes or content_length
        verify_download(part, model, expected=want, expected_size=want_size)
        os.replace(part, dest)
        _fsync_dir(dest.parent)
    except Exception:
        _clear_copy_temp(dest)
        raise
    if uid is not None and gid is not None and not dest.is_symlink():
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
    _clear_copy_temp(dest)
    # The catalog is the check. A shorter file found on disk must not replace it.
    if _catalog_file_complete(dest, model):
        return f"already {dest}"
    source, skipped = _acceptable_source(model, target, extra, dest)
    # Copy or move places a real file in the store. A symlink is not offered.
    # Move removes the original and always asks, so Apply copies.
    if dry_run:
        if source is None:
            planned = download(
                model,
                target.model_root,
                uid=target.uid,
                gid=target.gid,
                dry_run=True,
            )
            if skipped:
                return f"{skipped} {planned}"
            return planned
        return f"copy {source.path} -> {dest}"
    if source is not None:
        algo = integrity_algo_for(source)
        src_hash = hash_path(source.path, algo)
        _copy_into_store(
            source, dest, target.uid, target.gid, algo=algo, src_hash=src_hash
        )
        return f"copied {source.path} -> {dest}"

    def prog(done: int, total: int) -> None:
        if on_progress:
            on_progress(
                f"Downloading {model.filename}: {human_bytes(done)} / {human_bytes(total)}"
            )

    try:
        got = download(
            model,
            target.model_root,
            on_progress=prog,
            uid=target.uid,
            gid=target.gid,
        )
    except Exception as exc:
        if skipped:
            raise RuntimeError(f"{skipped} {exc}") from exc
        raise
    if skipped:
        return f"{skipped} {got}"
    return got
