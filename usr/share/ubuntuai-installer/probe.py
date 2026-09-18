"""Hardware probe. Reads lspci, sysfs, apt-cache policy, and a few binaries. Never installs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from domain import Device, Hardware, strix_halo_name

NPU_FIRMWARE_LINK = Path("/lib/firmware/amdnpu/17f0_11/npu.sbin.zst")
NPU_FIRMWARE_LINK_PLAIN = Path("/lib/firmware/amdnpu/17f0_11/npu.sbin")
NPU_ACCEL_NODE = Path("/dev/accel/accel0")
_FW_VERSION_RE = re.compile(r"(?<![0-9])(\d+\.\d+\.\d+\.\d+)(?![0-9])")

NVIDIA = "10de"
AMD_GPU = "1002"
AMD_SYS = "1022"
NPU_DEV = "17f0"
APT_NAME = re.compile(r"^[a-zA-Z0-9.+-]+$")
_SOURCES_LIST = Path("/etc/apt/sources.list")
_SOURCES_DIR = Path("/etc/apt/sources.list.d")


def _run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return p.stdout


def _cpu_name(cpuinfo: str | None = None) -> str:
    text = cpuinfo if cpuinfo is not None else Path("/proc/cpuinfo").read_text()
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown CPU"


def _ram_bytes(meminfo: str | None = None) -> int:
    text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return kb * 1024
    return 0


def _lspci(text: str | None = None) -> str:
    if text is not None:
        return text
    return _run(["lspci", "-nn"])


def _rocminfo_present(injected: bool | None) -> bool:
    if injected is not None:
        return bool(injected)
    return shutil.which("rocminfo") is not None


def _gfx_is_strix_halo(gfx: str | None) -> bool:
    if not gfx:
        return False
    return gfx.lower().replace(" ", "").startswith("gfx115")


def _parse_lspci(
    text: str,
    *,
    npu_firmware_link: Path | None = None,
    npu_accel_node: Path | None = None,
    rocminfo: bool = False,
    gfx: str | None = None,
) -> list[Device]:
    devices: list[Device] = []
    gfx_strix = _gfx_is_strix_halo(gfx)
    for line in text.splitlines():
        lower = line.lower()
        if f"[{AMD_SYS}:{NPU_DEV}]" in lower or "neural processing unit" in lower:
            accel = npu_accel_node if npu_accel_node is not None else NPU_ACCEL_NODE
            devices.append(
                Device(
                    kind="npu",
                    vendor="amd",
                    name=_strip_pci(line),
                    node=str(accel) if accel.exists() else None,
                    backend="xdna",
                    detail=inspect_npu_firmware(firmware_link=npu_firmware_link).detail,
                )
            )
            continue
        if f"[{NVIDIA}:" in line and _is_display(lower):
            cuda = shutil.which("nvidia-smi") is not None
            devices.append(
                Device(
                    kind="dgpu",
                    vendor="nvidia",
                    name=_strip_pci(line),
                    node=_first_existing(("/dev/nvidia0", "/dev/dri/renderD128")),
                    backend="cuda" if cuda else "vulkan",
                    detail="nvidia-smi present" if cuda else "CUDA runtime missing",
                )
            )
            continue
        if f"[{AMD_GPU}:" in line and _is_display(lower):
            name = _strip_pci(line)
            node = _first_existing(("/dev/dri/renderD128", "/dev/kfd"))
            vulkan_first = strix_halo_name(lower) or gfx_strix
            kind = "igpu" if vulkan_first else "dgpu"
            if vulkan_first:
                backend = "vulkan"
                detail = "Vulkan (Mesa)"
            elif rocminfo:
                backend = "rocm"
                detail = "ROCm"
            else:
                backend = "vulkan"
                detail = "Vulkan (Mesa)"
            devices.append(
                Device(
                    kind=kind,
                    vendor="amd",
                    name=name,
                    node=node,
                    backend=backend,
                    detail=detail,
                )
            )
            continue
        if "intel" in lower and _is_display(lower) and "amd" not in lower:
            devices.append(
                Device(
                    kind="igpu",
                    vendor="intel",
                    name=_strip_pci(line),
                    node=_first_existing(("/dev/dri/renderD128",)),
                    backend="vulkan",
                    detail="Intel display",
                )
            )
    return devices


def _is_display(lower: str) -> bool:
    return any(tok in lower for tok in ("vga", "display", "3d controller"))


def _strip_pci(line: str) -> str:
    if ": " in line:
        return line.split(": ", 1)[1].strip()
    return line.strip()


def _first_existing(paths: tuple[str, ...]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


@dataclass(frozen=True)
class NpuFirmwarePair:
    link: Path
    state: str
    active_name: str
    active_version: str
    disk_name: str
    disk_version: str
    detail: str


def _fmt_version(parts: tuple[int, ...] | None) -> str:
    if not parts:
        return ""
    return ".".join(str(p) for p in parts)


def _version_in(text: str) -> tuple[int, ...] | None:
    match = _FW_VERSION_RE.search(text)
    if not match:
        return None
    return tuple(int(p) for p in match.group(1).split("."))


def _blob_version(path: Path) -> tuple[int, ...] | None:
    for text in (path.name, os.path.realpath(path), str(path)):
        found = _version_in(text)
        if found:
            return found
    return None


def _is_npu_blob_name(name: str) -> bool:
    lower = name.lower()
    return "npu" in lower and "sbin" in lower


def _is_npu7_name(name: str) -> bool:
    stem = name[:-4] if name.endswith(".zst") else name
    return stem == "npu_7.sbin"


def _default_firmware_link() -> Path:
    if NPU_FIRMWARE_LINK.exists() or NPU_FIRMWARE_LINK.is_symlink():
        return NPU_FIRMWARE_LINK
    if NPU_FIRMWARE_LINK_PLAIN.exists() or NPU_FIRMWARE_LINK_PLAIN.is_symlink():
        return NPU_FIRMWARE_LINK_PLAIN
    return NPU_FIRMWARE_LINK


def _firmware_dirs(link: Path) -> tuple[Path, ...]:
    dirs: list[Path] = []
    if link.parent.is_dir():
        dirs.append(link.parent)
    try:
        real_parent = Path(os.path.realpath(link)).parent
    except OSError:
        real_parent = None
    if real_parent is not None and real_parent.is_dir() and real_parent not in dirs:
        dirs.append(real_parent)
    return tuple(dirs)


def _sibling_blobs(link: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    seen: set[Path] = set()
    for folder in _firmware_dirs(link):
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for item in entries:
            try:
                if not (item.is_file() or item.is_symlink()):
                    continue
            except OSError:
                continue
            if not _is_npu_blob_name(item.name):
                continue
            if item in seen:
                continue
            seen.add(item)
            found.append(item)
    return tuple(found)


def _newer_sibling(
    active: Path, siblings: tuple[Path, ...]
) -> tuple[Path | None, tuple[int, ...] | None]:
    # Active npu.sbin can stay on the legacy blob while npu_7.sbin or a
    # newer npu.sbin.<version> sits next to it. Read only. Never write.
    try:
        active_real = Path(os.path.realpath(active))
    except OSError:
        active_real = active
    active_ver = _blob_version(active)
    best_path: Path | None = None
    best_ver = active_ver
    npu7: Path | None = None
    for item in siblings:
        try:
            real = Path(os.path.realpath(item))
        except OSError:
            real = item
        if real == active_real:
            continue
        ver = _blob_version(item)
        if ver is not None and (best_ver is None or ver > best_ver):
            best_ver = ver
            best_path = item
        if _is_npu7_name(item.name):
            npu7 = item
    if best_path is not None and (
        active_ver is None or (best_ver is not None and best_ver > active_ver)
    ):
        return best_path, best_ver
    if npu7 is not None and best_path is None:
        npu7_ver = _blob_version(npu7)
        if npu7_ver is None or active_ver is None or npu7_ver > active_ver:
            return npu7, npu7_ver
    return None, None


def inspect_npu_firmware(*, firmware_link: Path | None = None) -> NpuFirmwarePair:
    link = firmware_link if firmware_link is not None else _default_firmware_link()
    present = link.exists() or link.is_symlink()
    if not present:
        return NpuFirmwarePair(
            link=link,
            state="missing",
            active_name="",
            active_version="",
            disk_name="",
            disk_version="",
            detail="firmware path missing",
        )
    try:
        target = Path(os.path.realpath(link))
    except OSError:
        target = link
    active_name = target.name
    active_ver = _blob_version(link)
    siblings = _sibling_blobs(link)
    newer, newer_ver = _newer_sibling(link, siblings)
    if newer is not None:
        disk_name = _pair_disk_name(newer, siblings)
        disk_version = _fmt_version(newer_ver)
        extra = disk_version or disk_name
        detail = (
            f"npu.sbin -> {active_name}; mismatched pair; "
            f"{extra} on disk as {disk_name}"
        )
        return NpuFirmwarePair(
            link=link,
            state="mismatch",
            active_name=active_name,
            active_version=_fmt_version(active_ver),
            disk_name=disk_name,
            disk_version=disk_version,
            detail=detail,
        )
    detail = f"npu.sbin -> {active_name}; matched pair"
    return NpuFirmwarePair(
        link=link,
        state="matched",
        active_name=active_name,
        active_version=_fmt_version(active_ver),
        disk_name="",
        disk_version="",
        detail=detail,
    )


def _pair_disk_name(newer: Path, siblings: tuple[Path, ...]) -> str:
    if _is_npu7_name(newer.name):
        return newer.name
    try:
        newer_real = Path(os.path.realpath(newer))
    except OSError:
        return newer.name
    for item in siblings:
        if not _is_npu7_name(item.name):
            continue
        try:
            if Path(os.path.realpath(item)) == newer_real:
                return item.name
        except OSError:
            continue
    return newer.name


def _notes(
    devices: list[Device],
    *,
    rocminfo: bool = False,
    gfx: str | None = None,
) -> tuple[str, ...]:
    notes: list[str] = []
    gfx_strix = _gfx_is_strix_halo(gfx)
    for d in devices:
        vulkan_first = (
            d.vendor == "amd"
            and d.kind in {"igpu", "dgpu"}
            and (strix_halo_name(d.name) or gfx_strix)
        )
        if vulkan_first and d.backend == "vulkan":
            notes.append(
                "Chat on this AMD iGPU uses Vulkan. "
                "ROCm is not the chat default. "
                "llama.cpp HIP still misses gfx1150. "
                "libgomp on this silicon is a known pain."
            )
        elif d.kind in {"igpu", "dgpu"} and d.vendor == "amd" and d.backend == "vulkan":
            if Path("/dev/kfd").exists() and not rocminfo:
                notes.append(
                    "/dev/kfd is present. rocminfo is not. GPU inference uses Vulkan."
                )
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for n in notes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def _ensure_cpu_device(cpu_name: str, devices: list[Device]) -> list[Device]:
    if any(d.kind == "cpu" for d in devices):
        return devices
    devices.append(
        Device(
            kind="cpu",
            vendor="cpu",
            name=cpu_name,
            node=None,
            backend="cpu",
            detail="",
        )
    )
    return devices


def apt_cache_policy(names: tuple[str, ...], text: str | None = None) -> str:
    """Read-only apt-cache policy. Never runs apt-get or update."""
    if text is not None:
        return text
    safe = tuple(n for n in names if APT_NAME.match(n))
    if not safe:
        return ""
    return _run(["apt-cache", "policy", "--", *safe])


def split_apt_policies(text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.endswith(":"):
            name = line[:-1].strip()
            if APT_NAME.match(name):
                current = name
                blocks[current] = [line]
                continue
        if current is not None:
            blocks[current].append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def apt_candidate(policy_text: str) -> str:
    for line in policy_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("candidate:"):
            value = stripped.split(":", 1)[1].strip()
            if not value or value == "(none)":
                return ""
            return value
    return ""


def apt_sources_text(text: str | None = None) -> str:
    if text is not None:
        return text
    parts: list[str] = []
    if _SOURCES_LIST.is_file():
        parts.append(_SOURCES_LIST.read_text(encoding="utf-8", errors="replace"))
    if _SOURCES_DIR.is_dir():
        for path in sorted(_SOURCES_DIR.iterdir()):
            if path.suffix in {".list", ".sources"} and path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def universe_in_sources(text: str) -> bool:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("deb") and "universe" in lower:
            return True
        if lower.startswith("components:") and "universe" in lower:
            return True
    return False


def probe(
    *,
    lspci_text: str | None = None,
    cpuinfo: str | None = None,
    meminfo: str | None = None,
    npu_firmware_link: Path | None = None,
    npu_accel_node: Path | None = None,
    rocminfo: bool | None = None,
    gfx: str | None = None,
) -> Hardware:
    cpu = _cpu_name(cpuinfo)
    ram = _ram_bytes(meminfo)
    have_rocminfo = _rocminfo_present(rocminfo)
    devices = _parse_lspci(
        _lspci(lspci_text),
        npu_firmware_link=npu_firmware_link,
        npu_accel_node=npu_accel_node,
        rocminfo=have_rocminfo,
        gfx=gfx,
    )
    devices = _ensure_cpu_device(cpu, devices)
    return Hardware(
        cpu_name=cpu,
        ram_bytes=ram,
        devices=tuple(devices),
        notes=_notes(devices, rocminfo=have_rocminfo, gfx=gfx),
        gfx=gfx or "",
    )
