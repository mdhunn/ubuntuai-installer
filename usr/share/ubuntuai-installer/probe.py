"""Hardware probe. Reads lspci, sysfs, and a few binaries. Never installs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from domain import Device, Hardware

NPU_FIRMWARE_LINK = Path("/lib/firmware/amdnpu/17f0_11/npu.sbin.zst")
NPU_FIRMWARE_LINK_PLAIN = Path("/lib/firmware/amdnpu/17f0_11/npu.sbin")
NPU_ACCEL_NODE = Path("/dev/accel/accel0")
_FW_VERSION_RE = re.compile(r"(?<![0-9])(\d+\.\d+\.\d+\.\d+)(?![0-9])")

NVIDIA = "10de"
AMD_GPU = "1002"
AMD_SYS = "1022"
NPU_DEV = "17f0"


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


def _parse_lspci(
    text: str,
    *,
    npu_firmware_link: Path | None = None,
    npu_accel_node: Path | None = None,
) -> list[Device]:
    devices: list[Device] = []
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
            rocm = shutil.which("rocminfo") is not None
            node = _first_existing(("/dev/dri/renderD128", "/dev/kfd"))
            kind = "igpu" if "strix" in lower or "radeon 80" in lower else "dgpu"
            if "strix halo" in lower or "8060s" in lower or "8050s" in lower:
                kind = "igpu"
            devices.append(
                Device(
                    kind=kind,
                    vendor="amd",
                    name=_strip_pci(line),
                    node=node,
                    backend="rocm" if rocm else "vulkan",
                    detail="ROCm" if rocm else "Vulkan (Mesa)",
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


def _notes(devices: list[Device]) -> tuple[str, ...]:
    notes: list[str] = []
    for d in devices:
        if d.kind in {"igpu", "dgpu"} and d.vendor == "amd" and d.backend == "vulkan":
            if Path("/dev/kfd").exists():
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


def probe(
    *,
    lspci_text: str | None = None,
    cpuinfo: str | None = None,
    meminfo: str | None = None,
    npu_firmware_link: Path | None = None,
    npu_accel_node: Path | None = None,
) -> Hardware:
    cpu = _cpu_name(cpuinfo)
    ram = _ram_bytes(meminfo)
    devices = _parse_lspci(
        _lspci(lspci_text),
        npu_firmware_link=npu_firmware_link,
        npu_accel_node=npu_accel_node,
    )
    devices = _ensure_cpu_device(cpu, devices)
    return Hardware(
        cpu_name=cpu,
        ram_bytes=ram,
        devices=tuple(devices),
        notes=_notes(devices),
    )
