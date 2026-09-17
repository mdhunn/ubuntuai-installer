"""Hardware probe. Reads lspci, sysfs, and a few binaries. Never installs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from domain import Device, Hardware

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


def _parse_lspci(text: str) -> list[Device]:
    devices: list[Device] = []
    for line in text.splitlines():
        lower = line.lower()
        if f"[{AMD_SYS}:{NPU_DEV}]" in lower or "neural processing unit" in lower:
            devices.append(
                Device(
                    kind="npu",
                    vendor="amd",
                    name=_strip_pci(line),
                    node=_first_existing(("/dev/accel/accel0",)),
                    backend="xdna",
                    detail=_npu_firmware_detail(),
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


def _npu_firmware_detail() -> str:
    link = Path("/lib/firmware/amdnpu/17f0_11/npu.sbin.zst")
    if not link.exists():
        accel = Path("/dev/accel/accel0")
        return "device present" if accel.exists() else "firmware path missing"
    target = os.path.realpath(link)
    name = Path(target).name
    return f"npu.sbin -> {name}"


def _notes(devices: list[Device]) -> tuple[str, ...]:
    notes: list[str] = []
    for d in devices:
        if d.backend == "xdna" and "1.0.0.166" in d.detail:
            notes.append(
                "NPU firmware symlink still points at 1.0.0.166. "
                "FastFlowLM wants >= 1.1.0.0. 1.1.2.65 is on disk as npu_7.sbin."
            )
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
) -> Hardware:
    cpu = _cpu_name(cpuinfo)
    ram = _ram_bytes(meminfo)
    devices = _parse_lspci(_lspci(lspci_text))
    devices = _ensure_cpu_device(cpu, devices)
    return Hardware(
        cpu_name=cpu,
        ram_bytes=ram,
        devices=tuple(devices),
        notes=_notes(devices),
    )
