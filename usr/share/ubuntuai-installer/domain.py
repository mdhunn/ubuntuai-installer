"""Domain types. Probe, catalog, apply, and UI all share these."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Strix Halo / gfx115x iGPU. Chat stays on Vulkan even when rocminfo exists.
# llama.cpp HIP still misses gfx1150. libgomp on this silicon is a known pain.
_STRIX_HALO_MARKERS = ("strix", "8060s", "8050s", "gfx115", "radeon 80")


def strix_halo_name(text: str) -> bool:
    lower = (text or "").lower()
    return any(tok in lower for tok in _STRIX_HALO_MARKERS)


@dataclass(frozen=True)
class Device:
    kind: str
    vendor: str
    name: str
    node: str | None
    backend: str
    detail: str = ""


@dataclass(frozen=True)
class Hardware:
    cpu_name: str
    ram_bytes: int
    devices: tuple[Device, ...]
    notes: tuple[str, ...] = ()
    gfx: str = ""

    def backends(self) -> frozenset[str]:
        found = {d.backend for d in self.devices}
        found.add("cpu")
        return frozenset(found)

    def installable_backends(self) -> frozenset[str]:
        found = set(self.backends())
        for d in self.devices:
            if d.kind not in {"igpu", "dgpu"}:
                continue
            if d.vendor == "amd":
                found.add("rocm")
            elif d.vendor == "nvidia":
                found.add("cuda")
        return frozenset(found)

    def hybrid_ok(self) -> bool:
        kinds = {d.kind for d in self.devices}
        return "npu" in kinds and bool(kinds & {"igpu", "dgpu"})

    def strix_halo_class(self) -> bool:
        if strix_halo_name(self.cpu_name) or strix_halo_name(self.gfx):
            return True
        return any(
            d.vendor == "amd"
            and d.kind in {"igpu", "dgpu"}
            and strix_halo_name(d.name)
            for d in self.devices
        )

    def primary_backend(self) -> str:
        # Chat on Strix Halo-class iGPU stays Vulkan. Do not rank ROCm first.
        order = ("cuda", "rocm", "xdna", "vulkan", "cpu")
        if self.strix_halo_class():
            order = ("cuda", "xdna", "vulkan", "rocm", "cpu")
        have = self.backends()
        for b in order:
            if b in have:
                return b
        return "cpu"


@dataclass(frozen=True)
class Workflow:
    id: str
    title: str
    summary: str
    default: bool
    always: bool
    requires: tuple[str, ...]
    apt: tuple[str, ...]
    groups: tuple[str, ...]
    model_subdirs: tuple[str, ...]
    needs_any_backend: tuple[str, ...]
    hide_unless_backend: tuple[str, ...]
    ports: tuple[int, ...]
    notes: str
    role: str = ""
    priority: int = 0
    min_ram_bytes: int = 0
    runtime_bins: tuple[str, ...] = ()
    vendor: str = ""
    required_weights: tuple[str, ...] = ()
    apt_for_backend: tuple[tuple[str, tuple[str, ...]], ...] = ()
    helpers_only: bool = False

    def offered(self, hw: Hardware) -> bool:
        if not self.hide_unless_backend:
            return True
        return bool(set(self.hide_unless_backend) & set(hw.installable_backends()))

    def satisfied(self, hw: Hardware) -> bool:
        if not self.needs_any_backend:
            return True
        return bool(set(self.needs_any_backend) & set(hw.installable_backends()))

    def ready(self, hw: Hardware) -> bool:
        if not self.needs_any_backend:
            return True
        return bool(set(self.needs_any_backend) & set(hw.backends()))

    def packages_for(self, hw: Hardware) -> tuple[str, ...]:
        pkgs: list[str] = list(self.apt)
        installable = hw.installable_backends()
        for backend, extra in self.apt_for_backend:
            if backend in installable:
                pkgs.extend(extra)
        return tuple(dict.fromkeys(pkgs))

    def ram_ok(self, hw: Hardware) -> bool:
        return hw.ram_bytes >= self.min_ram_bytes

    def runtime_ok(self) -> bool:
        if not self.runtime_bins:
            return True
        return any(shutil.which(b) for b in self.runtime_bins)

    def eligible(self, hw: Hardware) -> bool:
        return self.offered(hw) and self.satisfied(hw) and self.ram_ok(hw)


@dataclass(frozen=True)
class Action:
    kind: str
    summary: str
    payload: tuple[str, ...] = ()


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class FoundWeight:
    path: Path
    size: int
    subdir: str
    dest_name: str
    state: str
    fmt: str = ""
    kind: str = "file"
    foreign: bool = False


@dataclass(frozen=True)
class FileHash:
    algo: str
    hexdigest: str

    def label(self) -> str:
        return f"{self.algo}:{self.hexdigest}"


@dataclass(frozen=True)
class CatalogWeight:
    id: str
    title: str
    summary: str
    subdir: str
    filename: str
    url: str
    bytes: int
    workflows: tuple[str, ...]
    hash_algo: str = ""
    hash_hex: str = ""
    default: bool = False
    fmt: str = ""

    def published_hash(self) -> FileHash | None:
        if not self.hash_hex:
            return None
        return FileHash(self.hash_algo, self.hash_hex.lower())


@dataclass(frozen=True)
class UserTarget:
    name: str
    uid: int
    gid: int
    home: Path
    model_root: Path
    extra_model_paths: tuple[Path, ...] = ()
    bind: str = "127.0.0.1"
    selected: tuple[str, ...] = field(default_factory=tuple)
