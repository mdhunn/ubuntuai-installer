"""Domain types. Probe, catalog, apply, and UI all share these."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


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

    def backends(self) -> frozenset[str]:
        found = {d.backend for d in self.devices}
        found.add("cpu")
        return frozenset(found)

    def hybrid_ok(self) -> bool:
        kinds = {d.kind for d in self.devices}
        return "npu" in kinds and bool(kinds & {"igpu", "dgpu"})

    def primary_backend(self) -> str:
        order = ("cuda", "rocm", "xdna", "vulkan", "cpu")
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

    def offered(self, hw: Hardware) -> bool:
        if not self.hide_unless_backend:
            return True
        return bool(set(self.hide_unless_backend) & set(hw.backends()))

    def satisfied(self, hw: Hardware) -> bool:
        if not self.needs_any_backend:
            return True
        return bool(set(self.needs_any_backend) & set(hw.backends()))

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
