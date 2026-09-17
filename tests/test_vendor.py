from __future__ import annotations

import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from support import PKG

from domain import Device, Hardware, UserTarget
from vendor import install_vendor, pick_archive
from weights import classify, ensure_weight, load_catalog


def _hw_vulkan() -> Hardware:
    return Hardware(
        cpu_name="cpu",
        ram_bytes=16 * 1024**3,
        devices=(
            Device("igpu", "amd", "radeon", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


class VendorTests(unittest.TestCase):
    def test_picks_vulkan_not_rocm_without_rocm(self) -> None:
        spec = {
            "archives": {
                "vulkan": {"url": "vk", "filename": "vk.tar.gz", "bytes": 1},
                "rocm": {"url": "rk", "filename": "rk.tar.gz", "bytes": 1},
            }
        }
        picked = pick_archive(spec, _hw_vulkan())
        self.assertEqual(picked["filename"], "vk.tar.gz")

    def test_extracts_and_wraps(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            tarball = home / "openmoss.tar.gz"
            with tarfile.open(tarball, "w:gz") as tar:
                payload = home / "moss-tts-server"
                payload.write_bytes(b"#!/bin/sh\necho ok\n")
                tar.add(payload, arcname="moss-tts-server")
            vendors = {
                "openmoss": {
                    "binaries": ["moss-tts-server"],
                    "wrapper": "moss-tts-server",
                    "launcher": "ubuntuai-openmoss",
                    "port": 8081,
                    "archives": {
                        "vulkan": {
                            "url": "file://unused",
                            "filename": "openmoss.tar.gz",
                            "bytes": tarball.stat().st_size,
                        }
                    },
                }
            }
            cache = home / ".cache" / "ubuntuai" / "vendor"
            cache.mkdir(parents=True)
            (cache / "openmoss.tar.gz").write_bytes(tarball.read_bytes())
            target = UserTarget(
                name="u",
                uid=os_getuid(),
                gid=os_getgid(),
                home=home,
                model_root=home / "Models",
            )
            (home / "Models").mkdir()
            msg = install_vendor(
                "openmoss", _hw_vulkan(), target, vendors=vendors
            )
            self.assertIn("installed", msg)
            wrap = home / ".local" / "bin" / "moss-tts-server"
            self.assertTrue(wrap.is_file())
            self.assertIn("LD_LIBRARY_PATH", wrap.read_text())
            launcher = home / ".local" / "bin" / "ubuntuai-openmoss"
            self.assertTrue(launcher.is_file())
            again = install_vendor(
                "openmoss", _hw_vulkan(), target, vendors=vendors
            )
            self.assertTrue(again.startswith("already"))


def os_getuid() -> int:
    import os

    return os.getuid()


def os_getgid() -> int:
    import os

    return os.getgid()


class EnsureWeightTests(unittest.TestCase):
    def test_links_existing_filename(self) -> None:
        model = next(
            w for w in load_catalog(PKG / "weights.json") if w.id == "whisper-base-en"
        )
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "stash"
            extra.mkdir()
            blob = extra / model.filename
            blob.write_bytes(b"g" * (128 * 1024))
            store = home / "Models"
            store.mkdir()
            target = UserTarget(
                name="u",
                uid=os_getuid(),
                gid=os_getgid(),
                home=home,
                model_root=store,
            )
            msg = ensure_weight(model, target, (extra,))
            dest = store / "whisper" / model.filename
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(), blob.resolve())
            self.assertTrue(msg.startswith("linked"))

    def test_ggml_bin_is_whisper(self) -> None:
        self.assertEqual(classify(Path("/x/ggml-base.en.bin")), "whisper")
