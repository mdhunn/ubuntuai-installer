from __future__ import annotations

import errno
import hashlib
import http.server
import io
import os
import pwd
import shutil
import stat
import threading
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG

from domain import CatalogWeight, FileHash, UserTarget
from main import installer_main
from weights import (
    ANOTHER_DISK,
    FOREIGN_FSTYPES,
    UBUNTU_DISK,
    ForeignMountError,
    _hash_from_hf_row,
    _size_from_hf_row,
    classify,
    detect_bundle,
    disk_words,
    _copy_bytes_needed,
    _open_part,
    download,
    ensure_weight,
    foreign_source,
    move_off,
    hash_path,
    human_bytes,
    is_foreign_mount,
    load_catalog,
    normalize_scan_folder,
    organize,
    parse_named_hash,
    scan,
    scan_roots,
    verify_download,
)


class ClassifyTests(unittest.TestCase):
    def test_gguf_and_mmproj(self) -> None:
        self.assertEqual(classify(Path("/x/model.Q4_K_M.gguf")), "gguf")
        self.assertEqual(classify(Path("/x/mmproj-foo-BF16.gguf")), "mmproj")
        self.assertEqual(classify(Path("/x/foo.mmproj-f16.gguf")), "mmproj")

    def test_folder_hints(self) -> None:
        self.assertEqual(classify(Path("/models/loras/style.safetensors")), "loras")
        self.assertEqual(classify(Path("/models/vae/ae.safetensors")), "vae")
        self.assertEqual(classify(Path("/clip/clip_l.safetensors")), "clip")
        self.assertEqual(classify(Path("/q4nx_files/ornith/model.q4nx")), "flm")

    def test_skip_placeholders(self) -> None:
        self.assertIsNone(classify(Path("/x/put_checkpoints_here")))
        self.assertIsNone(classify(Path("/x/readme.txt")))
        self.assertIsNone(classify(Path("/x/model.safetensors.index.json")))

    def test_common_formats(self) -> None:
        self.assertEqual(classify(Path("/x/model.safetensors")), "safetensors")
        self.assertEqual(classify(Path("/x/unet.onnx")), "onnx")
        self.assertEqual(classify(Path("/x/sd15.ckpt")), "pytorch")
        self.assertEqual(classify(Path("/x/weights.pth")), "pytorch")
        self.assertEqual(classify(Path("/controlnet/canny.safetensors")), "controlnet")

    def test_detect_hf_bundle(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "Qwen3-4B"
            repo.mkdir()
            (repo / "config.json").write_text("{}", encoding="utf-8")
            (repo / "model.safetensors").write_bytes(b"s" * (128 * 1024))
            self.assertEqual(detect_bundle(repo), ("hf", "safetensors"))
            store = Path(tmp) / "Models"
            store.mkdir()
            found = scan((repo.parent,), store)
            dirs = [f for f in found if f.kind == "dir"]
            files = [f for f in found if f.kind == "file"]
            self.assertEqual(len(dirs), 1)
            self.assertEqual(dirs[0].subdir, "hf")
            self.assertEqual(dirs[0].fmt, "safetensors")
            self.assertFalse(any(f.path.name == "model.safetensors" for f in files))
            log = organize(tuple(dirs), store)
            dest = store / "hf" / "Qwen3-4B"
            self.assertTrue(dest.is_dir())
            self.assertFalse(dest.is_symlink())
            self.assertTrue((dest / "config.json").is_file())
            self.assertTrue((dest / "model.safetensors").is_file())
            self.assertTrue((repo / "config.json").is_file())
            self.assertTrue(any("copied" in line for line in log))

    def test_detect_diffusers_bundle(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "sdxl"
            repo.mkdir()
            (repo / "model_index.json").write_text("{}", encoding="utf-8")
            unet = repo / "unet"
            unet.mkdir()
            (unet / "diffusion_pytorch_model.safetensors").write_bytes(b"s" * (128 * 1024))
            self.assertEqual(detect_bundle(repo), ("diffusers", "diffusers"))


class ScanOrganizeTests(unittest.TestCase):
    def test_scan_and_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            src_dir = home / "AI models" / "Dolphin"
            src_dir.mkdir(parents=True)
            blob = src_dir / "Dolphin.Q8_0.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            mm = src_dir / "mmproj-Dolphin.gguf"
            mm.write_bytes(b"m" * (128 * 1024))
            store = home / "Models"
            store.mkdir()
            found = scan((src_dir.parent,), store)
            kinds = {f.subdir for f in found}
            self.assertIn("gguf", kinds)
            self.assertIn("mmproj", kinds)
            self.assertTrue(all(f.state == "new" for f in found))
            log = organize(found, store)
            dest = store / "gguf" / "Dolphin.Q8_0.gguf"
            self.assertTrue(dest.is_file())
            self.assertFalse(dest.is_symlink())
            self.assertNotEqual(dest.resolve(), blob.resolve())
            self.assertEqual(dest.read_bytes(), blob.read_bytes())
            self.assertTrue(blob.exists())
            self.assertTrue(any("copied" in line for line in log))
            again = scan((src_dir.parent, store), store)
            self.assertTrue(any(f.state == "already" for f in again))

    def test_copy_and_remove_after_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            src_dir = home / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "copy-me.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = home / "Models"
            found = scan((src_dir,), store)
            log = organize(found, store, mode="copy", remove_source=True)
            dest = store / "gguf" / "copy-me.gguf"
            self.assertTrue(dest.is_file())
            self.assertFalse(blob.exists())
            self.assertTrue(any("removed source" in line for line in log))
            self.assertTrue(any("copied" in line for line in log))

    def test_copy_keeps_source_without_remove(self) -> None:
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "keep-me.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = Path(tmp) / "Models"
            found = scan((src_dir,), store)
            organize(found, store, mode="copy", remove_source=False)
            self.assertTrue((store / "gguf" / "keep-me.gguf").is_file())
            self.assertTrue(blob.exists())

    def test_move_when_writable(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            src_dir = home / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "tiny.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = home / "Models"
            found = scan((src_dir,), store)
            log = organize(found, store, mode="move")
            dest = store / "gguf" / "tiny.gguf"
            self.assertTrue(dest.is_file())
            self.assertFalse(blob.exists())
            self.assertTrue(any("removed source" in line for line in log))
            self.assertTrue(any("moved" in line for line in log))

    def test_rejects_link_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "keep-link.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = Path(tmp) / "Models"
            found = scan((src_dir,), store)
            with self.assertRaises(ValueError) as ctx:
                organize(found, store, mode="link")
            self.assertIn("copy or move", str(ctx.exception))
            dest = store / "gguf" / "keep-link.gguf"
            self.assertFalse(dest.exists())
            self.assertTrue(blob.exists())

    def test_move_not_writable_does_not_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "keep-move.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = Path(tmp) / "Models"
            found = scan((src_dir,), store)
            with patch("weights.os.access", return_value=False):
                log = organize(found, store, mode="move")
            dest = store / "gguf" / "keep-move.gguf"
            self.assertTrue(dest.is_file())
            self.assertFalse(dest.is_symlink())
            self.assertTrue(blob.exists())
            self.assertTrue(any("moved" in line for line in log))
            self.assertTrue(any("kept source" in line for line in log))

    def test_copy_hash_mismatch_keeps_source(self) -> None:
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "bad-hash.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = Path(tmp) / "Models"
            found = scan((src_dir,), store)
            real = hash_path
            n = {"i": 0}

            def fake(path: Path, algo: str = "sha256") -> str:
                n["i"] += 1
                digest = real(path, algo)
                if n["i"] == 2:
                    return "0" * 64
                return digest

            with patch("weights.hash_path", fake):
                log = organize(found, store, mode="copy", remove_source=True)
            dest = store / "gguf" / "bad-hash.gguf"
            self.assertTrue(blob.exists())
            self.assertFalse(dest.exists())
            self.assertTrue(any("checksum mismatch" in line for line in log))

    def test_extra_scan_folder(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "elsewhere"
            extra.mkdir()
            blob = extra / "custom.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = home / "Models"
            store.mkdir()
            roots = scan_roots(home, store, (extra,))
            self.assertIn(extra.resolve(), roots)
            found = scan(roots, store)
            self.assertTrue(any(f.path == blob for f in found))

    def test_normalize_rejects_root(self) -> None:
        with self.assertRaises(ValueError):
            normalize_scan_folder("/", Path("/tmp"))

    def test_collision_renames(self) -> None:
        with TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            a.mkdir()
            b.mkdir()
            (a / "model.q4nx").write_bytes(b"q" * (128 * 1024))
            (b / "model.q4nx").write_bytes(b"r" * (128 * 1024))
            store = Path(tmp) / "Models"
            found = scan((a, b), store)
            names = {f.dest_name for f in found}
            self.assertIn("model.q4nx", names)
            self.assertTrue(any(n.endswith("model.q4nx") and n != "model.q4nx" for n in names))


class HashParseTests(unittest.TestCase):
    def test_parse_named_hash_schemes(self) -> None:
        sha = parse_named_hash("oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.assertIsNotNone(sha)
        assert sha is not None
        self.assertEqual(sha.algo, "sha256")
        md = parse_named_hash("MD5: 00112233445566778899aabbccddeeff")
        self.assertIsNotNone(md)
        assert md is not None
        self.assertEqual(md.algo, "md5")
        sha1 = parse_named_hash("SHA-1=0123456789abcdef0123456789abcdef01234567")
        self.assertIsNotNone(sha1)
        assert sha1 is not None
        self.assertEqual(sha1.algo, "sha1")
        self.assertIsNone(parse_named_hash("not a hash"))

    def test_hf_lfs_oid_without_prefix(self) -> None:
        row = {
            "path": "ggml-base.en.bin",
            "oid": "87c664c563ef3ff52424dd4fa925cf95b306dba6",
            "lfs": {
                "oid": "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
                "size": 147964211,
            },
        }
        found = _hash_from_hf_row(row, "ggml-base.en.bin")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.algo, "sha256")
        self.assertTrue(found.hexdigest.startswith("a03779c8"))
        git_only = {"path": "readme.md", "oid": "87c664c563ef3ff52424dd4fa925cf95b306dba6"}
        self.assertIsNone(_hash_from_hf_row(git_only, "readme.md"))
        self.assertEqual(_size_from_hf_row(row, "ggml-base.en.bin"), 147964211)
        self.assertEqual(_size_from_hf_row(git_only, "readme.md"), 0)


class CatalogTests(unittest.TestCase):
    def test_load_unique(self) -> None:
        items = load_catalog(PKG / "weights.json")
        ids = [w.id for w in items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(w.url.startswith("https://") for w in items))

    def test_tiny_chat_gguf_is_default(self) -> None:
        items = load_catalog(PKG / "weights.json")
        tiny = next(w for w in items if w.id == "qwen3-0.6b-q8_0")
        self.assertTrue(tiny.default)

    def test_human_bytes(self) -> None:
        self.assertEqual(human_bytes(2684354560), "2.5 GiB")

    def test_download_local_http(self) -> None:
        payload = b"w" * (128 * 1024)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                return

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            model = CatalogWeight(
                id="probe",
                title="probe",
                summary="",
                subdir="gguf",
                filename="probe.gguf",
                url=f"http://127.0.0.1:{port}/probe.gguf",
                bytes=len(payload),
                workflows=(),
            )
            with TemporaryDirectory() as tmp:
                store = Path(tmp)
                msg = download(model, store)
                dest = store / "gguf" / "probe.gguf"
                self.assertTrue(dest.is_file())
                self.assertEqual(dest.stat().st_size, len(payload))
                self.assertIn("downloaded", msg)
                again = download(model, store)
                self.assertTrue(again.startswith("already"))
                digest = hashlib.md5(payload).hexdigest()
                dest.unlink()
                ok = download(model, store, expected=FileHash("md5", digest))
                self.assertIn("downloaded", ok)
                dest.unlink()
                with self.assertRaises(RuntimeError):
                    download(model, store, expected=FileHash("md5", "0" * 32))
        finally:
            httpd.shutdown()

    def test_download_rejects_truncated_without_checksum(self) -> None:
        payload = b"w" * (32 * 1024)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                return

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            model = CatalogWeight(
                id="probe-short",
                title="probe",
                summary="",
                subdir="gguf",
                filename="probe-short.gguf",
                url=f"http://127.0.0.1:{port}/probe-short.gguf",
                bytes=len(payload) * 8,
                workflows=(),
            )
            with TemporaryDirectory() as tmp:
                store = Path(tmp)
                with self.assertRaises(RuntimeError) as ctx:
                    download(model, store)
                self.assertIn("incomplete download", str(ctx.exception))
                dest = store / "gguf" / "probe-short.gguf"
                self.assertFalse(dest.exists())
                self.assertFalse(dest.with_name(dest.name + ".part").exists())
        finally:
            httpd.shutdown()

    def test_download_rejects_zero_byte_without_checksum(self) -> None:
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, fmt, *args):
                return

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            model = CatalogWeight(
                id="probe-empty",
                title="probe",
                summary="",
                subdir="gguf",
                filename="probe-empty.gguf",
                url=f"http://127.0.0.1:{port}/probe-empty.gguf",
                bytes=128 * 1024,
                workflows=(),
            )
            with TemporaryDirectory() as tmp:
                store = Path(tmp)
                with self.assertRaises(RuntimeError) as ctx:
                    download(model, store)
                self.assertIn("empty download", str(ctx.exception))
                dest = store / "gguf" / "probe-empty.gguf"
                self.assertFalse(dest.exists())
                self.assertFalse(dest.with_name(dest.name + ".part").exists())
        finally:
            httpd.shutdown()

    def test_download_rejects_content_length_that_matches_a_short_body(self) -> None:
        payload = b"w" * (16 * 1024)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                return

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            model = CatalogWeight(
                id="probe-xet",
                title="probe",
                summary="",
                subdir="gguf",
                filename="probe-xet.gguf",
                url=f"http://127.0.0.1:{port}/probe-xet.gguf",
                bytes=len(payload) * 16,
                workflows=(),
            )
            with TemporaryDirectory() as tmp:
                store = Path(tmp)
                with self.assertRaises(RuntimeError) as ctx:
                    download(model, store)
                self.assertIn("incomplete download", str(ctx.exception))
                dest = store / "gguf" / "probe-xet.gguf"
                self.assertFalse(dest.exists())
        finally:
            httpd.shutdown()

    def test_download_uses_published_size_not_rounded_catalog(self) -> None:
        payload = b"w" * (64 * 1024)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                return

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            model = CatalogWeight(
                id="probe-lfs",
                title="probe",
                summary="",
                subdir="gguf",
                filename="probe-lfs.gguf",
                url=f"http://127.0.0.1:{port}/probe-lfs.gguf",
                bytes=2684354560,
                workflows=(),
            )
            with TemporaryDirectory() as tmp:
                store = Path(tmp)
                with patch(
                    "weights.lookup_published_meta",
                    return_value=(None, len(payload)),
                ):
                    msg = download(model, store)
                dest = store / "gguf" / "probe-lfs.gguf"
                self.assertTrue(dest.is_file())
                self.assertEqual(dest.stat().st_size, len(payload))
                self.assertIn("downloaded", msg)
        finally:
            httpd.shutdown()


class VerifyDownloadTests(unittest.TestCase):
    def _model(self, **kwargs) -> CatalogWeight:
        row = {
            "id": "probe",
            "title": "probe",
            "summary": "",
            "subdir": "gguf",
            "filename": "probe.gguf",
            "url": "https://huggingface.co/example/repo/resolve/main/probe.gguf",
            "bytes": 128 * 1024,
            "workflows": (),
        }
        row.update(kwargs)
        return CatalogWeight(**row)

    def test_rejects_missing_and_empty(self) -> None:
        model = self._model()
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.gguf"
            with self.assertRaises(RuntimeError) as ctx:
                verify_download(missing, model)
            self.assertIn("download missing", str(ctx.exception))
            empty = Path(tmp) / "empty.gguf"
            empty.write_bytes(b"")
            with self.assertRaises(RuntimeError) as ctx:
                verify_download(empty, model)
            self.assertIn("empty download", str(ctx.exception))

    def test_size_gate_when_checksum_empty(self) -> None:
        model = self._model(bytes=128 * 1024)
        with TemporaryDirectory() as tmp:
            short = Path(tmp) / "short.gguf"
            short.write_bytes(b"w" * 4096)
            with self.assertRaises(RuntimeError) as ctx:
                verify_download(short, model)
            self.assertIn("incomplete download", str(ctx.exception))
            ok = Path(tmp) / "ok.gguf"
            ok.write_bytes(b"w" * (128 * 1024))
            verify_download(ok, model)

    def test_published_size_overrides_catalog_bytes(self) -> None:
        model = self._model(bytes=99)
        with TemporaryDirectory() as tmp:
            blob = Path(tmp) / "blob.gguf"
            blob.write_bytes(b"w" * 4096)
            verify_download(blob, model, expected_size=4096)
            with self.assertRaises(RuntimeError):
                verify_download(blob, model, expected_size=8192)

    def test_hash_is_source_of_truth_when_present(self) -> None:
        payload = b"w" * 4096
        digest = hashlib.md5(payload).hexdigest()
        model = self._model(bytes=99, hash_algo="md5", hash_hex=digest)
        with TemporaryDirectory() as tmp:
            blob = Path(tmp) / "blob.gguf"
            blob.write_bytes(payload)
            verify_download(blob, model)
            with self.assertRaises(RuntimeError) as ctx:
                verify_download(blob, model, expected=FileHash("md5", "0" * 32))
            self.assertIn("md5 mismatch", str(ctx.exception))


class ForeignMountTests(unittest.TestCase):
    def test_disk_words_are_the_move_signal(self) -> None:
        self.assertEqual(ANOTHER_DISK, "another disk")
        self.assertEqual(UBUNTU_DISK, "this computer's Ubuntu disk")
        self.assertEqual(disk_words(True), ANOTHER_DISK)
        self.assertEqual(disk_words(False), UBUNTU_DISK)
        self.assertFalse(move_off(()))

    def test_prefix_or_fstype(self) -> None:
        self.assertEqual(
            FOREIGN_FSTYPES, frozenset({"ntfs", "fuseblk", "vfat", "exfat"})
        )
        info = "\n".join(
            (
                "194 167 254:32 / / rw - ext4 /dev/vdc rw",
                "200 194 8:1 / /home/user/Data rw - ntfs /dev/sdb1 rw",
                "201 194 8:2 / /home/user/Fat rw - vfat /dev/sdc1 rw",
                "202 194 8:3 / /home/user/Ex rw - exfat /dev/sdd1 rw",
                "203 194 8:4 / /home/user/Blk rw - fuseblk /dev/sde1 rw",
                "204 194 8:5 / /home/user/Ext rw - ext4 /dev/sdf1 rw",
                "205 194 8:6 / /home/user/My\\040Disk rw - ntfs /dev/sdg1 rw",
                "210 194 8:7 / /opt/models rw,noatime master:1 - fuseblk /dev/sdh1 rw",
            )
        )

        def check(path: str, want: bool) -> None:
            self.assertEqual(is_foreign_mount(Path(path), mountinfo=info), want, path)

        check("/media/disk/model.gguf", True)
        check("/mnt/disk/model.gguf", True)
        check("/mnt", True)
        check("/media", True)
        check("/mnt2/model.gguf", False)
        check("/home/user/model.gguf", False)
        check("/home/user/Data/model.gguf", True)
        check("/home/user/Fat/model.gguf", True)
        check("/home/user/Ex/model.gguf", True)
        check("/home/user/Blk/model.gguf", True)
        check("/home/user/Ext/model.gguf", False)
        check("/home/user/My Disk/model.gguf", True)
        check("/opt/models/a.gguf", True)

    def test_live_mountinfo_matches_explicit_text(self) -> None:
        path = Path("/tmp")
        info = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        self.assertEqual(is_foreign_mount(path), is_foreign_mount(path, mountinfo=info))

    def test_foreign_tree_copy_keeps_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "src" / "Qwen"
            repo.mkdir(parents=True)
            (repo / "config.json").write_text("{}", encoding="utf-8")
            blob = repo / "model.safetensors"
            blob.write_bytes(b"s" * (128 * 1024))
            alias = repo / "alias.safetensors"
            alias.symlink_to(blob.name)
            store = root / "Models"
            with patch("weights.is_foreign_mount", return_value=True):
                found = scan((repo.parent,), store)
                dirs = tuple(f for f in found if f.kind == "dir")
                self.assertEqual(len(dirs), 1)
                self.assertTrue(dirs[0].foreign)
                self.assertTrue(foreign_source(dirs[0]))
                log = organize(dirs, store, mode="copy")
            dest = store / "hf" / "Qwen"
            copied = dest / "model.safetensors"
            copied_alias = dest / "alias.safetensors"
            self.assertTrue(copied.is_file())
            self.assertFalse(copied.is_symlink())
            self.assertFalse(os.path.samefile(copied, blob))
            self.assertEqual(copied.read_bytes(), blob.read_bytes())
            self.assertTrue(copied_alias.is_file())
            self.assertFalse(copied_alias.is_symlink())
            self.assertTrue(blob.is_file())
            self.assertTrue(alias.is_symlink())
            self.assertTrue((repo / "config.json").is_file())
            self.assertTrue(any(line.startswith("copied ") for line in log))

    def test_move_refused_on_foreign(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "foreign.gguf"
            payload = b"g" * (128 * 1024)
            blob.write_bytes(payload)
            store = root / "Models"
            with patch("weights.is_foreign_mount", return_value=True):
                found = scan((src_dir,), store)
                with self.assertRaises(ForeignMountError) as ctx:
                    organize(found, store, mode="move")
                with self.assertRaises(ForeignMountError):
                    organize(found, store, mode="move", dry_run=True)
                with self.assertRaises(ForeignMountError):
                    organize(found, store, mode="copy", remove_source=True)
            self.assertIn(ANOTHER_DISK, str(ctx.exception))
            self.assertIn("Copy only", str(ctx.exception))
            self.assertEqual(blob.read_bytes(), payload)
            self.assertFalse((store / "gguf" / "foreign.gguf").exists())

    def test_foreign_flag_blocks_move(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "flagged.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = root / "Models"
            found = tuple(replace(item, foreign=True) for item in scan((src_dir,), store))
            with patch("weights.is_foreign_mount", return_value=False):
                self.assertTrue(foreign_source(found[0]))
                self.assertTrue(move_off(found))
                self.assertEqual(disk_words(found[0].foreign), ANOTHER_DISK)
                with self.assertRaises(ForeignMountError):
                    organize(found, store, mode="move")
            self.assertTrue(blob.is_file())
            self.assertFalse((store / "gguf" / "flagged.gguf").exists())

    def test_ensure_weight_copies_foreign_file(self) -> None:
        model = CatalogWeight(
            id="foreign-gguf",
            title="Foreign",
            summary="",
            subdir="gguf",
            filename="foreign.gguf",
            url="https://example.invalid/foreign.gguf",
            bytes=128 * 1024,
            workflows=(),
        )
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "stash"
            extra.mkdir()
            blob = extra / model.filename
            payload = b"g" * (128 * 1024)
            blob.write_bytes(payload)
            store = home / "Models"
            store.mkdir()
            target = self._target(home, store)
            with patch("weights.is_foreign_mount", return_value=True):
                msg = ensure_weight(model, target, (extra,))
            dest = store / "gguf" / model.filename
            self.assertTrue(dest.is_file())
            self.assertFalse(dest.is_symlink())
            self.assertFalse(os.path.samefile(dest, blob))
            self.assertEqual(dest.read_bytes(), payload)
            self.assertEqual(blob.read_bytes(), payload)
            self.assertTrue(msg.startswith("copied"))

    def _target(self, home: Path, store: Path) -> UserTarget:
        return UserTarget(
            name=pwd.getpwuid(os.getuid()).pw_name,
            uid=os.getuid(),
            gid=os.getgid(),
            home=home,
            model_root=store,
        )

    def test_cli_refuses_move_and_remove_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            src_dir = home / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "usb.gguf"
            payload = b"g" * (128 * 1024)
            blob.write_bytes(payload)
            store = home / "Models"
            store.mkdir()
            target = self._target(home, store)
            err = io.StringIO()
            with (
                patch("main.target_for", return_value=target),
                patch("main.saved_scan_folders", return_value=()),
                patch("weights.is_foreign_mount", return_value=True),
                patch("sys.stderr", err),
            ):
                rc_move = installer_main(["--organize-weights", "move"])
                rc_rm = installer_main(["--organize-weights", "copy", "--remove-source"])
            self.assertEqual(rc_move, 1)
            self.assertEqual(rc_rm, 1)
            self.assertIn(ANOTHER_DISK, err.getvalue())
            self.assertIn("Copy only", err.getvalue())
            self.assertEqual(blob.read_bytes(), payload)
            self.assertFalse((store / "gguf" / "usb.gguf").exists())

    def test_cli_copy_keeps_foreign_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            src_dir = home / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "usb.gguf"
            payload = b"g" * (128 * 1024)
            blob.write_bytes(payload)
            store = home / "Models"
            store.mkdir()
            target = self._target(home, store)
            out = io.StringIO()
            with (
                patch("main.target_for", return_value=target),
                patch("main.saved_scan_folders", return_value=()),
                patch("weights.is_foreign_mount", return_value=True),
                patch("sys.stdout", out),
            ):
                rc = installer_main(["--organize-weights", "copy"])
            dest = store / "gguf" / "usb.gguf"
            self.assertEqual(rc, 0)
            self.assertTrue(dest.is_file())
            self.assertFalse(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), payload)
            self.assertEqual(blob.read_bytes(), payload)
            self.assertIn("copied ", out.getvalue())


class StoreCopySafetyTests(unittest.TestCase):
    def _model(self, nbytes: int) -> CatalogWeight:
        return CatalogWeight(
            id="copy-probe",
            title="Copy",
            summary="",
            subdir="gguf",
            filename="model.gguf",
            url="https://example.invalid/model.gguf",
            bytes=nbytes,
            workflows=(),
        )

    def _place(self, home: Path, nbytes: int) -> tuple[Path, Path, bytes, Path, UserTarget]:
        extra = home / "stash"
        extra.mkdir()
        blob = extra / "model.gguf"
        payload = b"g" * nbytes
        blob.write_bytes(payload)
        os.chmod(blob, 0o777)
        store = home / "AI models"
        store.mkdir()
        target = UserTarget(
            name="tester",
            uid=os.getuid(),
            gid=os.getgid(),
            home=home,
            model_root=store,
        )
        return extra, blob, payload, store, target

    def test_disk_full_refuses_and_leaves_no_partial(self) -> None:
        nbytes = 2 * 1024 * 1024
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra, _blob, _payload, store, target = self._place(home, nbytes)
            model = self._model(nbytes)
            dest = store / "gguf" / model.filename

            class Stat:
                f_bavail = 128
                f_frsize = 4096

            with patch("weights.os.statvfs", return_value=Stat()):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_weight(model, target, (extra,))
            text = str(ctx.exception)
            self.assertIn("Could not copy this model.", text)
            self.assertIn("free space", text)
            self.assertIn("512 KiB", text)
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

            def fill_then_fail(src: Path, part: Path) -> None:
                part.write_bytes(b"x" * 4096)
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch("weights._transfer_file", side_effect=fill_then_fail):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_weight(model, target, (extra,))
            self.assertIn("filled up", str(ctx.exception))
            self.assertIn("partial file was removed", str(ctx.exception))
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_interrupted_copy_is_not_already(self) -> None:
        nbytes = 128 * 1024
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra, _blob, payload, store, target = self._place(home, nbytes)
            model = self._model(nbytes)
            dest = store / "gguf" / model.filename
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"p" * (64 * 1024))
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(b"t" * 1024)
            msg = ensure_weight(model, target, (extra,))
            self.assertTrue(msg.startswith("copied"), msg)
            self.assertFalse(msg.startswith("already"))
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(part.exists())
            self.assertEqual(dest.stat().st_mode & 0o777, 0o644)
            again = ensure_weight(model, target, (extra,))
            self.assertTrue(again.startswith("already"), again)

    def test_stale_temp_is_removed(self) -> None:
        nbytes = 128 * 1024
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra, _blob, payload, store, target = self._place(home, nbytes)
            model = self._model(nbytes)
            dest = store / "gguf" / model.filename
            part = dest.with_name(dest.name + ".part")
            part.parent.mkdir(parents=True)
            part.write_bytes(b"stale" * 20000)
            msg = ensure_weight(model, target, (extra,))
            self.assertTrue(msg.startswith("copied"), msg)
            self.assertFalse(part.exists())
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(dest.read_bytes().startswith(b"stale"))
            self.assertEqual(dest.stat().st_mode & 0o777, 0o644)

    def test_part_open_does_not_follow_a_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "sentinel"
            sentinel.write_bytes(b"precious")
            part = root / "model.gguf.part"
            part.symlink_to(sentinel)
            with self.assertRaises(OSError):
                _open_part(part)
            self.assertEqual(sentinel.read_bytes(), b"precious")
            self.assertTrue(part.is_symlink())

    def test_catalog_file_is_not_replaced_by_a_shorter_find(self) -> None:
        model = next(w for w in load_catalog() if w.id == "whisper-base-en")
        self.assertEqual(model.bytes, 147964211)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = home / "AI models"
            dest = store / model.subdir / model.filename
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"")
            os.truncate(dest, model.bytes)
            partial_dir = home / "Downloads"
            partial_dir.mkdir()
            partial = partial_dir / model.filename
            partial.write_bytes(b"p" * (100 * 1024))
            target = UserTarget(
                name="tester",
                uid=os.getuid(),
                gid=os.getgid(),
                home=home,
                model_root=store,
            )
            msg = ensure_weight(model, target)
            self.assertTrue(msg.startswith("already"), msg)
            self.assertEqual(dest.stat().st_size, model.bytes)
            with dest.open("rb") as fh:
                self.assertEqual(fh.read(1), b"\0")
            self.assertEqual(partial.read_bytes(), b"p" * (100 * 1024))
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_short_find_is_skipped_and_not_copied(self) -> None:
        model = next(w for w in load_catalog() if w.id == "whisper-base-en")
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            partial = home / "Downloads" / model.filename
            partial.parent.mkdir()
            partial.write_bytes(b"p" * (100 * 1024))
            store = home / "AI models"
            store.mkdir()
            dest = store / model.subdir / model.filename
            target = UserTarget(
                name="tester",
                uid=os.getuid(),
                gid=os.getgid(),
                home=home,
                model_root=store,
            )
            planned = ensure_weight(model, target, dry_run=True)
            self.assertIn("Skipped", planned)
            self.assertIn(str(partial), planned)
            self.assertIn("100 KiB", planned)
            self.assertIn("141 MiB", planned)
            self.assertIn("download ", planned)
            self.assertFalse(planned.startswith("copy "))
            self.assertFalse(dest.exists())
            with patch(
                "weights.urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_weight(model, target)
            text = str(ctx.exception)
            self.assertIn("Skipped", text)
            self.assertIn("100 KiB", text)
            self.assertIn("141 MiB", text)
            self.assertFalse(dest.exists())
            self.assertEqual(partial.read_bytes(), b"p" * (100 * 1024))
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_found_file_must_match_catalog_checksum(self) -> None:
        payload = b"g" * (128 * 1024)
        digest = hashlib.sha256(payload).hexdigest()
        model = CatalogWeight(
            id="hashed",
            title="Hashed",
            summary="",
            subdir="gguf",
            filename="hashed.gguf",
            url="https://example.invalid/hashed.gguf",
            bytes=len(payload),
            workflows=(),
            hash_algo="sha256",
            hash_hex=digest,
        )
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "Downloads"
            extra.mkdir()
            bad = extra / model.filename
            bad.write_bytes(b"x" * len(payload))
            store = home / "AI models"
            dest = store / model.subdir / model.filename
            dest.parent.mkdir(parents=True)
            dest.write_bytes(payload)
            target = UserTarget(
                name="tester",
                uid=os.getuid(),
                gid=os.getgid(),
                home=home,
                model_root=store,
            )
            msg = ensure_weight(model, target)
            self.assertTrue(msg.startswith("already"), msg)
            self.assertEqual(dest.read_bytes(), payload)
            dest.unlink()
            with patch(
                "weights.urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_weight(model, target)
            self.assertIn("checksum", str(ctx.exception).lower())
            self.assertFalse(dest.exists())
            self.assertEqual(bad.read_bytes(), b"x" * len(payload))

    def test_space_refusal_keeps_existing_file(self) -> None:
        nbytes = 128 * 1024
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra, blob, payload, store, target = self._place(home, nbytes)
            model = self._model(nbytes)
            dest = store / "gguf" / model.filename
            dest.parent.mkdir(parents=True)
            kept = b"keep" * 4000
            dest.write_bytes(kept)

            class Stat:
                f_bavail = 128
                f_frsize = 4096

            with patch("weights.os.statvfs", return_value=Stat()):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_weight(model, target, (extra,))
            self.assertIn("Could not copy this model.", str(ctx.exception))
            self.assertIn("free space", str(ctx.exception))
            self.assertEqual(dest.read_bytes(), kept)
            self.assertEqual(blob.read_bytes(), payload)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_download_failure_keeps_qwen_file(self) -> None:
        model = next(w for w in load_catalog() if w.id == "qwen3-0.6b-q8_0")
        self.assertEqual(model.filename, "Qwen3-0.6B-Q8_0.gguf")
        with TemporaryDirectory() as tmp:
            store = Path(tmp)
            dest = store / model.subdir / model.filename
            dest.parent.mkdir(parents=True)
            payload = b"user-file" * 10000
            dest.write_bytes(payload)
            with patch(
                "weights.urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ):
                with self.assertRaises(urllib.error.URLError):
                    download(model, store)
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

            dest.write_bytes(b"")
            os.truncate(dest, model.bytes)
            with patch(
                "weights.urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ):
                with self.assertRaises(urllib.error.URLError):
                    download(model, store, force=True)
            self.assertEqual(dest.stat().st_size, model.bytes)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())


class OrganizeSymlinkTests(unittest.TestCase):
    """organize() is the function the Weights tab and --organize-weights call."""

    def _assert_no_links(self, root: Path) -> None:
        self.assertTrue(root.exists(), root)
        self.assertFalse(os.path.islink(root), root)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in [*dirnames, *filenames]:
                path = os.path.join(dirpath, name)
                self.assertFalse(os.path.islink(path), path)

    def _dirs(self, scan_root: Path, store: Path) -> tuple:
        found = tuple(item for item in scan((scan_root,), store) if item.kind == "dir")
        self.assertEqual(len(found), 1)
        return found

    def _payload(self) -> bytes:
        return b"s" * (128 * 1024)

    def _bundle(self, repo: Path, payload: bytes, *, link: Path | None = None) -> None:
        repo.mkdir(parents=True)
        (repo / "config.json").write_text("{}", encoding="utf-8")
        blob = repo / "model.safetensors"
        if link is None:
            blob.write_bytes(payload)
        else:
            blob.symlink_to(link)

    def test_absolute_link_bundle_lands_as_a_regular_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            self.assertTrue(os.path.isabs(os.readlink(repo / "model.safetensors")))
            store = root / "store"
            found = self._dirs(repo.parent, store)
            log = organize(found, store)
            dest = store / "hf" / "Qwen"
            copied = dest / "model.safetensors"
            self._assert_no_links(store)
            self.assertTrue(stat.S_ISREG(os.lstat(copied).st_mode))
            self.assertEqual(copied.read_bytes(), payload)
            self.assertFalse(os.path.samefile(copied, blob))
            self.assertTrue((repo / "model.safetensors").is_symlink())
            self.assertTrue(any(line.startswith("copied ") and "sha256=" in line for line in log))
            self.assertFalse(any("checksum mismatch" in line for line in log))
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_move_dereferences_an_absolute_link(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            log = organize(found, store, mode="move")
            copied = store / "hf" / "Qwen" / "model.safetensors"
            self._assert_no_links(store)
            self.assertTrue(stat.S_ISREG(os.lstat(copied).st_mode))
            self.assertEqual(copied.read_bytes(), payload)
            self.assertFalse(repo.exists())
            self.assertEqual(blob.read_bytes(), payload)
            self.assertFalse(os.path.samefile(copied, blob))
            self.assertTrue(any(line.startswith("moved ") and "sha256=" in line for line in log))
            self.assertTrue(any("removed source" in line for line in log))

    def test_hf_snapshot_links_copy_and_verify(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            cache = root / "incoming" / "models--lab--demo"
            blobs = cache / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "cfg").write_text("{}", encoding="utf-8")
            weights = blobs / "weights"
            weights.write_bytes(payload)
            snap = cache / "snapshots" / "main"
            snap.mkdir(parents=True)
            (snap / "config.json").symlink_to("../../blobs/cfg")
            (snap / "model.safetensors").symlink_to("../../blobs/weights")
            store = root / "store"
            found = self._dirs(root / "incoming", store)
            log = organize(found, store)
            dest = store / "hf" / "lab--demo"
            copied = dest / "model.safetensors"
            self._assert_no_links(store)
            self.assertTrue(stat.S_ISREG(os.lstat(copied).st_mode))
            self.assertEqual(copied.read_bytes(), payload)
            self.assertEqual((dest / "config.json").read_text(encoding="utf-8"), "{}")
            self.assertFalse(os.path.samefile(copied, weights))
            self.assertTrue((snap / "model.safetensors").is_symlink())
            self.assertTrue(any(line.startswith("copied ") and "sha256=" in line for line in log))
            self.assertFalse(any("checksum mismatch" in line for line in log))

    def test_inside_folder_link_is_real_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            nested = repo / "nested"
            nested.mkdir(parents=True)
            (repo / "config.json").write_text("{}", encoding="utf-8")
            (repo / "model.safetensors").write_bytes(payload)
            (nested / "extra.bin").write_bytes(b"extra")
            (repo / "alias").symlink_to("nested", target_is_directory=True)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            log = organize(found, store)
            dest = store / "hf" / "Qwen"
            self._assert_no_links(store)
            self.assertEqual((dest / "alias" / "extra.bin").read_bytes(), b"extra")
            self.assertEqual((dest / "nested" / "extra.bin").read_bytes(), b"extra")
            self.assertTrue(any(line.startswith("copied ") and "sha256=" in line for line in log))
            self.assertFalse(any("checksum mismatch" in line for line in log))

    def test_dangling_link_is_refused_before_a_partial_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "incoming" / "Qwen"
            nested = repo / "nested"
            nested.mkdir(parents=True)
            (repo / "config.json").write_text("{}", encoding="utf-8")
            (repo / "model.safetensors").write_bytes(self._payload())
            (nested / "gone.safetensors").symlink_to("missing.safetensors")
            store = root / "store"
            store.mkdir()
            found = self._dirs(repo.parent, store)
            log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("points nowhere", text)
            self.assertIn("The copy was not started", text)
            self.assertNotIn("checksum", text)
            self.assertFalse((store / "hf").exists())
            self.assertTrue((repo / "model.safetensors").is_file())
            self.assertTrue((nested / "gone.safetensors").is_symlink())

    def test_link_loop_is_refused_before_a_partial_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "incoming" / "Qwen"
            repo.mkdir(parents=True)
            (repo / "config.json").write_text("{}", encoding="utf-8")
            (repo / "model.safetensors").write_bytes(self._payload())
            (repo / "a.safetensors").symlink_to("b.safetensors")
            (repo / "b.safetensors").symlink_to("a.safetensors")
            store = root / "store"
            store.mkdir()
            found = self._dirs(repo.parent, store)
            log = organize(found, store, mode="move")
            text = "\n".join(log)
            self.assertIn("loops", text)
            self.assertIn("The copy was not started", text)
            self.assertFalse((store / "hf").exists())
            self.assertTrue(repo.is_dir())
            self.assertTrue((repo / "model.safetensors").is_file())

    def test_folder_link_outside_the_bundle_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.bin").write_bytes(b"nope")
            repo = root / "incoming" / "Qwen"
            repo.mkdir(parents=True)
            (repo / "config.json").write_text("{}", encoding="utf-8")
            (repo / "model.safetensors").write_bytes(self._payload())
            (repo / "escape").symlink_to(outside, target_is_directory=True)
            store = root / "store"
            store.mkdir()
            found = self._dirs(repo.parent, store)
            log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("outside the bundle", text)
            self.assertIn("The copy was not started", text)
            self.assertFalse((store / "hf").exists())
            self.assertEqual((outside / "secret.bin").read_bytes(), b"nope")

    def test_unreadable_link_target_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            store.mkdir()
            found = self._dirs(repo.parent, store)
            with patch("weights._read_link_target", side_effect=PermissionError("denied")):
                log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("cannot be read", text)
            self.assertIn("The copy was not started", text)
            self.assertFalse((store / "hf").exists())
            self.assertEqual(blob.read_bytes(), payload)

    def test_cli_organize_weights_dereferences_hf_links(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            payload = self._payload()
            incoming = home / "incoming"
            cache = incoming / "models--lab--demo"
            blobs = cache / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "cfg").write_text("{}", encoding="utf-8")
            (blobs / "weights").write_bytes(payload)
            snap = cache / "snapshots" / "main"
            snap.mkdir(parents=True)
            (snap / "config.json").symlink_to("../../blobs/cfg")
            (snap / "model.safetensors").symlink_to("../../blobs/weights")
            store = home / "store"
            store.mkdir()
            target = UserTarget(
                name=pwd.getpwuid(os.getuid()).pw_name,
                uid=os.getuid(),
                gid=os.getgid(),
                home=home,
                model_root=store,
            )
            out = io.StringIO()
            err = io.StringIO()
            with (
                patch("main.target_for", return_value=target),
                patch("main.saved_scan_folders", return_value=()),
                patch("sys.stdout", out),
                patch("sys.stderr", err),
            ):
                rc = installer_main(
                    ["--organize-weights", "copy", "--scan-folder", str(incoming)]
                )
            dest = store / "hf" / "lab--demo"
            self.assertEqual(rc, 0, err.getvalue())
            self._assert_no_links(store)
            self.assertEqual((dest / "model.safetensors").read_bytes(), payload)
            self.assertIn("sha256=", out.getvalue())
            self.assertIn("copied ", out.getvalue())
            self.assertEqual(err.getvalue(), "")

    def test_cli_organize_weights_refuses_a_dangling_link(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            incoming = home / "incoming"
            repo = incoming / "Qwen"
            repo.mkdir(parents=True)
            (repo / "config.json").write_text("{}", encoding="utf-8")
            (repo / "model.safetensors").write_bytes(self._payload())
            (repo / "gone.safetensors").symlink_to("missing.safetensors")
            store = home / "store"
            store.mkdir()
            target = UserTarget(
                name=pwd.getpwuid(os.getuid()).pw_name,
                uid=os.getuid(),
                gid=os.getgid(),
                home=home,
                model_root=store,
            )
            err = io.StringIO()
            with (
                patch("main.target_for", return_value=target),
                patch("main.saved_scan_folders", return_value=()),
                patch("sys.stderr", err),
            ):
                rc = installer_main(
                    ["--organize-weights", "copy", "--scan-folder", str(incoming)]
                )
            self.assertEqual(rc, 1)
            self.assertIn("points nowhere", err.getvalue())
            self.assertIn("The copy was not started", err.getvalue())
            self.assertFalse((store / "hf").exists())
            self.assertTrue((repo / "model.safetensors").is_file())

    def test_weights_tab_calls_organize_and_shows_link_errors(self) -> None:
        for name in ("gtk_ui.py", "qt_ui.py"):
            text = (PKG / "ui" / name).read_text(encoding="utf-8")
            self.assertIn("log = organize(", text)
            self.assertIn("OrganizeLinkError", text)

    def test_store_copy_follows_file_links(self) -> None:
        text = (PKG / "weights.py").read_text(encoding="utf-8")
        self.assertNotIn("copytree", text)
        self.assertNotIn("rsync", text)
        self.assertNotIn("shutil.copy2", text)
        self.assertIn("def _transfer_file", text)
        self.assertIn("def _materialize_plan", text)

    def test_same_disk_move_without_links_renames(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            dir_ino = repo.stat().st_ino
            file_ino = (repo / "model.safetensors").stat().st_ino
            store = root / "store"
            found = self._dirs(repo.parent, store)
            with (
                patch("weights._transfer_file") as transfer,
                patch("weights.os.replace", wraps=os.replace) as repl,
            ):
                log = organize(found, store, mode="move")
            transfer.assert_not_called()
            repl.assert_called_once()
            dest = store / "hf" / "Qwen"
            copied = dest / "model.safetensors"
            self._assert_no_links(store)
            self.assertEqual(dest.stat().st_ino, dir_ino)
            self.assertEqual(copied.stat().st_ino, file_ino)
            self.assertEqual(copied.read_bytes(), payload)
            self.assertFalse(repo.exists())
            self.assertTrue(any(line.startswith("moved ") for line in log))
            self.assertFalse(any("sha256=" in line for line in log))

    def test_move_leaves_an_existing_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            dest = store / "hf" / "Qwen"
            dest.mkdir(parents=True)
            (dest / "config.json").write_text("keep", encoding="utf-8")
            log = organize(found, store, mode="move")
            self.assertEqual((dest / "config.json").read_text(encoding="utf-8"), "keep")
            self.assertEqual((repo / "model.safetensors").read_bytes(), payload)
            self.assertTrue(repo.is_dir())
            self.assertTrue(any("appeared" in line for line in log))

    def test_move_with_links_refuses_when_space_is_low(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            store.mkdir()
            found = self._dirs(repo.parent, store)
            need = _copy_bytes_needed(len(payload) + len(b"{}"))

            class Stat:
                f_bavail = 1
                f_frsize = 4096

            with (
                patch("weights.os.statvfs", return_value=Stat()),
                patch("weights._transfer_file") as transfer,
            ):
                log = organize(found, store, mode="move")
            transfer.assert_not_called()
            text = "\n".join(log)
            self.assertIn(human_bytes(need), text)
            self.assertIn(human_bytes(4096), text)
            self.assertIn("free space", text)
            self.assertIn("Could not copy this model.", text)
            self.assertFalse((store / "hf").exists())
            self.assertTrue((repo / "model.safetensors").is_symlink())
            self.assertEqual(blob.read_bytes(), payload)

    def test_move_verify_failure_keeps_the_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            real = hash_path
            calls = {"n": 0}

            def fake(path: Path, algo: str = "sha256") -> str:
                calls["n"] += 1
                digest = real(path, algo)
                if calls["n"] == 2:
                    return "0" * 64
                return digest

            with patch("weights.hash_path", fake):
                log = organize(found, store, mode="move")
            self.assertTrue(repo.is_dir())
            self.assertTrue((repo / "model.safetensors").is_symlink())
            self.assertEqual(blob.read_bytes(), payload)
            self.assertFalse((store / "hf" / "Qwen").exists())
            self.assertTrue(any("checksum mismatch" in line for line in log))
            self.assertFalse(any("removed source" in line for line in log))

    def test_killed_bundle_copy_leaves_no_final_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            dest = store / "hf" / "Qwen"
            stale = dest.with_name(dest.name + ".part")
            stale.mkdir(parents=True)
            (stale / "model.safetensors").write_bytes(b"short")
            sentinel = store / "hf" / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            def die(src: Path, part: Path, linked: bool = False) -> None:
                part.write_bytes(b"trunc")
                raise OSError(errno.EIO, "killed")

            with patch("weights._transfer_file", side_effect=die):
                log = organize(found, store)
            self.assertFalse(dest.exists())
            self.assertFalse(stale.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual((repo / "model.safetensors").read_bytes(), payload)
            self.assertTrue(any("Could not copy this model." in line for line in log))
            again = organize(found, store)
            self.assertEqual((dest / "model.safetensors").read_bytes(), payload)
            self.assertEqual((dest / "config.json").read_text(encoding="utf-8"), "{}")
            self.assertFalse(stale.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertTrue(any(line.startswith("copied ") for line in again))

    def test_source_removal_failure_leaves_a_complete_store_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            real_unlink = Path.unlink

            def boom(path: Path, *args: object, **kwargs: object) -> None:
                if path.is_relative_to(repo):
                    raise OSError(errno.EIO, "killed")
                real_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", boom):
                log = organize(found, store, mode="move")
            copied = store / "hf" / "Qwen" / "model.safetensors"
            self.assertEqual(copied.read_bytes(), payload)
            self.assertEqual(
                (store / "hf" / "Qwen" / "config.json").read_text(encoding="utf-8"),
                "{}",
            )
            self.assertFalse(copied.with_name(copied.name + ".part").exists())
            self.assertTrue((repo / "model.safetensors").is_symlink())
            self.assertEqual(blob.read_bytes(), payload)
            self.assertTrue(any("Kept source entries" in line for line in log))
            self.assertFalse(any("removed source" in line for line in log))

    def test_source_removal_keeps_unverified_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            real_replace = os.replace

            def add_extras(src: str | Path, dst: str | Path) -> None:
                src_path = Path(src)
                if src_path.name.endswith(".part"):
                    (repo / "later.txt").write_text("later", encoding="utf-8")
                    os.mkfifo(repo / "pipe")
                real_replace(src, dst)

            with (
                patch("weights.os.replace", side_effect=add_extras),
                patch("weights.shutil.rmtree", wraps=shutil.rmtree) as removed,
            ):
                log = organize(found, store, mode="move")
            dest = store / "hf" / "Qwen"
            self.assertEqual((dest / "model.safetensors").read_bytes(), payload)
            self.assertTrue((repo / "later.txt").is_file())
            self.assertTrue(stat.S_ISFIFO((repo / "pipe").stat().st_mode))
            self.assertFalse((repo / "model.safetensors").exists())
            self.assertFalse((repo / "config.json").exists())
            text = "\n".join(log)
            self.assertIn("later.txt", text)
            self.assertIn("pipe", text)
            self.assertIn("Kept source entries", text)
            for call in removed.call_args_list:
                self.assertNotEqual(Path(call.args[0]).resolve(), repo.resolve())

    def test_failed_rename_deletes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            sentinel = store / "hf" / "keep.txt"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("keep", encoding="utf-8")

            def fail_replace(src: str | Path, dst: str | Path) -> None:
                dest = Path(dst)
                dest.mkdir(parents=True)
                (dest / "keep.txt").write_text("keep", encoding="utf-8")
                raise OSError(errno.EIO, "fail")

            with patch("weights.os.replace", side_effect=fail_replace):
                log = organize(found, store, mode="move")
            text = "\n".join(log)
            self.assertIn("Nothing was written", text)
            self.assertIn("Organize is refused", text)
            self.assertEqual((repo / "model.safetensors").read_bytes(), payload)
            self.assertTrue(repo.is_dir())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            planted = store / "hf" / "Qwen" / "keep.txt"
            self.assertEqual(planted.read_text(encoding="utf-8"), "keep")

    def test_exdev_and_readonly_parent_keep_the_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            real_replace = os.replace

            def exdev(src: str | Path, dst: str | Path) -> None:
                if not Path(src).name.endswith(".part"):
                    raise OSError(errno.EXDEV, "cross-device")
                real_replace(src, dst)

            with patch("weights.os.replace", side_effect=exdev):
                log = organize(found, store, mode="move")
            dest = store / "hf" / "Qwen"
            self.assertEqual((dest / "model.safetensors").read_bytes(), payload)
            self.assertTrue(repo.is_dir())
            self.assertEqual((repo / "model.safetensors").read_bytes(), payload)
            self.assertTrue(any("kept source" in line for line in log))
            self.assertFalse(any("removed source" in line for line in log))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            with patch("weights.os.access", return_value=False):
                log = organize(found, store, mode="move")
            dest = store / "hf" / "Qwen"
            self.assertEqual((dest / "model.safetensors").read_bytes(), payload)
            self.assertNotEqual(dest.stat().st_ino, repo.stat().st_ino)
            self.assertTrue(repo.is_dir())
            self.assertTrue(any("kept source" in line and "not writable" in line for line in log))

    def test_plain_english_for_enospc_unreadable_missing_link_and_chown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)

            def fill(src: Path, part: Path, linked: bool = False) -> None:
                part.write_bytes(b"x")
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch("weights._transfer_file", side_effect=fill):
                log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("filled up", text)
            self.assertIn("partial file was removed", text)
            self.assertNotIn("Traceback", text)
            self.assertFalse((store / "hf" / "Qwen").exists())
            self.assertFalse((store / "hf" / "Qwen.part").exists())
            self.assertEqual((repo / "model.safetensors").read_bytes(), payload)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = root / "Downloads" / "plain.gguf"
            blob.parent.mkdir()
            blob.write_bytes(self._payload())
            store = root / "store"
            found = scan((blob.parent,), store)
            real_open = open

            def blocked(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
                if Path(str(file)) == blob and "b" in mode:
                    raise PermissionError(errno.EACCES, "denied")
                return real_open(file, mode, *args, **kwargs)

            with patch("builtins.open", side_effect=blocked):
                log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("cannot be read", text)
            self.assertNotIn("Traceback", text)
            self.assertFalse((store / "gguf" / "plain.gguf").exists())
            self.assertEqual(blob.read_bytes(), self._payload())

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            blob = root / "blob.safetensors"
            blob.write_bytes(payload)
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload, link=blob)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            import weights as weights_mod

            real_transfer = weights_mod._transfer_file

            def drop(src: Path, part: Path, linked: bool = False) -> None:
                if linked:
                    src.unlink()
                real_transfer(src, part, linked=linked)

            with patch("weights._transfer_file", side_effect=drop):
                log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("link target was removed", text)
            self.assertNotIn("Traceback", text)
            self.assertFalse((store / "hf" / "Qwen").exists())
            self.assertFalse((store / "hf" / "Qwen.part").exists())
            self.assertTrue((repo / "model.safetensors").is_symlink())

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            found = self._dirs(repo.parent, store)

            def deny(dest: Path, uid: int, gid: int) -> None:
                raise OSError(errno.EPERM, "not permitted")

            with patch("weights._chown_tree", side_effect=deny):
                log = organize(found, store, mode="move", uid=os.getuid(), gid=os.getgid())
            dest = store / "hf" / "Qwen"
            self.assertEqual((dest / "model.safetensors").read_bytes(), payload)
            self.assertFalse(repo.exists())
            text = "\n".join(log)
            self.assertIn("Could not change the owner", text)
            self.assertIn("in place", text)
            self.assertTrue(any(line.startswith("moved ") for line in log))

    def test_refused_bundle_keeps_earlier_success(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            incoming = root / "incoming"
            good = incoming / "Aaa"
            self._bundle(good, payload)
            bad = incoming / "Zed"
            bad.mkdir(parents=True)
            (bad / "config.json").write_text("{}", encoding="utf-8")
            (bad / "model.safetensors").write_bytes(payload)
            (bad / "gone.safetensors").symlink_to("missing.safetensors")
            store = root / "store"
            found = tuple(item for item in scan((incoming,), store) if item.kind == "dir")
            self.assertEqual(len(found), 2)
            log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("copied ", text)
            self.assertIn(str(good), text)
            self.assertIn("points nowhere", text)
            self.assertEqual((store / "hf" / "Aaa" / "model.safetensors").read_bytes(), payload)
            self.assertFalse((store / "hf" / "Zed").exists())

    def test_skips_file_with_partial_sibling(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            aria = root / "incoming" / "aria.gguf"
            aria.parent.mkdir(parents=True)
            aria.write_bytes(payload)
            (aria.parent / "aria.gguf.aria2").write_bytes(b"1")
            part = root / "more" / "part.gguf"
            part.parent.mkdir()
            part.write_bytes(payload)
            (part.parent / "part.gguf.part").write_bytes(b"2")
            store = root / "store"
            found = scan((aria.parent, part.parent), store)
            self.assertEqual({item.state for item in found}, {"busy"})
            log = organize(found, store)
            text = "\n".join(log)
            self.assertIn("aria.gguf.aria2", text)
            self.assertIn("part.gguf.part", text)
            self.assertIn("not finished", text)
            self.assertFalse((store / "gguf").exists())
            self.assertEqual(aria.read_bytes(), payload)
            self.assertEqual(part.read_bytes(), payload)

    def test_cross_device_move_checks_free_space(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            store = root / "store"
            store.mkdir()
            found = self._dirs(repo.parent, store)
            need = _copy_bytes_needed(len(payload) + len(b"{}"))

            class Stat:
                f_bavail = 1
                f_frsize = 4096

            with (
                patch("weights._same_device", return_value=False),
                patch("weights.os.statvfs", return_value=Stat()),
                patch("weights._transfer_file") as transfer,
            ):
                log = organize(found, store, mode="move")
            transfer.assert_not_called()
            text = "\n".join(log)
            self.assertIn(human_bytes(need), text)
            self.assertIn("free space", text)
            self.assertEqual((repo / "model.safetensors").read_bytes(), payload)
            self.assertFalse((store / "hf" / "Qwen").exists())

    def test_link_appearing_before_rename_is_copied(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._payload()
            repo = root / "incoming" / "Qwen"
            self._bundle(repo, payload)
            extra = root / "extra.safetensors"
            extra.write_bytes(b"e" * 32)
            store = root / "store"
            found = self._dirs(repo.parent, store)
            dir_ino = repo.stat().st_ino

            def appear(src: Path, parent: Path) -> bool:
                link = repo / "extra.safetensors"
                if not link.exists():
                    link.symlink_to(extra)
                return True

            with patch("weights._same_device", side_effect=appear):
                log = organize(found, store, mode="move")
            dest = store / "hf" / "Qwen"
            copied = dest / "extra.safetensors"
            self.assertTrue(stat.S_ISREG(os.lstat(copied).st_mode))
            self.assertEqual(copied.read_bytes(), b"e" * 32)
            self.assertNotEqual(dest.stat().st_ino, dir_ino)
            self.assertFalse(repo.exists())
            self.assertTrue(any("sha256=" in line for line in log))

    def test_snapshot_collision_name_uses_the_revision(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            def snap(rev: str) -> None:
                folder = root / rev / "models--org--m" / "snapshots" / rev
                folder.mkdir(parents=True)
                (folder / "config.json").write_text("{}", encoding="utf-8")
                (folder / "model.safetensors").write_bytes(self._payload())

            snap("rev1")
            snap("rev2")
            store = root / "store"
            found = tuple(
                item
                for item in scan((root / "rev1", root / "rev2"), store)
                if item.kind == "dir"
            )
            names = {item.dest_name for item in found}
            self.assertEqual(len(names), 2)
            self.assertIn("org--m", names)
            self.assertTrue(any(name.startswith("org--m-") for name in names))
            self.assertNotIn("snapshots-org--m", names)
            self.assertFalse(any(name.startswith("snapshots-") for name in names))


if __name__ == "__main__":
    unittest.main()
