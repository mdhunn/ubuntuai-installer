from __future__ import annotations

import hashlib
import http.server
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG

from domain import CatalogWeight, FileHash
from weights import (
    _hash_from_hf_row,
    _size_from_hf_row,
    classify,
    detect_bundle,
    download,
    hash_path,
    human_bytes,
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
            log = organize(tuple(dirs), store, mode="link")
            dest = store / "hf" / "Qwen3-4B"
            self.assertTrue(dest.is_symlink())
            self.assertTrue(any(line.startswith("link ") for line in log))

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
    def test_scan_and_link(self) -> None:
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
            log = organize(found, store, mode="link")
            dest = store / "gguf" / "Dolphin.Q8_0.gguf"
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(), blob.resolve())
            self.assertTrue(any(line.startswith("link ") for line in log))
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

    def test_link_ignores_remove_source(self) -> None:
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "Downloads"
            src_dir.mkdir()
            blob = src_dir / "keep-link.gguf"
            blob.write_bytes(b"g" * (128 * 1024))
            store = Path(tmp) / "Models"
            found = scan((src_dir,), store)
            organize(found, store, mode="link", remove_source=True)
            dest = store / "gguf" / "keep-link.gguf"
            self.assertTrue(dest.is_symlink())
            self.assertTrue(blob.exists())

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


if __name__ == "__main__":
    unittest.main()
