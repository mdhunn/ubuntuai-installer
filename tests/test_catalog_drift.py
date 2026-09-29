from __future__ import annotations

import io
import json
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG

from catalog_drift import (
    check_catalog,
    format_report,
    main,
    size_from_headers,
    size_from_hf_row,
)


class FakeHeaders(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if str(k).lower() == str(key).lower():
                return v
        return default


class FakeResp:
    def __init__(self, status=200, headers=None, body=b"", url="https://example.test/file"):
        self.status = status
        self.headers = FakeHeaders(headers or {})
        self._body = body
        self._url = url

    def read(self):
        return self._body

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _catalog(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "weights.json"
    path.write_text(json.dumps({"models": rows}), encoding="utf-8")
    return path


def _row(**overrides: object) -> dict:
    row = {
        "id": "probe-gguf",
        "title": "probe",
        "summary": "",
        "subdir": "gguf",
        "filename": "probe.gguf",
        "url": "https://huggingface.co/org/repo/resolve/main/probe.gguf",
        "bytes": 100,
        "workflows": [],
        "checksum": "",
        "default": False,
    }
    row.update(overrides)
    return row


class HeaderParseTests(unittest.TestCase):
    def test_prefers_x_linked_size(self) -> None:
        n, src = size_from_headers(FakeHeaders({"x-linked-size": "64", "Content-Length": "8"}))
        self.assertEqual(n, 64)
        self.assertEqual(src, "x-linked-size")

    def test_content_length(self) -> None:
        n, src = size_from_headers(FakeHeaders({"Content-Length": "512"}))
        self.assertEqual(n, 512)
        self.assertEqual(src, "content-length")

    def test_rejects_zero(self) -> None:
        n, src = size_from_headers(FakeHeaders({"Content-Length": "0"}))
        self.assertIsNone(n)
        self.assertEqual(src, "")


class HfRowTests(unittest.TestCase):
    def test_lfs_size(self) -> None:
        self.assertEqual(size_from_hf_row({"lfs": {"size": 2497280256}, "size": 1}), 2497280256)

    def test_plain_size(self) -> None:
        self.assertEqual(size_from_hf_row({"size": 147964211}), 147964211)

    def test_ignores_junk(self) -> None:
        self.assertIsNone(size_from_hf_row("nope"))
        self.assertIsNone(size_from_hf_row({"lfs": {"size": 0}}))


class FakeHttp:
    def __init__(self, routes: dict[str, FakeResp | Exception]):
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append(f"{req.get_method()} {url}")
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise urllib.error.URLError(f"unexpected URL {url}")


class CheckCatalogTests(unittest.TestCase):
    def test_match_is_clean(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": FakeResp(headers={"Content-Length": "100"}),
                "paths-info": FakeResp(body=json.dumps([{"path": "probe.gguf", "lfs": {"size": 100}}]).encode()),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row()])
            checks = check_catalog(path, http)
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].ok())
        self.assertIn("lfs.size", checks[0].source)
        self.assertTrue(any(c.startswith("HEAD ") for c in http.calls))

    def test_size_mismatch(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": FakeResp(headers={"Content-Length": "200"}),
                "paths-info": FakeResp(body=json.dumps([{"path": "probe.gguf", "lfs": {"size": 200}}]).encode()),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row(bytes=100)])
            checks = check_catalog(path, http)
        self.assertFalse(checks[0].ok())
        text = format_report(checks)
        self.assertIn("probe-gguf", text)
        self.assertIn("catalog 100", text)
        self.assertIn("upstream 200", text)

    def test_unresolved_url(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": urllib.error.URLError("nope"),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row()])
            checks = check_catalog(path, http)
        self.assertFalse(checks[0].ok())
        self.assertFalse(checks[0].resolved)
        self.assertIn("did not resolve", format_report(checks))

    def test_missing_size(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": FakeResp(headers={}),
                "paths-info": FakeResp(body=json.dumps([{"path": "probe.gguf"}]).encode()),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row()])
            checks = check_catalog(path, http)
        self.assertFalse(checks[0].ok())
        self.assertTrue(checks[0].resolved)
        self.assertIn("upstream size missing", format_report(checks))

    def test_disagreement_is_failure(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": FakeResp(headers={"Content-Length": "90"}),
                "paths-info": FakeResp(body=json.dumps([{"path": "probe.gguf", "lfs": {"size": 100}}]).encode()),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row()])
            checks = check_catalog(path, http)
        self.assertFalse(checks[0].ok())
        self.assertIn("disagree", format_report(checks))

    def test_main_match_exit_zero(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": FakeResp(headers={"x-linked-size": "100"}),
                "paths-info": FakeResp(body=json.dumps([{"lfs": {"size": 100}}]).encode()),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row()])
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main(["--catalog", str(path)], urlopen=http)
        self.assertEqual(code, 0)
        self.assertIn("match upstream", buf.getvalue())

    def test_main_drift_exit_one(self) -> None:
        http = FakeHttp(
            {
                "probe.gguf": FakeResp(headers={"Content-Length": "9"}),
                "paths-info": FakeResp(body=json.dumps([{"lfs": {"size": 9}}]).encode()),
            }
        )
        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row(bytes=100)])
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main(["--catalog", str(path)], urlopen=http)
        self.assertEqual(code, 1)

    def test_never_opens_real_network(self) -> None:
        def boom(req, timeout=None):
            raise AssertionError(f"network call {req.full_url}")

        with TemporaryDirectory() as tmp:
            path = _catalog(Path(tmp), [_row()])
            with self.assertRaises(AssertionError):
                check_catalog(path, boom)
        self.assertTrue((PKG / "weights.json").is_file())


if __name__ == "__main__":
    unittest.main()
