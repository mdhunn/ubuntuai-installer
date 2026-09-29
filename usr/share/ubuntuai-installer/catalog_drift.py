"""Compare weights.json .bytes to published upstream sizes.

Machine-independent. Reads the catalog and asks the download URLs plus the
Hugging Face paths-info API. Does not read local model files or checksums.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import WEIGHTS_FILE
from weights import parse_hf_url

UA = "ubuntuai-installer-catalog-drift/0.1"
TIMEOUT = 30

UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class SizeCheck:
    id: str
    url: str
    catalog_bytes: int
    published_bytes: int | None
    resolved: bool
    source: str
    error: str

    def ok(self) -> bool:
        return (
            self.resolved
            and self.published_bytes is not None
            and self.published_bytes == self.catalog_bytes
            and not self.error
        )


def _positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    if not text.isdigit():
        return None
    n = int(text)
    return n if n > 0 else None


def size_from_headers(headers: Any) -> tuple[int | None, str]:
    linked = _positive_int(headers.get("x-linked-size"))
    if linked is not None:
        return linked, "x-linked-size"
    length = _positive_int(headers.get("Content-Length"))
    if length is not None:
        return length, "content-length"
    return None, ""


def size_from_hf_row(row: object) -> int | None:
    if not isinstance(row, dict):
        return None
    lfs = row.get("lfs") or {}
    if isinstance(lfs, dict):
        n = _positive_int(lfs.get("size"))
        if n is not None:
            return n
    return _positive_int(row.get("size"))


def _request(url: str, *, method: str = "GET", data: bytes | None = None) -> urllib.request.Request:
    headers = {"User-Agent": UA}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def _open(req: urllib.request.Request, urlopen: UrlOpen) -> Any:
    return urlopen(req, timeout=TIMEOUT)


class _CaptureRedirect(urllib.request.HTTPRedirectHandler):
    """Keep x-linked-size from the first hop. Xet CDNs drop that header."""

    def __init__(self) -> None:
        self.sizes: list[tuple[int, str]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        n, src = size_from_headers(headers)
        if n is not None:
            self.sizes.append((n, src))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def default_urlopen(req: urllib.request.Request, timeout: int = TIMEOUT) -> Any:
    capture = _CaptureRedirect()
    opener = urllib.request.build_opener(capture)
    resp = opener.open(req, timeout=timeout)
    if capture.sizes and not size_from_headers(resp.headers)[0]:
        n, _src = capture.sizes[0]
        resp.headers["x-linked-size"] = str(n)
    return resp


def resolve_url(url: str, urlopen: UrlOpen) -> tuple[bool, int | None, str, str]:
    """HEAD the catalog URL. Fall back to a one-byte GET if HEAD is refused."""
    try:
        with _open(_request(url, method="HEAD"), urlopen) as resp:
            n, src = size_from_headers(resp.headers)
            if 200 <= int(resp.status) < 400:
                return True, n, src, ""
            return False, n, src, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {405, 501}:
            return _resolve_via_get(url, urlopen)
        return False, None, "", f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, None, "", str(exc.reason if isinstance(exc, urllib.error.URLError) else exc)


def _resolve_via_get(url: str, urlopen: UrlOpen) -> tuple[bool, int | None, str, str]:
    req = _request(url, method="GET")
    req.add_header("Range", "bytes=0-0")
    try:
        with _open(req, urlopen) as resp:
            n, src = size_from_headers(resp.headers)
            if n is None:
                cr = resp.headers.get("Content-Range") or ""
                if "/" in cr:
                    n = _positive_int(cr.rsplit("/", 1)[-1])
                    if n is not None:
                        src = "content-range"
            if 200 <= int(resp.status) < 400:
                return True, n, src, ""
            return False, n, src, f"HTTP {resp.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, None, "", str(exc.reason if isinstance(exc, urllib.error.URLError) else exc)


def hf_published_size(url: str, urlopen: UrlOpen) -> tuple[int | None, str]:
    parsed = parse_hf_url(url)
    if parsed is None:
        return None, ""
    repo, filename = parsed
    api = f"https://huggingface.co/api/models/{repo}/paths-info/main"
    body = json.dumps({"paths": [filename], "expand": True}).encode("utf-8")
    try:
        with _open(_request(api, method="POST", data=body), urlopen) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None, ""
    if not isinstance(rows, list):
        return None, ""
    for row in rows:
        n = size_from_hf_row(row)
        if n is not None:
            return n, "lfs.size"
    return None, ""


def check_model(row: dict, urlopen: UrlOpen) -> SizeCheck:
    model_id = str(row.get("id") or "")
    url = str(row.get("url") or "")
    catalog = _positive_int(row.get("bytes")) or 0
    if not model_id or not url:
        return SizeCheck(model_id, url, catalog, None, False, "", "missing id or url")
    resolved, header_n, header_src, resolve_err = resolve_url(url, urlopen)
    if not resolved:
        return SizeCheck(model_id, url, catalog, None, False, "", resolve_err or "URL did not resolve")
    hf_n, hf_src = hf_published_size(url, urlopen)
    sizes = []
    if hf_n is not None:
        sizes.append((hf_n, hf_src or "lfs.size"))
    if header_n is not None:
        sizes.append((header_n, header_src or "content-length"))
    if not sizes:
        return SizeCheck(model_id, url, catalog, None, True, "", "upstream size missing")
    unique = {n for n, _src in sizes}
    if len(unique) > 1:
        detail = " ".join(f"{src} {n}" for n, src in sizes)
        return SizeCheck(model_id, url, catalog, None, True, "", f"upstream sizes disagree ({detail})")
    published, source = sizes[0]
    return SizeCheck(model_id, url, catalog, published, True, source, "")


def check_catalog(path: Path, urlopen: UrlOpen) -> tuple[SizeCheck, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(check_model(row, urlopen) for row in raw["models"])


def format_report(checks: tuple[SizeCheck, ...]) -> str:
    bad = [c for c in checks if not c.ok()]
    if not bad:
        return f"catalog bytes match upstream ({len(checks)} models)"
    lines = ["catalog bytes drifted from upstream"]
    for i, c in enumerate(bad, start=1):
        if not c.resolved:
            lines.append(f"{i}. {c.id} URL did not resolve ({c.error})")
            lines.append(f"   {c.url}")
            continue
        if c.published_bytes is None:
            lines.append(f"{i}. {c.id} {c.error}")
            lines.append(f"   catalog {c.catalog_bytes} {c.url}")
            continue
        lines.append(
            f"{i}. {c.id} catalog {c.catalog_bytes} upstream {c.published_bytes}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, urlopen: UrlOpen = default_urlopen) -> int:
    parser = argparse.ArgumentParser(description="Compare weights.json bytes to upstream.")
    parser.add_argument("--catalog", type=Path, default=WEIGHTS_FILE)
    args = parser.parse_args(argv)
    checks = check_catalog(args.catalog, urlopen)
    print(format_report(checks))
    return 0 if all(c.ok() for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
