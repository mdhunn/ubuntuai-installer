"""Checks against the real machine. Importing this module is not a pass."""

from __future__ import annotations

import grp
import os
import pwd
import shutil
from pathlib import Path

from apply import dpkg_installed, user_in_group
from domain import Check, Hardware, UserTarget
from paths import ENV_FILE, LIMITS_FILE
from probe import probe
from users import target_for
from weights import store_inventory


def _ok(name: str, detail: str) -> Check:
    return Check(name, "ok", detail)


def _warn(name: str, detail: str) -> Check:
    return Check(name, "warn", detail)


def _fail(name: str, detail: str) -> Check:
    return Check(name, "fail", detail)


def _readable_by_user(path: Path, user: str) -> bool:
    if not path.exists():
        return False
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return False
    st = path.stat()
    mode = st.st_mode
    if st.st_uid == pw.pw_uid and mode & 0o400:
        return True
    if st.st_gid == pw.pw_gid and mode & 0o040:
        return True
    try:
        g = grp.getgrgid(st.st_gid)
        if user in g.gr_mem and mode & 0o040:
            return True
    except KeyError:
        pass
    if mode & 0o004:
        return True
    # POSIX ACL + sign. Treat as usable if the current process can open it
    # and the user is the seated owner. Group membership is the durable check.
    return False


def collect(target: UserTarget, hw: Hardware | None = None) -> tuple[Check, ...]:
    hw = hw or probe()
    checks: list[Check] = []
    checks.append(
        _ok(
            "cpu",
            f"{hw.cpu_name}, {hw.ram_bytes // (1024**3)} GiB RAM",
        )
    )
    backends = ", ".join(sorted(hw.backends()))
    checks.append(_ok("backends", backends))
    for d in hw.devices:
        if d.kind == "cpu":
            continue
        if d.node and Path(d.node).exists():
            checks.append(_ok(f"device:{d.kind}", f"{d.name} at {d.node} ({d.backend})"))
        else:
            checks.append(
                _fail(f"device:{d.kind}", f"{d.name} has no device node ({d.detail})")
            )
        if d.backend == "xdna" and "1.0.0.166" in d.detail:
            checks.append(
                _warn(
                    "npu-firmware",
                    d.detail
                    + ". FastFlowLM wants >= 1.1.0.0. Installer will not rewrite firmware.",
                )
            )
        elif d.backend == "xdna":
            checks.append(_ok("npu-firmware", d.detail or "present"))

    for g in ("render", "video"):
        if user_in_group(target.name, g):
            checks.append(_ok(f"group:{g}", f"{target.name} in {g}"))
        else:
            checks.append(
                _fail(
                    f"group:{g}",
                    f"{target.name} is not in {g}. Session ACLs may still work. Group is the durable fix.",
                )
            )

    for node in ("/dev/dri/renderD128", "/dev/kfd", "/dev/accel/accel0"):
        p = Path(node)
        if not p.exists():
            continue
        if _readable_by_user(p, target.name) or user_in_group(target.name, "render"):
            checks.append(_ok(f"access:{p.name}", str(p)))
        else:
            checks.append(_warn(f"access:{p.name}", f"{p} exists. {target.name} may need render."))

    if LIMITS_FILE.is_file():
        text = LIMITS_FILE.read_text(encoding="utf-8")
        if "memlock" in text:
            checks.append(_ok("memlock-file", str(LIMITS_FILE)))
        else:
            checks.append(_fail("memlock-file", f"{LIMITS_FILE} has no memlock"))
    else:
        checks.append(_warn("memlock-file", f"{LIMITS_FILE} not written yet"))

    if ENV_FILE.is_file():
        checks.append(_ok("env", str(ENV_FILE)))
    else:
        checks.append(_warn("env", f"{ENV_FILE} not written yet"))

    if target.model_root.exists():
        if os.access(target.model_root, os.W_OK):
            checks.append(_ok("model-root", str(target.model_root)))
        else:
            checks.append(
                _warn(
                    "model-root",
                    f"{target.model_root} exists but is not writable here",
                )
            )
        inv = store_inventory(target.model_root)
        if inv:
            detail = ", ".join(f"{k}={v}" for k, v in inv.items())
            checks.append(_ok("store-formats", detail))
        else:
            checks.append(_warn("store-formats", "model root has no classified weights yet"))
    else:
        checks.append(_warn("model-root", f"{target.model_root} does not exist yet"))

    for extra in target.extra_model_paths:
        if extra.exists():
            checks.append(_ok("extra-models", str(extra)))

    if shutil.which("llama-server") or dpkg_installed("llama.cpp-tools"):
        checks.append(_ok("llama.cpp", "llama.cpp-tools present"))
    else:
        checks.append(_warn("llama.cpp", "llama.cpp-tools not installed"))

    if dpkg_installed("libggml0-backend-vulkan"):
        checks.append(_ok("ggml-vulkan", "libggml0-backend-vulkan present"))
    elif "vulkan" in hw.backends():
        checks.append(_warn("ggml-vulkan", "Vulkan GPU found. ggml Vulkan backend not installed"))

    if shutil.which("flm"):
        checks.append(_ok("fastflowlm", shutil.which("flm") or "flm"))
    elif "xdna" in hw.backends():
        checks.append(
            _warn(
                "fastflowlm",
                "XDNA2 present. flm is not on PATH. Hybrid engine is missing.",
            )
        )

    if shutil.which("whisper-cli") or dpkg_installed("whisper.cpp"):
        checks.append(_ok("whisper.cpp", shutil.which("whisper-cli") or "whisper.cpp"))
    else:
        checks.append(_warn("whisper.cpp", "whisper-cli not installed"))

    if shutil.which("RHVoice-test") or dpkg_installed("rhvoice"):
        checks.append(_ok("rhvoice", shutil.which("RHVoice-test") or "rhvoice"))

    if shutil.which("espeak-ng") or dpkg_installed("espeak-ng"):
        checks.append(_ok("espeak-ng", shutil.which("espeak-ng") or "espeak-ng"))

    moss_bin = next(
        (p for p in ("moss-tts-cli", "moss-tts-server", "moss-tts") if shutil.which(p)),
        None,
    )
    moss_dir = target.model_root / "openmoss"
    if moss_bin:
        checks.append(_ok("openmoss", shutil.which(moss_bin) or moss_bin))
    elif moss_dir.exists():
        checks.append(
            _warn(
                "openmoss",
                "Speech dir exists. moss-tts-cli is not on PATH. OpenMOSS runtime is missing.",
            )
        )

    for n in hw.notes:
        checks.append(_warn("note", n))
    return tuple(checks)


def worst(checks: tuple[Check, ...]) -> str:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "ok"


def format_checks(checks: tuple[Check, ...]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{c.status.upper():4}  {c.name}: {c.detail}")
    return "\n".join(lines)


def format_health(checks: tuple[Check, ...]) -> str:
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    if not fails and not warns:
        return "This computer looks ready for local AI."
    lines: list[str] = []
    if fails:
        lines.append("These need a fix:")
        for c in fails:
            lines.append(f"- {c.detail}")
    if warns:
        if lines:
            lines.append("")
        lines.append("These are worth a look:")
        for c in warns:
            lines.append(f"- {c.detail}")
    lines.append("")
    lines.append("Open Ubuntu AI Installer and use Repair if you want a plan.")
    return "\n".join(lines)


def run_for_user(user: str, model_root: Path | None = None) -> tuple[Check, ...]:
    return collect(target_for(user, model_root))
