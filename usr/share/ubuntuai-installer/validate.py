"""Checks against the real machine. Importing this module is not a pass."""

from __future__ import annotations

import grp
import os
import pwd
import shutil
from pathlib import Path

from apply import dpkg_installed, user_in_group
from catalog import by_id, load_workflows
from domain import Check, Hardware, UserTarget, Workflow
from paths import ENV_FILE, LIMITS_FILE
from probe import (
    APT_NAME,
    apt_cache_policy,
    apt_candidate,
    apt_sources_text,
    probe,
    split_apt_policies,
    universe_in_sources,
)
from users import target_for
from weights import store_inventory

CANARY_WORKFLOW_IDS = ("ubuntuai-core", "ubuntuai-chat", "ubuntuai-stt-whisper")


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


def canary_apt_names(workflows: tuple[Workflow, ...] | None = None) -> tuple[str, ...]:
    catalog = workflows or load_workflows()
    index = by_id(catalog)
    names: list[str] = []
    for wid in CANARY_WORKFLOW_IDS:
        wf = index.get(wid)
        if wf is None:
            continue
        for pkg in wf.apt:
            if APT_NAME.match(pkg) and pkg not in names:
                names.append(pkg)
    return tuple(names)


def _policy_blocks(
    names: tuple[str, ...],
    policy: str | dict[str, str] | None,
) -> dict[str, str] | None:
    if isinstance(policy, dict):
        return {name: policy.get(name, "") for name in names}
    if policy is not None:
        return split_apt_policies(policy)
    if shutil.which("apt-cache") is None:
        return None
    return split_apt_policies(apt_cache_policy(names))


def archive_gate_checks(
    *,
    policy_text: str | dict[str, str] | None = None,
    sources_text: str | None = None,
    workflows: tuple[Workflow, ...] | None = None,
) -> tuple[Check, ...]:
    names = canary_apt_names(workflows)
    blocks = _policy_blocks(names, policy_text)
    if blocks is None:
        return (
            _warn(
                "apt-cache",
                "apt-cache is not on PATH. Archive preflight was skipped.",
            ),
        )
    have_universe = universe_in_sources(apt_sources_text(sources_text))
    checks: list[Check] = []
    unknown: list[str] = []
    for name in names:
        candidate = apt_candidate(blocks.get(name, ""))
        if candidate:
            checks.append(
                _ok(
                    f"apt-known:{name}",
                    f"apt knows {name}. Candidate {candidate} is visible.",
                )
            )
            continue
        unknown.append(name)
        if have_universe:
            detail = (
                f"apt has no candidate for {name}. "
                "Refresh apt lists or confirm this Ubuntu suite publishes the package."
            )
        else:
            detail = (
                f"apt has no candidate for {name}. "
                "Enable the universe archive or refresh apt lists for this Ubuntu suite."
            )
        checks.append(_fail(f"apt-known:{name}", detail))
    if len(unknown) >= 2 and not have_universe:
        checks.append(
            _warn(
                "apt-universe",
                "apt does not know several catalog packages. "
                "Universe is missing from apt sources. "
                "Enable the universe archive and run apt update.",
            )
        )
    return tuple(checks)


def collect(
    target: UserTarget,
    hw: Hardware | None = None,
    *,
    apt_policy: str | dict[str, str] | None = None,
    apt_sources: str | None = None,
) -> tuple[Check, ...]:
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
    checks.extend(
        archive_gate_checks(policy_text=apt_policy, sources_text=apt_sources)
    )
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
