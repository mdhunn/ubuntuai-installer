"""Which installer workflows are present, and how a user talks to them."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from apply import dpkg_installed
from catalog import helper_workflow_ids, load_workflows
from configstore import load as load_config
from domain import Hardware, UserTarget, Workflow
from paths import ENV_FILE, LIMITS_FILE
from weights import SKIP_NAMES

RECORD_ONLY = frozenset(
    {
        "ubuntuai-image",
        "ubuntuai-video",
        "ubuntuai-rag",
        "ubuntuai-coding",
        "ubuntuai-tts-espeak",
    }
)

BACKEND_LABELS = {
    "auto": "Choose for me",
    "vulkan": "GPU (Vulkan)",
    "rocm": "AMD GPU (ROCm)",
    "cuda": "NVIDIA GPU (CUDA)",
    "xdna": "AMD NPU (XDNA)",
    "cpu": "CPU only",
}

CHAT_MODEL_CTA = "Download a chat model"
CHAT_MODEL_NEEDED = (
    "Chat has no GGUF file in the model folder yet. "
    "Download one from the Weights tab."
)
CHAT_MODEL_NEEDED_CONFIG = (
    "Chat has no GGUF file in the model folder yet. "
    "Open Ubuntu AI Installer. Use the Weights tab to download a chat model."
)
HELPERS_SECTION = "Helpers"
HELPERS_BLURB = (
    "These prepare folders and pointers. They are not full AI apps."
)


def listen_words(bind: str) -> str:
    if bind == "0.0.0.0":
        return "Other devices on your home network can connect."
    return "This computer only."


def machine_words(hw: Hardware) -> str:
    have = hw.backends()
    parts: list[str] = []
    if "cpu" in have:
        parts.append("CPU")
    if have & {"vulkan", "rocm", "cuda"}:
        parts.append("GPU")
    if "xdna" in have:
        parts.append("NPU")
    if not parts:
        text = "This machine can run local AI."
    elif len(parts) == 1:
        text = f"This machine runs on the {parts[0]}."
    elif len(parts) == 2:
        text = f"This machine can use the {parts[0]} and the {parts[1]}."
    else:
        text = "This machine can use the CPU, GPU, and NPU."
    if hw.hybrid_ok():
        text += " The NPU and GPU can work together."
    return text


def workflow_row_title(title: str, wid: str, helpers: frozenset[str]) -> str:
    if wid not in helpers:
        return title
    if "helper" in title.lower():
        return title
    return f"{title} (helpers only)"


@dataclass(frozen=True)
class AppStatus:
    id: str
    title: str
    present: bool
    endpoint: str
    summary: str
    role: str = ""


def recorded_ids(user: str) -> frozenset[str]:
    raw = load_config(user).get("installed_workflows") or []
    return frozenset(str(x) for x in raw if x)


def live_present(wf: Workflow, target: UserTarget) -> bool:
    if wf.id == "ubuntuai-core":
        return ENV_FILE.is_file() or LIMITS_FILE.is_file()
    if wf.id == "ubuntuai-chat":
        return bool(shutil.which("llama-server") or dpkg_installed("llama.cpp-tools"))
    if wf.id == "ubuntuai-hybrid":
        return bool(shutil.which("flm"))
    if wf.id == "ubuntuai-tts-openmoss":
        return bool(shutil.which("moss-tts-server") or shutil.which("ubuntuai-openmoss"))
    if wf.id == "ubuntuai-tts-rhvoice":
        return bool(shutil.which("RHVoice-test") or dpkg_installed("rhvoice"))
    if wf.id == "ubuntuai-stt-whisper":
        return bool(shutil.which("whisper-cli") or dpkg_installed("whisper.cpp"))
    if wf.id == "ubuntuai-serve":
        return bool(
            dpkg_installed("libggml0-backend-hip") or dpkg_installed("nvidia-cuda-toolkit")
        )
    if wf.id == "ubuntuai-train":
        return bool(shutil.which("hipcc") or dpkg_installed("nvidia-cuda-toolkit"))
    if wf.runtime_bins and any(shutil.which(b) for b in wf.runtime_bins):
        return True
    return False


def is_present(wf: Workflow, user: str, target: UserTarget) -> bool:
    recorded = wf.id in recorded_ids(user)
    if wf.id in RECORD_ONLY:
        return recorded
    return recorded or live_present(wf, target)


def endpoint_for(wf: Workflow, bind: str) -> str:
    if not wf.ports:
        return ""
    host = "127.0.0.1" if bind != "0.0.0.0" else bind
    port = wf.ports[0]
    if wf.id in {"ubuntuai-chat", "ubuntuai-coding", "ubuntuai-serve"}:
        return f"http://{host}:{port}/v1"
    return f"http://{host}:{port}"


def app_statuses(user: str, target: UserTarget) -> tuple[AppStatus, ...]:
    out: list[AppStatus] = []
    for wf in load_workflows():
        present = is_present(wf, user, target)
        out.append(
            AppStatus(
                id=wf.id,
                title=wf.title,
                present=present,
                endpoint=endpoint_for(wf, target.bind) if present else "",
                summary=wf.summary,
                role=wf.role,
            )
        )
    return tuple(out)


def chat_models(model_root: Path) -> tuple[str, ...]:
    folder = model_root / "gguf"
    if not folder.is_dir():
        return ()
    names: list[str] = []
    for p in sorted(folder.iterdir()):
        if p.name.lower() in SKIP_NAMES:
            continue
        if p.suffix.lower() == ".gguf" or p.is_dir():
            names.append(p.name)
    return tuple(names)


def no_chat_gguf(model_root: Path) -> bool:
    return not chat_models(model_root)


def chat_weight_ids(workflows: tuple[Workflow, ...] | None = None) -> frozenset[str]:
    wfs = workflows if workflows is not None else load_workflows()
    ids: set[str] = set()
    for wf in wfs:
        if wf.id == "ubuntuai-chat":
            ids.update(wf.required_weights)
    return frozenset(ids)


def backend_choices(hw: Hardware) -> tuple[str, ...]:
    order = ("vulkan", "rocm", "cuda", "xdna", "cpu")
    have = hw.installable_backends()
    picked = [b for b in order if b in have]
    return ("auto",) + tuple(picked)


def explain_setup(user: str, target: UserTarget, hw: Hardware) -> str:
    apps = app_statuses(user, target)
    installed = [a for a in apps if a.present]
    helpers = helper_workflow_ids(load_workflows())
    missing = [
        a
        for a in apps
        if not a.present and a.id != "ubuntuai-core" and a.id not in helpers
    ]
    cfg = load_config(user)
    bind = cfg.get("bind") or target.bind
    listen = (
        "this computer only"
        if bind != "0.0.0.0"
        else "this computer and other devices on your network"
    )
    lines = [
        f"Acting for {target.name}.",
        f"{hw.cpu_name}. {hw.ram_bytes // (1024**3)} GiB RAM.",
        f"Listen: {listen}.",
        f"Model folder: {target.model_root}.",
        "",
        "Installed",
    ]
    if not installed:
        lines.append("- Nothing from the installer is on this computer yet.")
    for app in installed:
        extra = f"  {app.endpoint}" if app.endpoint else ""
        lines.append(f"- {app.title}{extra}")
    chat = cfg.get("chat_model") or "choose for me"
    backend = cfg.get("primary_backend") or "auto"
    lines.extend(
        [
            "",
            f"Chat model: {chat}.",
            f"GPU mode: {BACKEND_LABELS.get(backend, backend)}.",
        ]
    )
    tts = cfg.get("tts_engine") or ""
    stt = cfg.get("stt_engine") or ""
    if tts:
        lines.append(f"Speech out: {tts}.")
    if stt:
        lines.append(f"Speech in: {stt}.")
    if missing:
        lines.extend(
            [
                "",
                "Not installed. Open Ubuntu AI Installer to add them.",
            ]
        )
        for app in missing:
            lines.append(f"- {app.title}")
    return "\n".join(lines)
