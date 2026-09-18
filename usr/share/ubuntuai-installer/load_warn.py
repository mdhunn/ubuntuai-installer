"""Dialog copy for oversized Lemonade pre-load.

Risk math stays in lemonade.load_risk. This module fills the locked
GTK and Qt English. Policy is warn-only. Callers must keep Continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from domain import Hardware, UserTarget
from lemonade import (
    LOAD_RISK_POLICY,
    largest_gguf_bytes,
    load_risk,
    load_tuning,
    model_gguf_bytes,
    risk_english,
)
from weights import human_bytes

STRONG_TITLE = "This model may strain this computer"
STRONG_BODY = (
    "Largest file is {size}. This computer has about {ram}. "
    "Loading a model this big over Vulkan can freeze the screen or log you out. "
    "You can still continue. Lemonade will keep one model loaded, use a smaller "
    "context, and memory-map the file."
)
STRONG_PRIMARY = "Continue anyway"

SOFT_TITLE = "Large model on this computer"
SOFT_BODY = (
    "Largest file is {size} on about {ram} of RAM. "
    "That is a large share of memory. You can still continue. "
    "Lemonade will keep one model loaded and use a smaller context."
)
SOFT_PRIMARY = "Continue"

CANCEL = "Cancel"


@dataclass(frozen=True)
class LoadWarn:
    level: str
    title: str
    body: str
    primary: str
    policy: str
    action: str
    bytes: int
    ram_bytes: int
    cli: str = ""

    @property
    def should_prompt(self) -> bool:
        return self.level in {"warn", "strong"}


def warn_copy(risk: dict[str, object], ram_bytes: int) -> LoadWarn:
    """Locked dialog English. risk_english stays the CLI string."""
    level = str(risk.get("level") or "ok")
    size_n = int(risk.get("bytes") or 0)
    ram_n = int(ram_bytes or 0)
    size = human_bytes(size_n)
    ram = human_bytes(ram_n) if ram_n else "this machine's RAM"
    cli = risk_english(risk, ram_bytes=ram_n)
    policy = str(risk.get("policy") or LOAD_RISK_POLICY)
    action = str(risk.get("action") or "warn")
    if level == "strong":
        return LoadWarn(
            level=level,
            title=STRONG_TITLE,
            body=STRONG_BODY.format(size=size, ram=ram),
            primary=STRONG_PRIMARY,
            policy=policy,
            action=action,
            bytes=size_n,
            ram_bytes=ram_n,
            cli=cli,
        )
    if level == "warn":
        return LoadWarn(
            level=level,
            title=SOFT_TITLE,
            body=SOFT_BODY.format(size=size, ram=ram),
            primary=SOFT_PRIMARY,
            policy=policy,
            action=action,
            bytes=size_n,
            ram_bytes=ram_n,
            cli=cli,
        )
    return LoadWarn(
        level="ok",
        title="",
        body="",
        primary="",
        policy=policy,
        action=action,
        bytes=size_n,
        ram_bytes=ram_n,
        cli=cli,
    )


def warn_for_bytes(
    largest: int, ram_bytes: int, backend: str = ""
) -> LoadWarn:
    ram = max(int(ram_bytes or 0), 1)
    frac = int(largest or 0) / ram
    return warn_copy(load_risk(frac, int(largest or 0), backend), ram_bytes)


def _backend_for(hw: Hardware, largest: int) -> str:
    settings = load_tuning(hw, largest)
    return str(settings.get("llamacpp_backend") or "")


def warn_for_publish(target: UserTarget, hw: Hardware) -> LoadWarn:
    largest = largest_gguf_bytes(target)
    return warn_for_bytes(largest, hw.ram_bytes, _backend_for(hw, largest))


def warn_for_chat_model(
    target: UserTarget, hw: Hardware, model_name: str = ""
) -> LoadWarn:
    name = (model_name or "").strip()
    if not name or name == "auto":
        return warn_for_publish(target, hw)
    path = Path(target.model_root) / "gguf" / name
    largest = model_gguf_bytes(path)
    return warn_for_bytes(largest, hw.ram_bytes, _backend_for(hw, largest))


def plan_publishes_lemonade(actions) -> bool:
    return any(getattr(action, "kind", "") == "lemonade" for action in actions)
