"""Load workflows.json. This is the only catalog parser."""

from __future__ import annotations

import json
from pathlib import Path

from domain import Hardware, Workflow
from paths import WORKFLOWS_FILE

SPEECH_ALIAS = "ubuntuai-speech"


def load_workflows(path: Path | None = None) -> tuple[Workflow, ...]:
    src = path or WORKFLOWS_FILE
    raw = json.loads(src.read_text(encoding="utf-8"))
    items = []
    seen: set[str] = set()
    for row in raw["workflows"]:
        wf = Workflow(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            default=bool(row["default"]),
            always=bool(row["always"]),
            requires=tuple(row.get("requires") or ()),
            apt=tuple(row.get("apt") or ()),
            groups=tuple(row.get("groups") or ()),
            model_subdirs=tuple(row.get("model_subdirs") or ()),
            needs_any_backend=tuple(row.get("needs_any_backend") or ()),
            hide_unless_backend=tuple(row.get("hide_unless_backend") or ()),
            ports=tuple(int(p) for p in (row.get("ports") or ())),
            notes=row.get("notes") or "",
            role=row.get("role") or "",
            priority=int(row.get("priority") or 0),
            min_ram_bytes=int(row.get("min_ram_bytes") or 0),
            runtime_bins=tuple(row.get("runtime_bins") or ()),
            vendor=row.get("vendor") or "",
            required_weights=tuple(row.get("required_weights") or ()),
        )
        if wf.id in seen:
            raise ValueError(f"duplicate workflow id {wf.id}")
        seen.add(wf.id)
        items.append(wf)
    if "ubuntuai-core" not in seen:
        raise ValueError("catalog missing ubuntuai-core")
    return tuple(items)


def by_id(workflows: tuple[Workflow, ...]) -> dict[str, Workflow]:
    return {w.id: w for w in workflows}


def pick_role_winners(
    workflows: tuple[Workflow, ...], hw: Hardware
) -> dict[str, str]:
    winners: dict[str, str] = {}
    for role in ("tts", "stt"):
        cands = [w for w in workflows if w.role == role and w.eligible(hw)]
        if not cands:
            continue
        best = max(cands, key=lambda w: (w.runtime_ok(), w.priority))
        winners[role] = best.id
    return winners


def recommended_ids(
    workflows: tuple[Workflow, ...], hw: Hardware
) -> frozenset[str]:
    rec = set(pick_role_winners(workflows, hw).values())
    for w in workflows:
        if not w.role and w.default and w.offered(hw) and w.satisfied(hw):
            rec.add(w.id)
        if w.always:
            rec.add(w.id)
    return frozenset(rec)


def expand_selection(
    wanted: tuple[str, ...],
    workflows: tuple[Workflow, ...],
    hw: Hardware | None = None,
) -> tuple[str, ...]:
    index = by_id(workflows)
    out: list[str] = []

    def add(wid: str) -> None:
        if wid in out:
            return
        if wid == SPEECH_ALIAS:
            if hw is None:
                for w in workflows:
                    if w.role in {"tts", "stt"}:
                        add(w.id)
                return
            for rec in pick_role_winners(workflows, hw).values():
                add(rec)
            return
        wf = index.get(wid)
        if wf is None:
            raise KeyError(f"unknown workflow {wid}")
        for req in wf.requires:
            add(req)
        out.append(wid)

    add("ubuntuai-core")
    for wid in wanted:
        add(wid)
    return tuple(out)
