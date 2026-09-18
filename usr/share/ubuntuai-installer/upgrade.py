"""Human-in-the-loop upgrades for vendor runtimes, catalog models, and large-model load."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from lemonade import (
    check_model_updates,
    detect as lemonade_detect,
    largest_gguf_bytes,
    load_risk,
    load_tuning,
    report_load_tuning,
    risk_english,
)
from probe import probe
from users import target_for
from vendor import (
    install_vendor,
    load_vendors,
    pick_archive,
    recorded_vendor_version,
    vendor_present,
)
from weights import (
    UA,
    catalog_dest,
    download,
    human_bytes,
    load_catalog,
    lookup_published_hash,
)

ALLOW_KINDS = frozenset(
    {
        "note",
        "upgrade_vendor",
        "redownload",
        "lemonade_optimize",
        "lemonade_update_models",
    }
)
STEP_KEYS = frozenset(
    {"kind", "id", "why", "text", "url", "version", "algo", "hash", "sha256"}
)


def plan_path(home: Path) -> Path:
    return home / ".config" / "ubuntuai" / "upgrade-plan.json"


def _http_json(url: str) -> object | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def github_latest(repo: str) -> dict | None:
    data = _http_json(f"https://api.github.com/repos/{repo}/releases/latest")
    return data if isinstance(data, dict) else None


def _asset_url(release: dict, filename: str) -> str:
    want = Path(filename).name.lower()
    for asset in release.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").lower()
        if name == want:
            return str(asset.get("browser_download_url") or "")
    stem = want.replace(".tar.gz", "").replace(".tgz", "")
    for asset in release.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").lower()
        if stem and stem in name and name.endswith((".tar.gz", ".tgz")):
            return str(asset.get("browser_download_url") or "")
    return ""


def diagnose(user: str) -> dict:
    t = target_for(user)
    hw = probe()
    vendors = []
    table = load_vendors()
    for vid, spec in table.items():
        pinned = str(spec.get("version") or "")
        installed = recorded_vendor_version(t.home, vid) or (
            pinned if vendor_present(dict(spec, id=vid), t.home) else ""
        )
        latest = pinned
        asset = ""
        repo = str(spec.get("github") or "")
        release = github_latest(repo) if repo else None
        if release:
            latest = str(release.get("tag_name") or latest)
            archive = pick_archive(dict(spec, id=vid), hw)
            asset = _asset_url(release, str(archive.get("filename") or ""))
        vendors.append(
            {
                "id": vid,
                "title": spec.get("title") or vid,
                "pinned": pinned,
                "installed": installed,
                "latest": latest,
                "newer": bool(latest and installed and latest != installed)
                or bool(latest and pinned and latest != pinned and vendor_present(dict(spec, id=vid), t.home)),
                "url": asset,
                "present": vendor_present(dict(spec, id=vid), t.home),
            }
        )
    models = []
    for model in load_catalog():
        dest = catalog_dest(model, t.model_root)
        present = dest.exists() or dest.is_symlink()
        published = model.published_hash()
        if present and published is None and model.bytes < 512 * 1024 * 1024:
            published = lookup_published_hash(model.url)
        row = {
            "id": model.id,
            "title": model.title,
            "present": present,
            "bytes": model.bytes,
            "stale": False,
            "algo": published.algo if published else "",
            "hash": published.hexdigest if published else "",
        }
        if present and published:
            from weights import hash_path

            try:
                actual = hash_path(dest, published.algo)
            except (OSError, ValueError):
                actual = ""
            row["stale"] = bool(actual and actual.lower() != published.hexdigest.lower())
        models.append(row)
    largest = largest_gguf_bytes(t)
    tuning = load_tuning(hw, largest) if lemonade_detect() else {}
    lemon_updates = check_model_updates() if lemonade_detect() else ""
    risk: dict = {}
    if lemonade_detect():
        frac = largest / max(int(hw.ram_bytes) or 1, 1)
        risk = load_risk(
            frac, largest, str(tuning.get("llamacpp_backend") or "")
        )
    return {
        "user": t.name,
        "ram_bytes": hw.ram_bytes,
        "largest_gguf_bytes": largest,
        "vendors": vendors,
        "models": models,
        "lemonade": lemonade_detect(),
        "tuning": tuning,
        "load_risk": risk,
        "lemonade_updates": lemon_updates,
    }


def classical_plan(diag: dict) -> dict:
    steps: list[dict] = []
    for row in diag.get("vendors") or []:
        if row.get("newer") and row.get("url"):
            steps.append(
                {
                    "kind": "upgrade_vendor",
                    "id": row["id"],
                    "version": row.get("latest") or "",
                    "url": row.get("url") or "",
                    "why": (
                        f"{row.get('title') or row['id']} is {row.get('installed') or row.get('pinned')}. "
                        f"{row.get('latest')} is on GitHub."
                    ),
                }
            )
        elif row.get("present") and not row.get("newer"):
            steps.append(
                {
                    "kind": "note",
                    "text": f"{row.get('title') or row['id']} is current ({row.get('installed') or row.get('pinned')}).",
                }
            )
        elif not row.get("present"):
            steps.append(
                {
                    "kind": "note",
                    "text": f"{row.get('title') or row['id']} is not installed. Check it on Workflows if you want it.",
                }
            )
    for row in diag.get("models") or []:
        if row.get("stale"):
            steps.append(
                {
                    "kind": "redownload",
                    "id": row["id"],
                    "algo": row.get("algo") or "",
                    "hash": row.get("hash") or "",
                    "why": f"{row.get('title') or row['id']} no longer matches the published checksum.",
                }
            )
    if diag.get("lemonade"):
        tuning = diag.get("tuning") or {}
        largest = int(diag.get("largest_gguf_bytes") or 0)
        ram = int(diag.get("ram_bytes") or 0)
        if tuning:
            why = (
                f"Largest GGUF tree is {human_bytes(largest)} on {human_bytes(ram)} RAM. "
                f"Set context {tuning.get('ctx_size')}, timeout {tuning.get('global_timeout')}s, "
                f"backend {tuning.get('llamacpp_backend')}."
            )
            if tuning.get("llamacpp_args"):
                why += f" Memory-map with {tuning['llamacpp_args']}."
            steps.append({"kind": "lemonade_optimize", "why": why})
        risk = diag.get("load_risk") or {}
        if risk.get("level") in {"warn", "strong"}:
            text = risk_english(risk, ram_bytes=ram)
            if text:
                steps.append({"kind": "note", "text": text})
        text = str(diag.get("lemonade_updates") or "")
        low = text.lower()
        if text and any(tok in low for tok in ("update", "newer", "available")) and "no update" not in low:
            steps.append(
                {
                    "kind": "lemonade_update_models",
                    "why": "Lemonade reported model updates.",
                    "text": text[:1000],
                }
            )
    if not any(s.get("kind") != "note" for s in steps):
        steps.append(
            {
                "kind": "note",
                "text": "No vendor, catalog, or Lemonade load changes are waiting.",
            }
        )
    acting = sum(1 for s in steps if s.get("kind") != "note")
    return {
        "summary": (
            "Updates are ready for your approval."
            if acting
            else "Nothing needs an update."
        ),
        "source": "classical",
        "steps": steps,
        "tuning": diag.get("tuning") or {},
    }


def _sanitize(plan: dict) -> dict:
    steps = []
    for item in plan.get("steps") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind not in ALLOW_KINDS:
            steps.append(
                {"kind": "note", "text": f"Dropped unsupported step ({kind})."}
            )
            continue
        steps.append({k: item[k] for k in item if k in STEP_KEYS})
    return {
        "summary": str(plan.get("summary") or "Upgrade plan"),
        "source": str(plan.get("source") or "classical"),
        "steps": steps,
        "tuning": plan.get("tuning") or {},
    }


def step_english(step: dict) -> str:
    kind = step.get("kind")
    if kind == "note":
        return str(step.get("text") or "A note.")
    if kind == "upgrade_vendor":
        return (
            f"Upgrade {step.get('id')} to {step.get('version') or 'the latest GitHub release'} "
            "in your home folder. This is not an apt or snap package."
        )
    if kind == "redownload":
        return (
            f"Download {step.get('id')} again. The copy on disk does not match the published checksum."
        )
    if kind == "lemonade_optimize":
        return (
            "Tune Lemonade for large GGUF files: one model in memory, a bounded context, "
            "a longer first-load timeout, and the GPU backend this machine already has."
        )
    if kind == "lemonade_update_models":
        return "Ask Lemonade to download newer copies of the models it already tracks."
    return f"Skip an unsupported step ({kind})."


def format_plan(plan: dict) -> str:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    acting = [s for s in steps if s.get("kind") != "note"]
    notes = [s for s in steps if s.get("kind") == "note"]
    lines = ["What this means", ""]
    if not acting:
        lines.append("Nothing will be upgraded until a later Diagnose finds a change.")
    else:
        lines.append(
            "The installer found vendor, model, or Lemonade load updates. "
            "Nothing will change until you approve this plan."
        )
        lines.append("")
        lines.append("If you approve, the installer will:")
        for step in acting:
            lines.append(f"- {step_english(step)}")
    if notes:
        lines.append("")
        lines.append("Notes:")
        for step in notes:
            lines.append(f"- {step_english(step)}")
    lines.append("")
    lines.append("You can approve, or do nothing.")
    lines.append("")
    lines.append("Steps")
    if not steps:
        lines.append("1. Nothing to do.")
        return "\n".join(lines)
    for i, step in enumerate(steps, 1):
        extra = step.get("why") or ""
        line = f"{i}. {step_english(step)}"
        if extra and extra not in line:
            line += f" {extra}"
        lines.append(line)
    return "\n".join(lines)


def build_plan_for(user: str) -> tuple[dict, dict]:
    diag = diagnose(user)
    plan = _sanitize(classical_plan(diag))
    t = target_for(user)
    path = plan_path(t.home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"diagnosis": diag, "plan": plan, "english": format_plan(plan)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    try:
        os.chown(path, t.uid, t.gid)
        os.chmod(path, 0o600)
    except OSError:
        pass
    return diag, plan


def load_saved_plan(user: str) -> dict | None:
    path = plan_path(target_for(user).home)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def execute_plan_steps(user: str, plan: dict | None, *, dry_run: bool = False) -> list[str]:
    t = target_for(user)
    hw = probe()
    if plan is None:
        saved = load_saved_plan(user)
        plan = (saved or {}).get("plan")
    plan = _sanitize(plan or {})
    log: list[str] = []
    from domain import FileHash

    for step in plan.get("steps") or []:
        kind = step.get("kind")
        if kind == "note":
            log.append(step_english(step))
            continue
        if dry_run:
            log.append(f"would {kind} {step.get('id') or ''}".strip())
            continue
        if kind == "upgrade_vendor":
            vid = str(step.get("id") or "")
            log.append(
                install_vendor(
                    vid,
                    hw,
                    t,
                    force=True,
                    archive_url=str(step.get("url") or ""),
                    version=str(step.get("version") or ""),
                )
            )
            continue
        if kind == "redownload":
            catalog = {w.id: w for w in load_catalog()}
            model = catalog.get(step.get("id"))
            if model is None:
                log.append(f"unknown weight {step.get('id')}")
                continue
            expected = None
            hexdigest = str(step.get("hash") or step.get("sha256") or "")
            algo = str(step.get("algo") or "")
            if hexdigest and algo:
                expected = FileHash(algo, hexdigest.lower())
            log.append(
                download(
                    model,
                    t.model_root,
                    uid=t.uid,
                    gid=t.gid,
                    force=True,
                    expected=expected,
                )
            )
            continue
        if kind == "lemonade_optimize":
            text = report_load_tuning(t, hw)
            log.extend(line for line in text.splitlines() if line)
            continue
        if kind == "lemonade_update_models":
            exe = shutil.which("lemonade-server") or shutil.which("lemonade")
            if not exe:
                log.append("lemonade-server is not on PATH")
                continue
            import subprocess

            p = subprocess.run(
                [exe, "update-models"],
                check=False,
                capture_output=True,
                text=True,
            )
            log.append((p.stdout or p.stderr or "update-models finished").strip())
            continue
        log.append(f"skip unsupported {kind}")
    return log
