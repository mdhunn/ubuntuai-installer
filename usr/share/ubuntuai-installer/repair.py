"""Human-in-the-loop repair for config, models, and vendor installs."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from apply import (
    ApplyError,
    _technical_block,
    build_plan,
    execute_plan,
    explain_helper_failure,
    run_privileged,
)
from catalog import load_workflows
from configstore import load as load_config
from configstore import save as save_config
from domain import FileHash, UserTarget
from paths import user_config_path
from probe import probe
from users import target_for
from validate import collect
from vendor import install_vendor
from weights import (
    UA,
    catalog_dest,
    download,
    hash_path,
    load_catalog,
    lookup_published_hash,
    parse_named_hash,
    store_inventory,
)

ALLOW_KINDS = frozenset(
    {
        "note",
        "redownload",
        "reinstall_vendor",
        "write_core",
        "fix_bind",
        "hash_file",
        "install_workflow",
    }
)
STEP_KEYS = frozenset(
    {"kind", "id", "why", "text", "bind", "path", "algo", "hash", "sha256"}
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_checksum",
            "description": (
                "Look up the checksum published on a model download page. "
                "Returns algo:hex, for example sha256:abc or md5:def."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for checksums, install docs, or error messages.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def plan_path(home: Path) -> Path:
    return user_config_path(home).parent / "repair-plan.json"


def web_search(query: str) -> list[dict]:
    q = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    req = urllib.request.Request(
        f"https://api.duckduckgo.com/?{q}",
        headers={"User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    hits: list[dict] = []
    abstract = data.get("AbstractText") or ""
    abs_url = data.get("AbstractURL") or ""
    if abstract:
        hits.append({"title": data.get("Heading") or query, "url": abs_url, "snippet": abstract})
    for key in ("RelatedTopics", "Results"):
        for item in data.get(key) or []:
            if not isinstance(item, dict):
                continue
            text = item.get("Text") or ""
            first = item.get("FirstURL") or ""
            if text:
                hits.append({"title": text[:80], "url": first, "snippet": text})
            if len(hits) >= 5:
                return hits
    return hits[:5]


def checksum_report(target: UserTarget) -> list[dict]:
    rows: list[dict] = []
    for model in load_catalog():
        dest = catalog_dest(model, target.model_root)
        present = dest.exists() or dest.is_symlink()
        published = model.published_hash()
        if present and published is None:
            published = lookup_published_hash(model.url)
        row = {
            "id": model.id,
            "title": model.title,
            "path": str(dest),
            "url": model.url,
            "present": present,
            "algo": published.algo if published else "",
            "expected": published.hexdigest if published else "",
            "actual": None,
            "match": None,
        }
        if present and published:
            try:
                row["actual"] = hash_path(dest, published.algo)
            except (OSError, ValueError):
                row["actual"] = None
        if row["expected"] and row["actual"]:
            row["match"] = row["expected"].lower() == row["actual"].lower()
        rows.append(row)
    return rows


def diagnose(user: str) -> dict:
    t = target_for(user)
    hw = probe()
    checks = collect(t, hw)
    cfg = load_config(user)
    return {
        "user": t.name,
        "model_root": str(t.model_root),
        "bind": t.bind,
        "backends": sorted(hw.backends()),
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail} for c in checks
        ],
        "inventory": store_inventory(t.model_root),
        "checksums": checksum_report(t),
        "openai_base_url": cfg.get("openai_base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or "",
        "wrappers": {
            "moss-tts-server": shutil.which("moss-tts-server")
            or str(t.home / ".local/bin/moss-tts-server"),
            "ubuntuai-openmoss": str(t.home / ".local/bin/ubuntuai-openmoss"),
            "llama-server": shutil.which("llama-server") or "",
            "whisper-cli": shutil.which("whisper-cli") or "",
        },
    }


def classical_plan(diag: dict) -> dict:
    steps: list[dict] = []
    for check in diag.get("checks") or []:
        name = check.get("name") or ""
        status = check.get("status")
        detail = check.get("detail") or ""
        if status == "ok":
            continue
        if name == "whisper.cpp":
            steps.append(
                {
                    "kind": "install_workflow",
                    "id": "ubuntuai-stt-whisper",
                    "why": detail,
                }
            )
        elif name == "llama.cpp":
            steps.append(
                {
                    "kind": "install_workflow",
                    "id": "ubuntuai-chat",
                    "why": detail,
                }
            )
        elif name in {"openmoss", "fastflowlm"} and "missing" in detail.lower():
            if name == "openmoss":
                steps.append(
                    {"kind": "reinstall_vendor", "id": "openmoss", "why": detail}
                )
            else:
                steps.append({"kind": "note", "text": detail})
        elif name in {"env", "memlock-file"}:
            steps.append({"kind": "write_core", "why": detail})
        elif name == "npu-firmware":
            steps.append({"kind": "note", "text": detail})
        elif name.startswith("group:"):
            steps.append(
                {
                    "kind": "note",
                    "text": f"{detail} Re-run Apply on Workflows if groups are still missing.",
                }
            )
        else:
            text = detail if name == "note" else f"{name}: {detail}"
            steps.append({"kind": "note", "text": text})
    bind = diag.get("bind") or "127.0.0.1"
    if bind not in {"127.0.0.1", "0.0.0.0"}:
        steps.append({"kind": "fix_bind", "bind": "127.0.0.1", "why": f"illegal bind {bind}"})
    for row in diag.get("checksums") or []:
        if row.get("present") and row.get("expected") and row.get("match") is False:
            algo = row.get("algo") or "checksum"
            steps.append(
                {
                    "kind": "redownload",
                    "id": row["id"],
                    "algo": row.get("algo") or "",
                    "hash": row["expected"],
                    "why": f"{algo} mismatch for {row['path']}",
                }
            )
        elif row.get("present") and not row.get("expected"):
            steps.append(
                {
                    "kind": "note",
                    "text": (
                        f"No checksum was published on the download page for "
                        f"{row.get('title') or row['id']}."
                    ),
                }
            )
    if not steps:
        steps.append({"kind": "note", "text": "Classical checks found nothing to repair."})
    return {
        "summary": "Classical repair plan from validate, config, and checksum lookup.",
        "source": "classical",
        "steps": steps,
    }


def _run_tool(name: str, args: dict) -> str:
    if name == "lookup_checksum":
        url = str(args.get("url") or "")
        found = lookup_published_hash(url)
        return found.label() if found else "no published checksum found"
    if name == "web_search":
        hits = web_search(str(args.get("query") or ""))
        return json.dumps(hits[:5])
    return f"unknown tool {name}"


def _post_chat(base_url: str, api_key: str, body: dict) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _sanitize_plan(raw: dict, source: str) -> dict:
    steps_in = raw.get("steps") if isinstance(raw.get("steps"), list) else []
    steps: list[dict] = []
    for item in steps_in:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind not in ALLOW_KINDS:
            steps.append(
                {
                    "kind": "note",
                    "text": f"Dropped unsupported step {kind!r}.",
                }
            )
            continue
        steps.append({k: item[k] for k in item if k in STEP_KEYS})
    if not steps:
        steps.append({"kind": "note", "text": "Advisor returned no usable steps."})
    return {
        "summary": str(raw.get("summary") or "Advisor repair plan."),
        "source": source,
        "steps": steps,
    }


def advise_plan(
    diag: dict,
    *,
    base_url: str,
    api_key: str = "",
    source: str = "openai",
    comment: str = "",
) -> dict:
    system = (
        "You repair a local Ubuntu AI installer. Reply with JSON only: "
        '{"summary": str, "steps": [{"kind": "...", ...}]}. '
        "Allowed kind values: note, redownload, reinstall_vendor, write_core, fix_bind, hash_file, install_workflow. "
        "Do not invent shell commands. Prefer smallest safe steps. "
        "Use tools to look up the checksum algorithm and value from the download page."
    )
    user_msg = json.dumps(diag)
    if comment:
        user_msg += "\nHuman change request: " + comment
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    body = {
        "model": "local",
        "temperature": 0,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    try:
        data = _post_chat(base_url, api_key, body)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404, 422} and "tools" in body:
            body.pop("tools", None)
            body.pop("tool_choice", None)
            try:
                data = _post_chat(base_url, api_key, body)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as retry_exc:
                fallback = classical_plan(diag)
                fallback["summary"] = (
                    f"Advisor failed ({retry_exc}). Classical plan used instead."
                )
                fallback["source"] = "classical"
                return fallback
        else:
            fallback = classical_plan(diag)
            fallback["summary"] = f"Advisor failed ({exc}). Classical plan used instead."
            fallback["source"] = "classical"
            return fallback
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        fallback = classical_plan(diag)
        fallback["summary"] = f"Advisor failed ({exc}). Classical plan used instead."
        fallback["source"] = "classical"
        return fallback
    message = ((data.get("choices") or [{}])[0].get("message")) or {}
    rounds = 0
    while message.get("tool_calls") and rounds < 4:
        rounds += 1
        messages.append(message)
        for call in message.get("tool_calls") or []:
            fn = (call.get("function") or {})
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(name, args if isinstance(args, dict) else {})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "content": result,
                }
            )
        body["messages"] = messages
        try:
            data = _post_chat(base_url, api_key, body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            break
        message = ((data.get("choices") or [{}])[0].get("message")) or {}
    text = message.get("content") or ""
    parsed = _extract_json(text) if isinstance(text, str) else None
    if parsed is None:
        fallback = classical_plan(diag)
        fallback["summary"] = "Advisor did not return JSON. Classical plan used instead."
        return fallback
    return _sanitize_plan(parsed, source)


def detect_local_openai(cfg: dict) -> str:
    candidates = []
    if cfg.get("openai_base_url"):
        candidates.append(str(cfg["openai_base_url"]))
    candidates.extend(
        [
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:8081/v1",
            "http://127.0.0.1:1234/v1",
            "http://127.0.0.1:11434/v1",
        ]
    )
    for base in candidates:
        url = base.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return base.rstrip("/")
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return ""


def build_plan_for(
    user: str,
    advisor: str,
    *,
    comment: str = "",
    openai_base_url: str = "",
    openai_api_key: str = "",
) -> tuple[dict, dict]:
    diag = diagnose(user)
    cfg = load_config(user)
    base = (
        openai_base_url
        or cfg.get("openai_base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    )
    key = (
        openai_api_key
        or cfg.get("openai_api_key")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if advisor == "openai":
        if not base:
            plan = classical_plan(diag)
            plan["summary"] = (
                "No OpenAI URI was set. Enter a URI and API key, then diagnose again. "
                "A classical plan is shown instead."
            )
            plan["steps"].insert(
                0,
                {
                    "kind": "note",
                    "text": (
                        "The OpenAI path needs a server URI. Many providers also "
                        "need an API key. Fill both fields and diagnose again."
                    ),
                },
            )
        else:
            plan = advise_plan(diag, base_url=base, api_key=key, source="openai", comment=comment)
    elif advisor == "local":
        local = detect_local_openai(cfg) or base
        if not local:
            plan = classical_plan(diag)
            plan["summary"] = "No local OpenAI-compatible server. Classical plan used."
        else:
            plan = advise_plan(
                diag, base_url=local, api_key=key, source="local", comment=comment
            )
    else:
        plan = classical_plan(diag)
        if comment:
            plan["steps"].insert(
                0,
                {"kind": "note", "text": f"Human comment recorded: {comment}"},
            )
    t = target_for(user)
    path = plan_path(t.home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"diagnosis": diag, "plan": plan, "english": format_plan(plan)}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        os.chown(path, t.uid, t.gid)
        os.chmod(path, 0o600)
    except OSError:
        pass
    return diag, plan


def _workflow_title(wid: str) -> str:
    if not wid:
        return "a workflow"
    try:
        for wf in load_workflows():
            if wf.id == wid:
                return wf.title
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return wid


def step_english(step: dict) -> str:
    kind = step.get("kind")
    if kind == "note":
        return str(step.get("text") or "A note with no extra detail.")
    if kind == "redownload":
        algo = step.get("algo") or ""
        name = step.get("id") or "a model file"
        if algo:
            return (
                f"Download {name} again. After the download, check it with the "
                f"{algo} value published on its download page."
            )
        return (
            f"Download {name} again. After the download, check it with the "
            "checksum published on its download page."
        )
    if kind == "reinstall_vendor":
        vid = step.get("id") or "the vendor runtime"
        return (
            f"Reinstall {vid} into your home folder under .local. "
            "This does not use apt."
        )
    if kind == "write_core":
        return (
            "Rewrite the installer core files. That includes the environment "
            "file, memlock limits, and PATH."
        )
    if kind == "fix_bind":
        bind = step.get("bind") or "127.0.0.1"
        return f"Set the listen address to {bind}."
    if kind == "hash_file":
        path = step.get("path") or "the selected file"
        algo = step.get("algo") or "the published algorithm"
        return f"Compute the {algo} checksum of {path}."
    if kind == "install_workflow":
        title = _workflow_title(str(step.get("id") or ""))
        return (
            f"Install {title}. Ubuntu packages may be added. "
            "You may be asked for your password."
        )
    return f"Skip an unsupported step ({kind})."


def explain_plan(plan: dict) -> str:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    acting = [s for s in steps if s.get("kind") and s.get("kind") != "note"]
    notes = [s for s in steps if s.get("kind") == "note"]
    lines = ["What this means", ""]
    if not acting:
        lines.append(
            "Nothing on this computer needs an automatic change. "
            "The notes below are for your information."
        )
    else:
        lines.append(
            "The installer found problems it can try to fix. "
            "Nothing will change until you approve this plan."
        )
        lines.append("")
        lines.append("If you approve, the installer will:")
        for step in acting:
            lines.append(f"- {step_english(step)}")
    if notes:
        lines.append("")
        lines.append("Things it will not change automatically:")
        for step in notes:
            lines.append(f"- {step_english(step)}")
    lines.append("")
    src = plan.get("source") or "classical"
    if src == "openai":
        lines.append(
            "This plan was suggested by the OpenAI-compatible server you configured."
        )
    elif src == "local":
        lines.append("This plan was suggested by a local model on this computer.")
    else:
        lines.append(
            "This plan was built from local checks and checksums published on download pages."
        )
    lines.append("You can request changes, approve, or do nothing.")
    return "\n".join(lines)


def format_plan(plan: dict) -> str:
    lines = [explain_plan(plan), "", "Steps"]
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        lines.append("1. Nothing to do.")
        return "\n".join(lines)
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step_english(step)}")
    return "\n".join(lines)


def load_saved_plan(user: str) -> dict | None:
    t = target_for(user)
    path = plan_path(t.home)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def execute_plan_steps(
    user: str,
    plan: dict | None = None,
    *,
    dry_run: bool = False,
    on_progress: object | None = None,
) -> list[str]:
    saved = load_saved_plan(user)
    if plan is None:
        if saved is None:
            raise RuntimeError("no repair plan to approve")
        plan = saved.get("plan") or {}
    t = target_for(user)
    hw = probe()
    log: list[str] = []
    for step in plan.get("steps") or []:
        kind = step.get("kind")
        if kind not in ALLOW_KINDS:
            log.append(f"skip unsupported {kind}")
            continue
        if kind == "note":
            log.append(step.get("text") or "note")
            continue
        if dry_run:
            log.append(f"would {kind} {step.get('id') or step.get('bind') or ''}".strip())
            continue
        if on_progress:
            on_progress(f"Repair: {kind}")
        if kind == "write_core":
            rc, out = run_privileged(
                "core-files",
                [t.name, str(t.model_root), t.bind],
                dry_run=False,
            )
            if rc != 0:
                raise ApplyError(
                    explain_helper_failure("core-files", out, rc),
                    _technical_block(
                        [
                            "ubuntuai-installer-helper",
                            "core-files",
                            t.name,
                            str(t.model_root),
                            t.bind,
                        ],
                        rc,
                        out,
                    ),
                )
            log.append("wrote core files")
            continue
        if kind == "fix_bind":
            bind = step.get("bind") or "127.0.0.1"
            save_config(user, {"bind": bind})
            log.append(f"bind set to {bind}")
            continue
        if kind == "redownload":
            wid = step.get("id")
            catalog = {w.id: w for w in load_catalog()}
            model = catalog.get(wid)
            if model is None:
                log.append(f"unknown weight {wid}")
                continue
            expected = _expected_hash(step, model)
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
            dest = catalog_dest(model, t.model_root)
            want = expected or model.published_hash() or lookup_published_hash(model.url)
            if want and dest.exists():
                actual = hash_path(dest, want.algo)
                if actual.lower() != want.hexdigest.lower():
                    log.append(f"checksum still wrong for {wid} ({want.algo})")
                else:
                    log.append(f"checksum OK for {wid} ({want.algo})")
            elif dest.exists() and not want:
                log.append(f"no published checksum found for {wid}")
            continue
        if kind == "reinstall_vendor":
            vid = step.get("id") or "openmoss"
            log.append(install_vendor(vid, hw, t, force=True))
            continue
        if kind == "install_workflow":
            wid = step.get("id")
            if not wid:
                log.append("missing workflow id")
                continue
            actions = build_plan((wid,), hw, t)
            log.extend(
                execute_plan(
                    actions,
                    t,
                    hw=hw,
                    dry_run=False,
                    on_progress=on_progress,
                )
            )
            continue
        if kind == "hash_file":
            path = Path(step.get("path") or "")
            if not path.exists():
                log.append(f"missing {path}")
                continue
            algo = str(step.get("algo") or "")
            if not algo:
                found = parse_named_hash(str(step.get("hash") or step.get("sha256") or ""))
                algo = found.algo if found else ""
            if not algo:
                log.append(f"no published hash algorithm for {path}")
                continue
            log.append(f"{path} {algo}={hash_path(path, algo)}")
            continue
    return log


def _expected_hash(step: dict, model) -> FileHash | None:
    algo = str(step.get("algo") or "")
    hexdigest = str(step.get("hash") or step.get("sha256") or "")
    if hexdigest and not algo:
        parsed = parse_named_hash(hexdigest)
        if parsed:
            return parsed
        if model.published_hash():
            return FileHash(model.published_hash().algo, hexdigest.lower())
        looked = lookup_published_hash(model.url)
        if looked:
            return FileHash(looked.algo, hexdigest.lower())
        return None
    if hexdigest and algo:
        return FileHash(algo, hexdigest.lower())
    return model.published_hash()
