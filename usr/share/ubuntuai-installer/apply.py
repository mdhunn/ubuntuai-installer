"""Build an idempotent action plan and run it."""

from __future__ import annotations

import grp
import json
import os
import pwd
import re
import shutil
import subprocess
from pathlib import Path

from catalog import by_id, expand_selection, load_workflows
from configstore import saved_scan_folders
from domain import Action, Hardware, UserTarget, Workflow
from paths import ENV_FILE, LIMITS_FILE, LIMITS_TEMPLATE, PROFILE_FILE, helper_path
from vendor import install_vendor
from weights import ensure_weight, load_catalog as load_weight_catalog

PKG_RE = re.compile(r"^[a-zA-Z0-9.+-]+$")
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]*[$]?$")


def dpkg_installed(name: str) -> bool:
    p = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return p.returncode == 0 and "install ok installed" in p.stdout


def user_in_group(user: str, group: str) -> bool:
    try:
        g = grp.getgrnam(group)
    except KeyError:
        return False
    if user in g.gr_mem:
        return True
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return False
    return pw.pw_gid == g.gr_gid


def _assert_pkg(name: str) -> str:
    if not PKG_RE.match(name):
        raise ValueError(f"illegal package name {name!r}")
    return name


def _assert_user(name: str) -> str:
    if not USER_RE.match(name):
        raise ValueError(f"illegal user name {name!r}")
    return name


def assert_model_root(user: str, model_root: Path) -> Path:
    pw = pwd.getpwnam(user)
    home = Path(pw.pw_dir).resolve()
    root = model_root.expanduser()
    if not root.is_absolute():
        root = home / root
    root = root.resolve(strict=False)
    try:
        root.relative_to(home)
    except ValueError as exc:
        raise ValueError(f"model root {root} is outside {home}") from exc
    return root


def build_plan(
    selected: tuple[str, ...],
    hw: Hardware,
    target: UserTarget,
    workflows: tuple[Workflow, ...] | None = None,
) -> tuple[Action, ...]:
    catalog = workflows or load_workflows()
    ids = expand_selection(selected, catalog, hw)
    index = by_id(catalog)
    actions: list[Action] = []
    apt: list[str] = []
    groups: list[str] = []
    subdirs: list[str] = []
    for wid in ids:
        wf = index[wid]
        if not wf.offered(hw) or not wf.satisfied(hw):
            if wf.always:
                pass
            else:
                actions.append(
                    Action(
                        "skip",
                        f"skip {wf.id}. backends {sorted(hw.backends())} do not satisfy it",
                        (wf.id,),
                    )
                )
                continue
        apt.extend(wf.apt)
        groups.extend(wf.groups)
        subdirs.extend(wf.model_subdirs)

    missing_apt = []
    for pkg in dict.fromkeys(apt):
        _assert_pkg(pkg)
        if not dpkg_installed(pkg):
            missing_apt.append(pkg)
    if missing_apt:
        actions.append(Action("apt_install", "install apt packages", tuple(missing_apt)))

    missing_groups = []
    for g in dict.fromkeys(groups):
        if not user_in_group(target.name, g):
            missing_groups.append(g)
    if missing_groups:
        actions.append(
            Action(
                "groups",
                f"add {target.name} to groups",
                tuple(missing_groups),
            )
        )

    actions.append(
        Action("core_files", "write env, limits, profile, and model dirs", ())
    )
    unique_subdirs = tuple(dict.fromkeys(subdirs))
    if unique_subdirs:
        actions.append(
            Action("model_dirs", f"create dirs under {target.model_root}", unique_subdirs)
        )
    vendors: list[str] = []
    weight_ids: list[str] = []
    for wid in ids:
        wf = index.get(wid)
        if wf is None:
            continue
        if not wf.offered(hw) or not wf.satisfied(hw):
            if not wf.always:
                continue
        if wf.vendor:
            vendors.append(wf.vendor)
        weight_ids.extend(wf.required_weights)
    for vendor_id in dict.fromkeys(vendors):
        actions.append(Action("vendor", f"install {vendor_id} runtime", (vendor_id,)))
    unique_weights = tuple(dict.fromkeys(weight_ids))
    if unique_weights:
        actions.append(
            Action("weights", "link or download required weights", unique_weights)
        )
    return tuple(actions)


def format_plan(actions: tuple[Action, ...]) -> str:
    lines = []
    for a in actions:
        extra = f" ({', '.join(a.payload)})" if a.payload else ""
        lines.append(f"- {a.kind}: {a.summary}{extra}")
    return "\n".join(lines) if lines else "- nothing to do"


def run_privileged(verb: str, args: list[str], *, dry_run: bool = False) -> int:
    if dry_run:
        return 0
    helper = helper_path()
    cmd = [str(helper), verb, *args]
    if os.geteuid() == 0:
        p = subprocess.run(cmd, check=False)
        return p.returncode
    pkexec = shutil.which("pkexec")
    if not pkexec:
        raise RuntimeError("root or pkexec is required to apply system changes")
    p = subprocess.run([pkexec, *cmd], check=False)
    return p.returncode


def execute_plan(
    actions: tuple[Action, ...],
    target: UserTarget,
    *,
    hw: Hardware | None = None,
    dry_run: bool = False,
    on_progress: object | None = None,
) -> list[str]:
    log: list[str] = []
    _assert_user(target.name)
    model_root = assert_model_root(target.name, target.model_root)
    for action in actions:
        if action.kind == "skip":
            log.append(action.summary)
            continue
        if action.kind == "apt_install":
            log.append("apt install " + " ".join(action.payload))
            rc = run_privileged("install", list(action.payload), dry_run=dry_run)
            if rc != 0:
                raise RuntimeError(f"apt install failed with {rc}")
            continue
        if action.kind == "groups":
            log.append(f"usermod -aG {','.join(action.payload)} {target.name}")
            rc = run_privileged(
                "groups",
                [target.name, *action.payload],
                dry_run=dry_run,
            )
            if rc != 0:
                raise RuntimeError(f"group update failed with {rc}")
            continue
        if action.kind == "core_files":
            log.append(f"write {ENV_FILE}, {LIMITS_FILE}, {PROFILE_FILE}")
            rc = run_privileged(
                "core-files",
                [
                    target.name,
                    str(model_root),
                    target.bind,
                ],
                dry_run=dry_run,
            )
            if rc != 0:
                raise RuntimeError(f"core file write failed with {rc}")
            continue
        if action.kind == "model_dirs":
            log.append(f"mkdir {model_root} / " + ",".join(action.payload))
            if not dry_run:
                _mkdirs(model_root, action.payload, target)
            continue
        if action.kind == "vendor":
            vendor_id = action.payload[0]
            if hw is None:
                raise RuntimeError("vendor install needs a hardware probe")
            if on_progress:
                on_progress(f"Installing {vendor_id}")
            msg = install_vendor(
                vendor_id,
                hw,
                target,
                on_progress=on_progress,
                dry_run=dry_run,
            )
            log.append(msg)
            continue
        if action.kind == "weights":
            extra = saved_scan_folders(target.name)
            catalog = {w.id: w for w in load_weight_catalog()}
            for wid in action.payload:
                model = catalog.get(wid)
                if model is None:
                    raise KeyError(f"unknown weight {wid}")
                if on_progress:
                    on_progress(f"Ensuring {model.filename}")
                msg = ensure_weight(
                    model,
                    target,
                    extra,
                    on_progress=on_progress,
                    dry_run=dry_run,
                )
                log.append(msg)
            continue
        raise RuntimeError(f"unknown action {action.kind}")
    if not dry_run:
        _write_user_config(target, model_root)
    return log


def _mkdirs(root: Path, subdirs: tuple[str, ...], target: UserTarget) -> None:
    root.mkdir(parents=True, exist_ok=True)
    os.chown(root, target.uid, target.gid)
    os.chmod(root, 0o755)
    for name in subdirs:
        if "/" in name or name.startswith("."):
            raise ValueError(f"illegal model subdir {name!r}")
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        os.chown(d, target.uid, target.gid)


def _write_user_config(target: UserTarget, model_root: Path) -> None:
    from paths import user_config_path

    cfg = user_config_path(target.home)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_root": str(model_root),
        "extra_model_paths": [str(p) for p in target.extra_model_paths],
        "bind": target.bind,
        "user": target.name,
    }
    if cfg.is_file():
        try:
            old = json.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                old.update(data)
                data = old
        except json.JSONDecodeError:
            pass
    cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.chown(cfg, target.uid, target.gid)
    os.chown(cfg.parent, target.uid, target.gid)


def write_core_files(user: str, model_root: str, bind: str) -> None:
    """Called as root from the helper."""
    _assert_user(user)
    root = assert_model_root(user, Path(model_root))
    if bind not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError(f"illegal bind {bind}")
    ETC = ENV_FILE.parent
    ETC.mkdir(parents=True, exist_ok=True)
    env = (
        f"UBUNTUAI_MODELS={root}\n"
        f"UBUNTUAI_BIND={bind}\n"
        f"UBUNTUAI_USER={user}\n"
    )
    ENV_FILE.write_text(env, encoding="utf-8")
    os.chmod(ENV_FILE, 0o644)
    PROFILE_FILE.write_text(
        "# Ubuntu AI Installer\n"
        "[ -r /etc/ubuntuai/ubuntuai.env ] && . /etc/ubuntuai/ubuntuai.env\n"
        "[ -d \"$HOME/.local/bin\" ] && case \":$PATH:\" in\n"
        "  *\":$HOME/.local/bin:\"*) ;;\n"
        "  *) PATH=\"$HOME/.local/bin:$PATH\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    os.chmod(PROFILE_FILE, 0o644)
    text = LIMITS_TEMPLATE.read_text(encoding="utf-8")
    LIMITS_FILE.write_text(text, encoding="utf-8")
    os.chmod(LIMITS_FILE, 0o644)


def add_user_to_groups(user: str, groups: list[str]) -> None:
    _assert_user(user)
    for g in groups:
        if not re.match(r"^[a-z_][a-z0-9_-]*$", g):
            raise ValueError(f"illegal group {g!r}")
    if not groups:
        return
    subprocess.run(
        ["usermod", "-aG", ",".join(groups), user],
        check=True,
    )


def apt_install(packages: list[str]) -> None:
    pkgs = [_assert_pkg(p) for p in packages]
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    subprocess.run(["apt-get", "update"], check=True, env=env)
    subprocess.run(
        ["apt-get", "install", "-y", "--no-install-recommends", *pkgs],
        check=True,
        env=env,
    )
