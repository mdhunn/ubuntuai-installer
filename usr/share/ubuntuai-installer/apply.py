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
from progress import ProgressEvent, emit, english_for_action, new_apply_log
from lemonade import APPLY_PUBLISH_VERB, detect as lemonade_detect
from vendor import install_vendor
from weights import ensure_weight, load_catalog as load_weight_catalog

PKG_RE = re.compile(r"^[a-zA-Z0-9.+-]+$")
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]*[$]?$")
GGML_VULKAN_PKG = "libggml0-backend-vulkan"
GGML_VULKAN_APPS = frozenset({"llama.cpp-tools", "whisper.cpp"})


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


def ensure_ggml_vulkan(packages: list[str] | tuple[str, ...]) -> list[str]:
    """Keep llama.cpp and whisper.cpp from landing as silent CPU-only installs."""
    pkgs = list(packages)
    if any(p in GGML_VULKAN_APPS for p in pkgs) and GGML_VULKAN_PKG not in pkgs:
        pkgs.append(GGML_VULKAN_PKG)
    return pkgs


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
        apt.extend(wf.packages_for(hw))
        groups.extend(wf.groups)
        subdirs.extend(wf.model_subdirs)

    apt = ensure_ggml_vulkan(apt)
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
    if lemonade_detect():
        actions.append(
            Action("lemonade", "publish GGUF files to Lemonade", (target.name,))
        )
    return tuple(actions)


def format_plan(actions: tuple[Action, ...]) -> str:
    lines = []
    for a in actions:
        extra = f" ({', '.join(a.payload)})" if a.payload else ""
        lines.append(f"- {a.kind}: {a.summary}{extra}")
    return "\n".join(lines) if lines else "- nothing to do"


class ApplyError(RuntimeError):
    def __init__(self, english: str, technical: str = "") -> None:
        super().__init__(english)
        self.english = english
        self.technical = (technical or "").strip()


def format_failure(err: ApplyError) -> str:
    lines = [err.english]
    if err.technical:
        lines.extend(["", "Technical details:", err.technical])
    return "\n".join(lines)


def explain_apt_failure(output: str, rc: int) -> str:
    text = (output or "").lower()
    if "could not get lock" in text or "unable to acquire the dpkg frontend lock" in text:
        return (
            "Could not install packages because another install is running. "
            "Wait for it to finish, then try again."
        )
    if "unmet dependencies" in text or ("depends:" in text and "but it is not" in text):
        return (
            "Could not install the Ubuntu packages. "
            "Some required packages are missing or clash with what is already installed."
        )
    if "conflict" in text:
        return (
            "Could not install the Ubuntu packages. "
            "apt reported a conflict between packages."
        )
    if "no space" in text or "not enough space" in text:
        return "Could not install the Ubuntu packages. The disk is full."
    if "unable to locate package" in text or "has no installation candidate" in text:
        return (
            "Could not install the Ubuntu packages. "
            "apt cannot find one of them. Enable the universe repository. "
            "Run apt update. Confirm this machine is Ubuntu 26.04 or newer."
        )
    if "network" in text or "temporary failure resolving" in text or "404" in text:
        return "Could not install the Ubuntu packages. A download or network step failed."
    if rc in {126, 127} or "not authorized" in text or "dismissed" in text:
        return "Administrator permission was not granted."
    if not (output or "").strip():
        return (
            "Could not install the Ubuntu packages. "
            "The privileged helper failed with no extra text. Open technical details for the exit code."
        )
    return "Could not install the Ubuntu packages. apt refused the request."


def explain_helper_failure(verb: str, output: str, rc: int) -> str:
    text = (output or "").lower()
    if rc in {126, 127} or "not authorized" in text or "dismissed" in text:
        return "Administrator permission was not granted."
    if verb == "install":
        return explain_apt_failure(output, rc)
    if verb == "groups":
        return "Could not add this user to the device groups."
    if verb == "core-files":
        return "Could not write the installer core files (environment, memlock, and PATH)."
    return "The privileged install step failed."


def _technical_block(cmd: list[str], rc: int, output: str) -> str:
    lines = ["command: " + " ".join(cmd), f"exit: {rc}"]
    if output.strip():
        lines.extend(["", output.strip()])
    return "\n".join(lines)


def _pump_output(proc: subprocess.Popen, on_line: object | None) -> str:
    chunks: list[str] = []
    buf = ""
    assert proc.stdout is not None
    while True:
        piece = proc.stdout.read(512)
        if not piece:
            break
        buf += piece
        while True:
            npos = buf.find("\n")
            rpos = buf.find("\r")
            cuts = [i for i in (npos, rpos) if i >= 0]
            if not cuts:
                break
            cut = min(cuts)
            line = buf[:cut]
            buf = buf[cut + 1 :]
            if line:
                chunks.append(line + "\n")
                if on_line:
                    on_line(line)
    if buf:
        chunks.append(buf)
        if on_line:
            on_line(buf)
    return "".join(chunks)


def run_privileged(
    verb: str,
    args: list[str],
    *,
    dry_run: bool = False,
    on_line: object | None = None,
) -> tuple[int, str]:
    if dry_run:
        return 0, ""
    helper = helper_path()
    cmd = [str(helper), verb, *args]
    if os.geteuid() != 0:
        pkexec = shutil.which("pkexec")
        if not pkexec:
            raise ApplyError(
                "Administrator permission is required to change system files. "
                "Install pkexec or run the installer as root."
            )
        cmd = [pkexec, *cmd]
    if on_line is None:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out = _pump_output(proc, on_line)
    rc = proc.wait()
    return rc, out


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
    work = [a for a in actions if a.kind != "skip"]
    total_steps = max(len(work), 1)
    step_i = 0
    log_path = None if dry_run else new_apply_log(
        target.home, uid=target.uid, gid=target.gid
    )
    if log_path is not None:
        log.append(f"log {log_path}")

    def relay(english: str, technical: str = "", fraction: float | None = None) -> None:
        frac = step_i / total_steps if fraction is None else min(max(fraction, 0.0), 1.0)
        emit(
            on_progress,
            ProgressEvent(english, technical, frac),
            log_path,
        )

    try:
        for action in actions:
            if action.kind == "skip":
                log.append(action.summary)
                continue
            relay(english_for_action(action), action.summary)
            if action.kind == "apt_install":
                log.append("apt install " + " ".join(action.payload))
                rc, out = run_privileged(
                    "install",
                    list(action.payload),
                    dry_run=dry_run,
                    on_line=None
                    if dry_run
                    else (lambda line: relay(english_for_action(action), line)),
                )
                if rc != 0:
                    raise ApplyError(
                        explain_helper_failure("install", out, rc),
                        _technical_block(
                            ["ubuntuai-installer-helper", "install", *action.payload],
                            rc,
                            out,
                        ),
                    )
            elif action.kind == "groups":
                log.append(f"usermod -aG {','.join(action.payload)} {target.name}")
                rc, out = run_privileged(
                    "groups",
                    [target.name, *action.payload],
                    dry_run=dry_run,
                )
                if rc != 0:
                    raise ApplyError(
                        explain_helper_failure("groups", out, rc),
                        _technical_block(
                            [
                                "ubuntuai-installer-helper",
                                "groups",
                                target.name,
                                *action.payload,
                            ],
                            rc,
                            out,
                        ),
                    )
            elif action.kind == "core_files":
                log.append(f"write {ENV_FILE}, {LIMITS_FILE}, {PROFILE_FILE}")
                rc, out = run_privileged(
                    "core-files",
                    [target.name, str(model_root), target.bind],
                    dry_run=dry_run,
                )
                if rc != 0:
                    raise ApplyError(
                        explain_helper_failure("core-files", out, rc),
                        _technical_block(
                            [
                                "ubuntuai-installer-helper",
                                "core-files",
                                target.name,
                                str(model_root),
                                target.bind,
                            ],
                            rc,
                            out,
                        ),
                    )
            elif action.kind == "model_dirs":
                log.append(f"mkdir {model_root} / " + ",".join(action.payload))
                if not dry_run:
                    _mkdirs(model_root, action.payload, target)
            elif action.kind == "vendor":
                vendor_id = action.payload[0]
                if hw is None:
                    raise ApplyError(
                        "Could not install the extra runtime. Hardware probe is missing."
                    )
                try:
                    msg = install_vendor(
                        vendor_id,
                        hw,
                        target,
                        on_progress=lambda msg: relay(str(msg), str(msg)),
                        dry_run=dry_run,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise ApplyError(
                        "Could not install the extra runtime into your home folder.",
                        str(exc),
                    ) from exc
                log.append(msg)
            elif action.kind == "weights":
                extra = saved_scan_folders(target.name)
                catalog = {w.id: w for w in load_weight_catalog()}
                wids = list(action.payload)
                for wi, wid in enumerate(wids):
                    model = catalog.get(wid)
                    if model is None:
                        raise ApplyError(
                            f"Could not find the model catalog entry {wid}.",
                        )
                    relay(
                        f"Getting {model.title}.",
                        model.filename,
                        fraction=(step_i + wi / max(len(wids), 1)) / total_steps,
                    )

                    def weight_cb(msg: object) -> None:
                        relay(str(msg), str(msg))

                    try:
                        msg = ensure_weight(
                            model,
                            target,
                            extra,
                            on_progress=weight_cb,
                            dry_run=dry_run,
                        )
                    except Exception as exc:  # noqa: BLE001
                        detail = str(exc)
                        low = detail.lower()
                        if "mismatch" in low:
                            english = (
                                "A downloaded model file did not match the checksum "
                                "published on its page. The broken file was not kept."
                            )
                        elif "incomplete download" in low or "empty download" in low:
                            english = (
                                "A downloaded model file was incomplete or the wrong size. "
                                "The broken file was not kept."
                            )
                        else:
                            english = "Could not get a required model file."
                        raise ApplyError(english, detail) from exc
                    log.append(msg)
            elif action.kind == "lemonade":
                rc, out = run_privileged(
                    APPLY_PUBLISH_VERB,
                    [target.name],
                    dry_run=dry_run,
                )
                if rc != 0:
                    raise ApplyError(
                        "Could not make downloaded models visible to Lemonade.",
                        out,
                    )
                log.append((out or "published models to Lemonade").strip())
            else:
                raise ApplyError(
                    f"The installer does not know how to run step {action.kind}."
                )
            step_i += 1
        emit(
            on_progress,
            ProgressEvent(
                "Finished.",
                f"log {log_path}" if log_path else "",
                1.0,
                done=True,
            ),
            log_path,
        )
        if not dry_run:
            _write_user_config(target, model_root)
        return log
    except ApplyError as exc:
        emit(
            on_progress,
            ProgressEvent(
                exc.english,
                exc.technical,
                min(step_i / total_steps, 1.0),
                failed=True,
            ),
            log_path,
        )
        raise
    except Exception as exc:
        emit(
            on_progress,
            ProgressEvent(
                "Apply failed.",
                str(exc),
                min(step_i / total_steps, 1.0),
                failed=True,
            ),
            log_path,
        )
        raise


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
    p = subprocess.run(
        ["usermod", "-aG", ",".join(groups), user],
        check=False,
    )
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def apt_install(packages: list[str]) -> None:
    pkgs = [_assert_pkg(p) for p in packages]
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    for cmd in (
        ["apt-get", "update"],
        ["apt-get", "install", "-y", "--no-install-recommends", *pkgs],
    ):
        p = subprocess.run(cmd, check=False, env=env)
        if p.returncode != 0:
            raise SystemExit(p.returncode)
