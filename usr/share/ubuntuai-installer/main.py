"""CLI and GUI entry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from apply import ApplyError, build_plan, execute_plan, format_failure, format_plan
from catalog import expand_selection, load_workflows, recommended_ids
from configstore import add_scan_folder
from configstore import load as load_config
from configstore import remove_scan_folder
from configstore import record_installed
from configstore import save as save_config
from configstore import saved_scan_folders
from probe import probe
from users import guess_user, target_for
from validate import collect, format_checks, worst
from repair import (
    build_plan_for,
    execute_plan_steps,
    format_plan as format_repair_plan,
    load_saved_plan,
)
from weights import (
    download,
    human_bytes,
    load_catalog as load_weight_catalog,
    normalize_scan_folder,
    organize,
    scan,
    scan_roots,
)


def _detect_toolkit() -> str:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    session = os.environ.get("DESKTOP_SESSION", "").lower()
    if any(tok in desktop for tok in ("kde", "plasma", "lxqt")):
        return "qt"
    if "plasma" in session or "lxqt" in session:
        return "qt"
    return "gtk"


def _gui_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_gui(mode: str) -> int:
    preferred = _detect_toolkit()
    errors: list[str] = []
    order = [preferred, "gtk", "qt"]
    seen: set[str] = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        try:
            if name == "qt":
                from ui.qt_ui import run as run_qt

                return run_qt(mode)
            from ui.gtk_ui import run as run_gtk

            return run_gtk(mode)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    sys.stderr.write(
        "Neither GTK 4 (python3-gi + gir1.2-adw-1) nor PyQt6 is usable.\n"
        + "\n".join(errors)
        + "\n"
    )
    return 1


def cmd_list(user: str) -> int:
    hw = probe()
    wfs = load_workflows()
    t = target_for(user)
    print(f"user: {t.name}")
    print(f"cpu: {hw.cpu_name}")
    print(f"ram_giB: {hw.ram_bytes // (1024 ** 3)}")
    print(f"backends: {', '.join(sorted(hw.backends()))}")
    print(f"hybrid: {hw.hybrid_ok()}")
    print(f"model_root: {t.model_root}")
    print()
    rec = recommended_ids(wfs, hw)
    for wf in wfs:
        offered = wf.offered(hw) and wf.satisfied(hw)
        mark = "*" if wf.id in rec else " "
        if not offered:
            state = "unavailable"
        elif not wf.ready(hw):
            state = "install"
        else:
            state = "on"
        note = " recommended" if wf.role and wf.id in rec else ""
        print(f"{mark} {wf.id:24} {state:12} {wf.title}  {wf.summary}{note}")
    return 0


def cmd_plan(user: str, selected: tuple[str, ...], dry_run: bool) -> int:
    hw = probe()
    t = target_for(user)
    t = UserTargetWith(t, selected)
    actions = build_plan(selected, hw, t)
    print(format_plan(actions))
    if dry_run:
        return 0
    try:
        log = execute_plan(actions, t, hw=hw, dry_run=False)
    except ApplyError as exc:
        print(format_failure(exc), file=sys.stderr)
        return 1
    record_installed(user, expand_selection(selected, load_workflows(), hw))
    for line in log:
        print(line)
    print()
    checks = collect(t, hw)
    print(format_checks(checks))
    return 0 if worst(checks) != "fail" else 1


def UserTargetWith(t, selected):
    from dataclasses import replace

    return replace(t, selected=tuple(selected))


def cmd_validate(user: str, as_json: bool) -> int:
    t = target_for(user)
    checks = collect(t)
    if as_json:
        print(
            json.dumps(
                [
                    {"name": c.name, "status": c.status, "detail": c.detail}
                    for c in checks
                ],
                indent=2,
            )
        )
    else:
        print(format_checks(checks))
    status = worst(checks)
    print(f"\noverall: {status}")
    return 0 if status != "fail" else 1


def cmd_config_show(user: str) -> int:
    print(json.dumps(load_config(user), indent=2))
    return 0


def cmd_config_explain(user: str) -> int:
    from runtime import explain_setup
    from validate import collect, format_health

    hw = probe()
    t = target_for(user)
    print(explain_setup(user, t, hw))
    print()
    print("Health")
    print(format_health(collect(t, hw)))
    return 0


def _roots(user: str, extra: tuple[str, ...] = ()) -> tuple:
    t = target_for(user)
    folders = list(saved_scan_folders(user))
    for raw in extra:
        folders.append(normalize_scan_folder(raw, t.home))
    return scan_roots(t.home, t.model_root, tuple(folders))


def cmd_scan_weights(user: str, extra: tuple[str, ...] = ()) -> int:
    t = target_for(user)
    roots = _roots(user, extra)
    found = scan(roots, t.model_root)
    print(f"model_root: {t.model_root}")
    print(f"scan_roots: {', '.join(str(p) for p in roots)}")
    if not found:
        print("no weights found")
        return 0
    for item in found:
        print(
            f"{item.state:8} {item.kind:4} {item.fmt:11} {item.subdir:12} "
            f"{human_bytes(item.size):>8}  "
            f"{item.path} -> {item.subdir}/{item.dest_name}"
        )
    return 0


def cmd_organize_weights(
    user: str,
    mode: str,
    dry_run: bool,
    extra: tuple[str, ...] = (),
    remove_source: bool = False,
) -> int:
    t = target_for(user)
    roots = _roots(user, extra)
    found = tuple(f for f in scan(roots, t.model_root) if f.state == "new")
    if not found:
        print("nothing new to organize")
        return 0
    log = organize(
        found,
        t.model_root,
        mode=mode,
        remove_source=remove_source,
        uid=t.uid,
        gid=t.gid,
        dry_run=dry_run,
    )
    for line in log:
        print(line)
    return 0


def cmd_repair(
    user: str,
    advisor: str,
    *,
    approve: bool,
    comment: str,
    openai_base_url: str,
    openai_api_key: str,
    dry_run: bool,
) -> int:
    if approve:
        if comment:
            print(
                "--repair-comment revises a plan. Run --repair --repair-comment first.",
                file=sys.stderr,
            )
            return 2
        saved = load_saved_plan(user)
        if saved is None:
            print("no saved repair plan. run --repair first.", file=sys.stderr)
            return 2
        try:
            log = execute_plan_steps(user, saved.get("plan"), dry_run=dry_run)
        except ApplyError as exc:
            print(format_failure(exc), file=sys.stderr)
            return 1
        for line in log:
            print(line)
        return 0
    updates = {"repair_advisor": advisor}
    if openai_base_url:
        updates["openai_base_url"] = openai_base_url
    if openai_api_key:
        updates["openai_api_key"] = openai_api_key
    save_config(user, updates)
    diag, plan = build_plan_for(
        user,
        advisor,
        comment=comment,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )
    print(format_repair_plan(plan))
    print("")
    print("Review this text. Then --repair-approve to apply, or --repair-comment to change it.")
    return 0


def cmd_list_downloads(user: str) -> int:
    t = target_for(user)
    for w in load_weight_catalog():
        dest = t.model_root / w.subdir / w.filename
        have = "have" if dest.exists() or dest.is_symlink() else "get"
        print(f"{have:4} {w.id:28} {human_bytes(w.bytes):>8}  {w.title}")
    return 0


def cmd_download(user: str, ids: tuple[str, ...], dry_run: bool) -> int:
    t = target_for(user)
    catalog = {w.id: w for w in load_weight_catalog()}
    for mid in ids:
        model = catalog.get(mid)
        if model is None:
            print(f"unknown weight {mid}", file=sys.stderr)
            return 2
        print(download(model, t.model_root, uid=t.uid, gid=t.gid, dry_run=dry_run))
    return 0


def cmd_config_set(
    user: str,
    bind: str | None,
    model_root: str | None,
    chat_model: str | None = None,
    backend: str | None = None,
    tts: str | None = None,
    stt: str | None = None,
    openai_uri: str | None = None,
    openai_key: str | None = None,
) -> int:
    updates = {}
    if bind:
        updates["bind"] = bind
    if model_root:
        updates["model_root"] = model_root
    if chat_model is not None:
        updates["chat_model"] = "" if chat_model == "auto" else chat_model
    if backend is not None:
        updates["primary_backend"] = "" if backend == "auto" else backend
    if tts is not None:
        updates["tts_engine"] = "" if tts == "auto" else tts
    if stt is not None:
        updates["stt_engine"] = "" if stt == "auto" else stt
    if openai_uri is not None:
        updates["openai_base_url"] = openai_uri
    if openai_key is not None:
        updates["openai_api_key"] = openai_key
    path = save_config(user, updates)
    print(path)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ubuntuai-installer",
        description="Add local-AI workstation packages to this Ubuntu flavor.",
    )
    p.add_argument(
        "--user",
        help="desktop user to configure. default is SUDO_USER, PKEXEC_UID, UBUNTUAI_USER, or the current uid",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--install", nargs="+", metavar="WORKFLOW")
    p.add_argument("--cli", action="store_true", help="do not open the GUI")
    p.add_argument("--scan-weights", action="store_true")
    p.add_argument(
        "--organize-weights",
        nargs="?",
        const="link",
        choices=("link", "copy", "move"),
        help="symlink (default), copy, or move discovered weights into the model root",
    )
    p.add_argument(
        "--remove-source",
        action="store_true",
        help="after copy, delete the original only if the checksum matches. move always does this",
    )
    p.add_argument("--repair", action="store_true", help="diagnose and write a repair plan")
    p.add_argument(
        "--repair-approve",
        action="store_true",
        help="apply the last saved repair plan after you have reviewed it",
    )
    p.add_argument("--repair-comment", default="", help="ask the advisor to revise the plan")
    p.add_argument(
        "--repair-advisor",
        choices=("classical", "openai", "local"),
        default="classical",
    )
    p.add_argument(
        "--openai-uri",
        default="",
        help="OpenAI-compatible base URL for repair. also OPENAI_BASE_URL",
    )
    p.add_argument(
        "--openai-key",
        default="",
        help="API key for --openai-uri. also OPENAI_API_KEY",
    )
    p.add_argument("--list-downloads", action="store_true")
    p.add_argument("--download", nargs="+", metavar="WEIGHT_ID")
    p.add_argument(
        "--scan-folder",
        action="append",
        default=[],
        metavar="DIR",
        help="extra folder to search this run (repeatable)",
    )
    p.add_argument(
        "--add-scan-folder",
        action="append",
        default=[],
        metavar="DIR",
        help="remember a folder and search it on later scans",
    )
    p.add_argument(
        "--remove-scan-folder",
        action="append",
        default=[],
        metavar="DIR",
        help="forget a remembered scan folder",
    )
    p.add_argument("--list-scan-folders", action="store_true")
    return p


def installer_main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    user = guess_user(args.user)
    extra = tuple(args.scan_folder or ())
    for raw in args.add_scan_folder or ():
        print(add_scan_folder(user, raw))
    for raw in args.remove_scan_folder or ():
        remove_scan_folder(user, raw)
        print(f"removed {raw}")
    if args.list_scan_folders:
        t = target_for(user)
        print("builtin:")
        for p in scan_roots(t.home, t.model_root):
            print(f"  {p}")
        print("saved:")
        saved = saved_scan_folders(user)
        if not saved:
            print("  (none)")
        for p in saved:
            print(f"  {p}")
        return 0
    if args.list:
        return cmd_list(user)
    if args.scan_weights:
        return cmd_scan_weights(user, extra)
    if args.repair or args.repair_approve:
        return cmd_repair(
            user,
            args.repair_advisor,
            approve=args.repair_approve,
            comment=args.repair_comment,
            openai_base_url=args.openai_uri,
            openai_api_key=args.openai_key,
            dry_run=args.dry_run,
        )
    if args.organize_weights:
        return cmd_organize_weights(
            user,
            args.organize_weights,
            dry_run=args.dry_run,
            extra=extra,
            remove_source=args.remove_source,
        )
    if args.list_downloads:
        return cmd_list_downloads(user)
    if args.download:
        return cmd_download(user, tuple(args.download), dry_run=args.dry_run)
    if args.add_scan_folder or args.remove_scan_folder:
        return 0
    if args.install:
        return cmd_plan(user, tuple(args.install), dry_run=args.dry_run)
    if args.dry_run:
        return cmd_plan(user, ("ubuntuai-core", "ubuntuai-chat"), dry_run=True)
    if args.cli or not _gui_available():
        return cmd_list(user)
    return run_gui("installer")


def validate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ubuntuai-validate")
    p.add_argument("--user")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    return cmd_validate(guess_user(args.user), args.json)


def config_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ubuntuai-config")
    p.add_argument("--user")
    p.add_argument("--show", action="store_true", help="print config.json")
    p.add_argument(
        "--explain",
        action="store_true",
        help="plain-language view of installed apps and health",
    )
    p.add_argument("--bind", choices=("127.0.0.1", "0.0.0.0"))
    p.add_argument("--model-root")
    p.add_argument("--chat-model")
    p.add_argument("--backend")
    p.add_argument("--tts")
    p.add_argument("--stt")
    p.add_argument("--openai-uri")
    p.add_argument("--openai-key")
    p.add_argument("--cli", action="store_true")
    args = p.parse_args(argv)
    user = guess_user(args.user)
    if args.explain:
        return cmd_config_explain(user)
    if args.show:
        return cmd_config_show(user)
    if any(
        (
            args.bind,
            args.model_root,
            args.chat_model,
            args.backend,
            args.tts,
            args.stt,
            args.openai_uri,
            args.openai_key,
        )
    ):
        return cmd_config_set(
            user,
            args.bind,
            args.model_root,
            chat_model=args.chat_model,
            backend=args.backend,
            tts=args.tts,
            stt=args.stt,
            openai_uri=args.openai_uri,
            openai_key=args.openai_key,
        )
    if args.cli or not _gui_available():
        return cmd_config_explain(user)
    return run_gui("config")


def _dispatch() -> int:
    name = Path(sys.argv[0]).name
    if "validate" in name:
        return validate_main()
    if "config" in name:
        return config_main()
    return installer_main()


if __name__ == "__main__":
    sys.exit(_dispatch())
