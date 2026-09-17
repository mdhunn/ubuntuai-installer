"""GTK4 + libadwaita UI."""

from __future__ import annotations

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from apply import build_plan, execute_plan, format_plan
from catalog import load_workflows, recommended_ids
from configstore import add_scan_folder
from configstore import load as load_config
from configstore import remove_scan_folder
from configstore import save as save_config
from configstore import saved_scan_folders
from repair import (
    build_plan_for,
    execute_plan_steps,
    format_plan as format_repair_plan,
    load_saved_plan,
)
from probe import probe
from users import guess_user, target_for
from validate import collect, format_checks, worst
from weights import (
    catalog_dest,
    download,
    human_bytes,
    load_catalog as load_weight_catalog,
    organize,
    scan,
    scan_roots,
)


def run(mode: str) -> int:
    app = Adw.Application(application_id="org.ubuntuai.Installer")
    app.connect("activate", lambda a: _on_activate(a, mode))
    return app.run([])


def _on_activate(app: Adw.Application, mode: str) -> None:
    win = Adw.ApplicationWindow(application=app)
    win.set_title("Ubuntu AI Configuration" if mode == "config" else "Ubuntu AI Installer")
    win.set_default_size(800, 720)
    if mode == "config":
        win.set_content(_config_box(win))
    else:
        win.set_content(_installer_box(win))
    win.present()


def _banner(hw) -> Gtk.Widget:
    backends = ", ".join(sorted(hw.backends()))
    ram = hw.ram_bytes // (1024**3)
    hybrid = "Hybrid available." if hw.hybrid_ok() else "Hybrid not detected."
    text = (
        f"{hw.cpu_name}\n"
        f"{ram} GiB RAM. Backends: {backends}. {hybrid}"
    )
    label = Gtk.Label(label=text, xalign=0, wrap=True)
    label.add_css_class("title-4")
    return label


def _installer_box(win: Adw.ApplicationWindow) -> Gtk.Widget:
    hw = probe()
    user = guess_user()
    target = target_for(user)
    workflows = load_workflows()
    checks = {wf.id: Gtk.CheckButton() for wf in workflows}

    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    header = Gtk.Label(label="Ubuntu AI Installer", xalign=0)
    header.add_css_class("title-1")
    outer.append(header)
    outer.append(_banner(hw))
    hint = Gtk.Label(
        label=f"Acting for {target.name}. Model root {target.model_root}. Bind {target.bind}.",
        xalign=0,
        wrap=True,
    )
    outer.append(hint)

    stack = Adw.ViewStack()
    switcher = Adw.ViewSwitcher()
    switcher.set_policy(Adw.ViewSwitcherPolicy.WIDE)
    switcher.set_stack(stack)
    outer.append(switcher)

    status = Gtk.Label(label="", xalign=0, wrap=True)
    wf_page = _workflows_page(win, hw, user, workflows, checks, status)
    wt_page = _weights_page(win, user, status)
    rp_page = _repair_page(win, user, status)
    stack.add_titled(wf_page, "workflows", "Workflows")
    stack.add_titled(wt_page, "weights", "Weights")
    stack.add_titled(rp_page, "repair", "Repair")
    outer.append(stack)
    outer.append(status)
    return outer


def _check_row(title: str, summary: str, check: Gtk.CheckButton) -> Gtk.Widget:
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
    row.set_margin_top(8)
    row.set_margin_bottom(8)
    row.set_margin_start(8)
    row.set_margin_end(8)
    check.set_valign(Gtk.Align.CENTER)
    text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
    head = Gtk.Label(label=title, xalign=0)
    head.add_css_class("heading")
    sub = Gtk.Label(label=summary, xalign=0, wrap=True)
    sub.add_css_class("dim-label")
    text.append(head)
    text.append(sub)
    row.append(check)
    row.append(text)
    return row


def _scroller(child: Gtk.Widget) -> Gtk.ScrolledWindow:
    scroller = Gtk.ScrolledWindow(vexpand=True)
    scroller.set_overlay_scrolling(False)
    scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scroller.set_child(child)
    return scroller


def _workflows_page(win, hw, user, workflows, checks, status) -> Gtk.Widget:
    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    listbox = Gtk.ListBox()
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    listbox.add_css_class("boxed-list")
    rec = recommended_ids(workflows, hw)
    for wf in workflows:
        cb = checks[wf.id]
        offered = wf.offered(hw) and wf.satisfied(hw)
        cb.set_active(wf.id in rec)
        cb.set_sensitive(offered and not wf.always)
        summary = wf.summary
        if wf.id in rec and wf.role:
            summary = "Recommended on this machine. " + wf.summary
        if not offered:
            summary = wf.summary + " Unavailable on this hardware."
        elif not wf.ready(hw):
            summary = wf.summary + " Apply will install the missing GPU packages."
        listbox.append(_check_row(wf.title, summary, cb))
    page.append(_scroller(listbox))
    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    exit_btn = Gtk.Button(label="Exit")
    dry = Gtk.Button(label="Dry run")
    apply_btn = Gtk.Button(label="Apply")
    apply_btn.add_css_class("suggested-action")
    buttons.append(exit_btn)
    buttons.append(Gtk.Box(hexpand=True))
    buttons.append(dry)
    buttons.append(apply_btn)
    page.append(buttons)

    def selected_ids() -> tuple[str, ...]:
        return tuple(wf.id for wf in workflows if checks[wf.id].get_active())

    def do_apply(dry_run: bool) -> None:
        import threading

        t = target_for(user)
        actions = build_plan(selected_ids(), hw, t)
        if dry_run:
            status.set_text("Dry run\n" + format_plan(actions))
            return

        def work() -> None:
            try:
                execute_plan(
                    actions,
                    t,
                    hw=hw,
                    dry_run=False,
                    on_progress=lambda msg: GLib.idle_add(status.set_text, str(msg)),
                )
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(status.set_text, f"Apply failed. {exc}")
                return
            report = collect(t, hw)
            extra = ""
            if any(c.name.startswith("group:") for c in report):
                extra = "\nLog out and back in if group membership just changed."
            GLib.idle_add(status.set_text, format_checks(report) + extra)
            if worst(report) == "fail":
                GLib.idle_add(apply_btn.add_css_class, "destructive-action")

        threading.Thread(target=work, daemon=True).start()

    exit_btn.connect("clicked", lambda *_: win.close())
    dry.connect("clicked", lambda *_: do_apply(True))
    apply_btn.connect("clicked", lambda *_: do_apply(False))
    return page


def _weights_page(win, user, status) -> Gtk.Widget:
    import threading

    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    t0 = target_for(user)
    hint = Gtk.Label(
        label=(
            f"Search known folders plus any you add, then symlink into {t0.model_root}. "
            "Move is offered when the source directory is writable. "
            "Downloads are opt-in and go into the same store."
        ),
        xalign=0,
        wrap=True,
    )
    hint.add_css_class("dim-label")
    page.append(hint)

    folders_head = Gtk.Label(label="Folders", xalign=0)
    folders_head.add_css_class("heading")
    page.append(folders_head)
    folder_box = Gtk.ListBox()
    folder_box.set_selection_mode(Gtk.SelectionMode.NONE)
    folder_box.add_css_class("boxed-list")
    page.append(_scroller(folder_box))
    folder_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    folder_entry = Gtk.Entry(placeholder_text="Path to a folder of weights", hexpand=True)
    add_folder_btn = Gtk.Button(label="Add")
    browse_btn = Gtk.Button(label="Browse")
    folder_row.append(folder_entry)
    folder_row.append(browse_btn)
    folder_row.append(add_folder_btn)
    page.append(folder_row)

    found_box = Gtk.ListBox()
    found_box.set_selection_mode(Gtk.SelectionMode.NONE)
    found_box.add_css_class("boxed-list")
    found_checks: dict[str, Gtk.CheckButton] = {}
    found_items = {}

    dl_box = Gtk.ListBox()
    dl_box.set_selection_mode(Gtk.SelectionMode.NONE)
    dl_box.add_css_class("boxed-list")
    dl_checks: dict[str, Gtk.CheckButton] = {}

    def refill_found() -> None:
        t = target_for(user)
        while True:
            row = found_box.get_row_at_index(0)
            if row is None:
                break
            found_box.remove(row)
        found_checks.clear()
        found_items.clear()
        roots = scan_roots(t.home, t.model_root, saved_scan_folders(user))
        found = scan(roots, t.model_root)
        new_n = sum(1 for f in found if f.state == "new")
        already_n = sum(1 for f in found if f.state == "already")
        status.set_text(
            f"Found {len(found)} weight files in {len(roots)} folders. "
            f"{new_n} new, {already_n} already in the store."
        )
        for item in found:
            if item.state == "already":
                continue
            key = str(item.path)
            cb = Gtk.CheckButton()
            cb.set_active(item.state == "new")
            cb.set_sensitive(item.state == "new")
            found_checks[key] = cb
            found_items[key] = item
            title = f"{item.dest_name} ({human_bytes(item.size)})"
            summary = (
                f"{item.state} · {item.fmt} · {item.kind} · {item.subdir} · {item.path}"
            )
            found_box.append(_check_row(title, summary, cb))

    def refill_downloads() -> None:
        t = target_for(user)
        while True:
            row = dl_box.get_row_at_index(0)
            if row is None:
                break
            dl_box.remove(row)
        dl_checks.clear()
        for model in load_weight_catalog():
            dest = catalog_dest(model, t.model_root)
            have = dest.exists() or dest.is_symlink()
            cb = Gtk.CheckButton()
            cb.set_active(model.default and not have)
            cb.set_sensitive(not have)
            dl_checks[model.id] = cb
            mark = "already in store" if have else human_bytes(model.bytes)
            dl_box.append(_check_row(model.title, f"{mark}. {model.summary}", cb))

    def refill_folders() -> None:
        while True:
            row = folder_box.get_row_at_index(0)
            if row is None:
                break
            folder_box.remove(row)
        saved = saved_scan_folders(user)
        if not saved:
            empty = Gtk.Label(
                label="No extra folders yet. Built-in locations are always searched.",
                xalign=0,
            )
            empty.add_css_class("dim-label")
            empty.set_margin_top(8)
            empty.set_margin_bottom(8)
            empty.set_margin_start(8)
            folder_box.append(empty)
            return
        for folder in saved:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row.set_margin_top(6)
            row.set_margin_bottom(6)
            row.set_margin_start(8)
            row.set_margin_end(8)
            lab = Gtk.Label(label=str(folder), xalign=0, hexpand=True, wrap=True)
            remove_btn = Gtk.Button(label="Remove")
            remove_btn.connect(
                "clicked",
                lambda _b, path=folder: do_remove_folder(path),
            )
            row.append(lab)
            row.append(remove_btn)
            folder_box.append(row)

    def do_add_folder(raw: str) -> None:
        text = raw.strip()
        if not text:
            status.set_text("Enter a folder path or use Browse.")
            return
        try:
            folder = add_scan_folder(user, text)
        except ValueError as exc:
            status.set_text(str(exc))
            return
        folder_entry.set_text("")
        status.set_text(f"Added {folder}")
        refill_folders()
        refill_found()

    def do_remove_folder(path) -> None:
        remove_scan_folder(user, path)
        status.set_text(f"Removed {path}")
        refill_folders()
        refill_found()

    def do_browse() -> None:
        dialog = Gtk.FileDialog(title="Folder to scan")

        def done(dlg, result) -> None:
            try:
                gfile = dlg.select_folder_finish(result)
            except Exception:
                return
            if gfile is None:
                return
            folder_entry.set_text(gfile.get_path() or "")
            do_add_folder(gfile.get_path() or "")

        dialog.select_folder(win, None, done)

    found_head = Gtk.Label(label="On disk", xalign=0)
    found_head.add_css_class("heading")
    page.append(found_head)
    page.append(_scroller(found_box))
    dl_head = Gtk.Label(label="Download", xalign=0)
    dl_head.add_css_class("heading")
    page.append(dl_head)
    page.append(_scroller(dl_box))

    mode_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    link = Gtk.CheckButton(label="Symlink")
    copy = Gtk.CheckButton(label="Copy", group=link)
    move = Gtk.CheckButton(label="Move", group=link)
    link.set_active(True)
    remove_src = Gtk.CheckButton(label="Remove original after checksum match")
    mode_row.append(link)
    mode_row.append(copy)
    mode_row.append(move)
    page.append(mode_row)
    page.append(remove_src)

    def on_mode(*_args) -> None:
        if link.get_active():
            remove_src.set_sensitive(False)
            remove_src.set_active(False)
        elif move.get_active():
            remove_src.set_sensitive(False)
            remove_src.set_active(True)
        else:
            remove_src.set_sensitive(True)

    link.connect("toggled", on_mode)
    copy.connect("toggled", on_mode)
    move.connect("toggled", on_mode)
    on_mode()

    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    exit_btn = Gtk.Button(label="Exit")
    scan_btn = Gtk.Button(label="Scan")
    org_btn = Gtk.Button(label="Organize")
    dl_btn = Gtk.Button(label="Download selected")
    dl_btn.add_css_class("suggested-action")
    buttons.append(exit_btn)
    buttons.append(Gtk.Box(hexpand=True))
    buttons.append(scan_btn)
    buttons.append(org_btn)
    buttons.append(dl_btn)
    page.append(buttons)

    def do_organize() -> None:
        t = target_for(user)
        picked = tuple(
            found_items[k] for k, cb in found_checks.items() if cb.get_active()
        )
        if not picked:
            status.set_text("No new weights selected.")
            return
        if copy.get_active():
            mode = "copy"
        elif move.get_active():
            mode = "move"
        else:
            mode = "link"
        log = organize(
            picked,
            t.model_root,
            mode=mode,
            remove_source=remove_src.get_active(),
            uid=t.uid,
            gid=t.gid,
        )
        status.set_text("\n".join(log))
        refill_found()

    def do_download() -> None:
        t = target_for(user)
        catalog = {m.id: m for m in load_weight_catalog()}
        ids = [mid for mid, cb in dl_checks.items() if cb.get_active()]
        if not ids:
            status.set_text("No catalog weights selected.")
            return

        def work() -> None:
            lines: list[str] = []
            for mid in ids:
                model = catalog[mid]

                def progress(done: int, total: int, name=model.title) -> None:
                    GLib.idle_add(
                        status.set_text,
                        f"Downloading {name}: {human_bytes(done)} / {human_bytes(total)}",
                    )

                try:
                    lines.append(
                        download(
                            model,
                            t.model_root,
                            on_progress=progress,
                            uid=t.uid,
                            gid=t.gid,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"{mid}: {exc}")
            def finish() -> None:
                status.set_text("\n".join(lines))
                refill_downloads()
                refill_found()
            GLib.idle_add(finish)

        threading.Thread(target=work, daemon=True).start()

    exit_btn.connect("clicked", lambda *_: win.close())
    scan_btn.connect("clicked", lambda *_: refill_found())
    org_btn.connect("clicked", lambda *_: do_organize())
    dl_btn.connect("clicked", lambda *_: do_download())
    add_folder_btn.connect(
        "clicked", lambda *_: do_add_folder(folder_entry.get_text())
    )
    folder_entry.connect(
        "activate", lambda *_: do_add_folder(folder_entry.get_text())
    )
    browse_btn.connect("clicked", lambda *_: do_browse())
    refill_folders()
    refill_found()
    refill_downloads()
    return page


def _repair_page(win, user, status) -> Gtk.Widget:
    import threading

    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    cfg = load_config(user)
    hint = Gtk.Label(
        label=(
            "Diagnose with local checks and the checksum published on each model page. "
            "The OpenAI path needs a server URI. Many providers also need an API key. "
            "The plan is shown in plain language. Nothing is applied until you approve."
        ),
        xalign=0,
        wrap=True,
    )
    hint.add_css_class("dim-label")
    page.append(hint)
    advisor_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    classical = Gtk.CheckButton(label="Classical")
    openai = Gtk.CheckButton(label="OpenAI URI", group=classical)
    local = Gtk.CheckButton(label="Local model", group=classical)
    classical.set_active(True)
    advisor_row.append(classical)
    advisor_row.append(openai)
    advisor_row.append(local)
    page.append(advisor_row)
    uri_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    uri_row.append(Gtk.Label(label="URI", xalign=0))
    uri_entry = Gtk.Entry(
        text=str(
            cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or ""
        ),
        hexpand=True,
        placeholder_text="https://api.openai.com/v1",
    )
    uri_row.append(uri_entry)
    page.append(uri_row)
    key_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    key_row.append(Gtk.Label(label="API key", xalign=0))
    key_entry = Gtk.Entry(
        text=str(cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or ""),
        hexpand=True,
        placeholder_text="API key for the URI",
    )
    key_entry.set_visibility(False)
    key_row.append(key_entry)
    page.append(key_row)
    plan_view = Gtk.TextView()
    plan_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    plan_view.set_editable(False)
    page.append(_scroller(plan_view))
    comment_entry = Gtk.Entry(placeholder_text="Request changes to the plan")
    page.append(comment_entry)
    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    exit_btn = Gtk.Button(label="Exit")
    diag_btn = Gtk.Button(label="Diagnose")
    revise_btn = Gtk.Button(label="Request changes")
    approve_btn = Gtk.Button(label="Approve")
    approve_btn.add_css_class("suggested-action")
    buttons.append(exit_btn)
    buttons.append(Gtk.Box(hexpand=True))
    buttons.append(diag_btn)
    buttons.append(revise_btn)
    buttons.append(approve_btn)
    page.append(buttons)
    plan_buf = plan_view.get_buffer()

    def advisor_name() -> str:
        if openai.get_active():
            return "openai"
        if local.get_active():
            return "local"
        return "classical"

    def set_plan_text(text: str) -> None:
        plan_buf.set_text(text, -1)

    def do_diagnose(comment: str) -> None:
        save_config(
            user,
            {
                "openai_base_url": uri_entry.get_text().strip(),
                "openai_api_key": key_entry.get_text().strip(),
                "repair_advisor": advisor_name(),
            },
        )

        def work() -> None:
            try:
                _diag, plan = build_plan_for(
                    user,
                    advisor_name(),
                    comment=comment,
                    openai_base_url=uri_entry.get_text().strip(),
                    openai_api_key=key_entry.get_text().strip(),
                )
                text = format_repair_plan(plan)
            except Exception as exc:  # noqa: BLE001
                text = f"Diagnose failed. {exc}"
            GLib.idle_add(set_plan_text, text)
            GLib.idle_add(status.set_text, "Review the plan. Approve only if you accept every step.")

        threading.Thread(target=work, daemon=True).start()

    def do_approve() -> None:
        saved = load_saved_plan(user)
        if saved is None:
            status.set_text("No plan to approve. Diagnose first.")
            return

        def work() -> None:
            try:
                log = execute_plan_steps(
                    user,
                    saved.get("plan"),
                    on_progress=lambda msg: GLib.idle_add(status.set_text, str(msg)),
                )
                GLib.idle_add(set_plan_text, "\n".join(log))
                GLib.idle_add(status.set_text, "Repair steps finished. Re-diagnose if you want a new plan.")
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(status.set_text, f"Repair failed. {exc}")

        threading.Thread(target=work, daemon=True).start()

    exit_btn.connect("clicked", lambda *_: win.close())
    diag_btn.connect("clicked", lambda *_: do_diagnose(""))
    revise_btn.connect(
        "clicked", lambda *_: do_diagnose(comment_entry.get_text().strip())
    )
    approve_btn.connect("clicked", lambda *_: do_approve())
    saved = load_saved_plan(user)
    if saved and saved.get("plan"):
        set_plan_text(str(saved.get("english") or format_repair_plan(saved["plan"])))
    return page


def _config_box(win: Adw.ApplicationWindow) -> Gtk.Widget:
    user = guess_user()
    data = load_config(user)
    hw = probe()
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    header = Gtk.Label(label="Ubuntu AI Configuration", xalign=0)
    header.add_css_class("title-1")
    outer.append(header)
    outer.append(_banner(hw))

    root_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    root_row.append(Gtk.Label(label="Model root", xalign=0))
    root_entry = Gtk.Entry(text=str(data.get("model_root") or ""), hexpand=True)
    root_row.append(root_entry)
    outer.append(root_row)

    bind_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    bind_row.append(Gtk.Label(label="Bind", xalign=0))
    local = Gtk.CheckButton(label="127.0.0.1")
    lan = Gtk.CheckButton(label="0.0.0.0 (LAN)", group=local)
    if data.get("bind") == "0.0.0.0":
        lan.set_active(True)
    else:
        local.set_active(True)
    bind_row.append(local)
    bind_row.append(lan)
    outer.append(bind_row)
    warn = Gtk.Label(
        label="LAN bind exposes local models on the network. Leave localhost unless you mean it.",
        xalign=0,
        wrap=True,
    )
    warn.add_css_class("dim-label")
    outer.append(warn)

    status = Gtk.Label(label="", xalign=0, wrap=True, vexpand=True)
    outer.append(status)

    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    exit_btn = Gtk.Button(label="Exit")
    save_btn = Gtk.Button(label="Save")
    val_btn = Gtk.Button(label="Validate")
    val_btn.add_css_class("suggested-action")
    spacer = Gtk.Box(hexpand=True)
    buttons.append(exit_btn)
    buttons.append(spacer)
    buttons.append(save_btn)
    buttons.append(val_btn)
    outer.append(buttons)

    def do_save() -> None:
        bind = "0.0.0.0" if lan.get_active() else "127.0.0.1"
        path = save_config(
            user,
            {"bind": bind, "model_root": root_entry.get_text().strip()},
        )
        status.set_text(f"Saved {path}")

    def do_validate() -> None:
        t = target_for(user)
        status.set_text(format_checks(collect(t, hw)))

    exit_btn.connect("clicked", lambda *_: win.close())
    save_btn.connect("clicked", lambda *_: do_save())
    val_btn.connect("clicked", lambda *_: do_validate())
    do_validate()
    return outer
