"""GTK4 + libadwaita UI."""

from __future__ import annotations

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from apply import ApplyError, build_plan, execute_plan, format_plan
from lemonade import detect as lemonade_detect
from load_warn import (
    CANCEL,
    LoadWarn,
    plan_publishes_lemonade,
    warn_for_chat_model,
    warn_for_publish,
)
from progress import PAINT_MS, ProgressFrame, ProgressPump
from catalog import expand_selection, helper_workflow_ids, load_workflows, recommended_ids
from configstore import add_scan_folder
from configstore import load as load_config
from configstore import record_installed
from configstore import remove_scan_folder
from configstore import save as save_config
from configstore import saved_scan_folders
from repair import (
    build_plan_for,
    execute_plan_steps,
    format_plan as format_repair_plan,
    load_saved_plan,
)
from upgrade import (
    build_plan_for as build_upgrade_plan,
    execute_plan_steps as execute_upgrade_steps,
    format_plan as format_upgrade_plan,
    load_saved_plan as load_upgrade_plan,
)
from probe import probe
from users import guess_user, target_for
from runtime import (
    BACKEND_LABELS,
    CHAT_MODEL_CTA,
    CHAT_MODEL_NEEDED,
    CHAT_MODEL_NEEDED_CONFIG,
    HELPERS_BLURB,
    HELPERS_SECTION,
    app_statuses,
    backend_choices,
    chat_models,
    chat_weight_ids,
    listen_words,
    machine_words,
    no_chat_gguf,
    workflow_row_title,
)
from validate import collect, format_checks, format_health, worst
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
    ram = hw.ram_bytes // (1024**3)
    text = f"{hw.cpu_name}\n{ram} GiB RAM. {machine_words(hw)}"
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
        label=(
            f"Acting for {target.name}. Model folder {target.model_root}. "
            f"{listen_words(target.bind)}"
        ),
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
    wt_page = _weights_page(win, user, status)

    def show_chat_download() -> None:
        stack.set_visible_child_name("weights")
        prepare = getattr(wt_page, "prepare_chat_download", None)
        if prepare:
            prepare()

    wf_page = _workflows_page(
        win, hw, user, workflows, checks, status, show_chat_download
    )
    rp_page = _repair_page(win, user, status)
    up_page = _upgrade_page(win, user, status)
    stack.add_titled(wf_page, "workflows", "Workflows")
    stack.add_titled(wt_page, "weights", "Weights")
    stack.add_titled(up_page, "updates", "Updates")
    stack.add_titled(rp_page, "repair", "Repair")
    outer.append(stack)
    outer.append(status)
    return outer


def _confirm_load_warn(win, warn: LoadWarn, on_continue) -> None:
    if not warn.should_prompt:
        on_continue()
        return
    dialog = Gtk.Window(transient_for=win, modal=True, title=warn.title)
    dialog.set_default_size(520, 280)
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    body = Gtk.Label(label=warn.body, wrap=True, xalign=0)
    body.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    outer.append(body)

    def accept(*_args) -> None:
        dialog.close()
        on_continue()

    go = Gtk.Button(label=warn.primary)
    go.add_css_class("suggested-action")
    go.connect("clicked", accept)
    cancel = Gtk.Button(label=CANCEL)
    cancel.connect("clicked", lambda *_: dialog.close())
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    row.append(cancel)
    row.append(Gtk.Box(hexpand=True))
    row.append(go)
    outer.append(row)
    dialog.set_child(outer)
    dialog.present()


def _show_chat_model_cta(win, on_download, body_text: str = "") -> None:
    dialog = Gtk.Window(transient_for=win, modal=True, title="Chat needs a model")
    dialog.set_default_size(520, 220)
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    body = Gtk.Label(label=body_text or CHAT_MODEL_NEEDED, wrap=True, xalign=0)
    outer.append(body)
    go = Gtk.Button(label=CHAT_MODEL_CTA)
    go.add_css_class("suggested-action")

    def accept(*_args) -> None:
        dialog.close()
        on_download()

    go.connect("clicked", accept)
    later = Gtk.Button(label="Later")
    later.connect("clicked", lambda *_: dialog.close())
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    row.append(later)
    row.append(Gtk.Box(hexpand=True))
    row.append(go)
    outer.append(row)
    dialog.set_child(outer)
    dialog.present()


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


def _show_failure(win, english: str, technical: str = "") -> None:
    dialog = Gtk.Window(transient_for=win, modal=True, title="Install failed")
    dialog.set_default_size(560, 280)
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    body = Gtk.Label(label=english, wrap=True, xalign=0)
    body.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    outer.append(body)
    if technical.strip():
        expander = Gtk.Expander(label="Technical details")
        view = Gtk.TextView()
        view.set_editable(False)
        view.set_cursor_visible(False)
        view.set_monospace(True)
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        view.get_buffer().set_text(technical)
        scroll = Gtk.ScrolledWindow(vexpand=True)
        scroll.set_min_content_height(140)
        scroll.set_child(view)
        expander.set_child(scroll)
        outer.append(expander)
    ok = Gtk.Button(label="OK")
    ok.add_css_class("suggested-action")
    ok.connect("clicked", lambda *_: dialog.close())
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    row.append(Gtk.Box(hexpand=True))
    row.append(ok)
    outer.append(row)
    dialog.set_child(outer)
    dialog.present()


def _open_apply_progress(win) -> tuple[object, object]:
    dialog = Gtk.Window(transient_for=win, modal=True, title="Installing")
    dialog.set_default_size(580, 400)
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    english = Gtk.Label(label="Starting.", wrap=True, xalign=0)
    outer.append(english)
    bar = Gtk.ProgressBar()
    bar.set_show_text(True)
    bar.set_fraction(0)
    outer.append(bar)
    hint = Gtk.Label(
        label="This can take a while for large model files. You can open Technical details for apt and download lines.",
        wrap=True,
        xalign=0,
    )
    hint.add_css_class("dim-label")
    outer.append(hint)
    expander = Gtk.Expander(label="Technical details")
    view = Gtk.TextView()
    view.set_editable(False)
    view.set_cursor_visible(False)
    view.set_monospace(True)
    view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    buf = view.get_buffer()
    scroll = Gtk.ScrolledWindow(vexpand=True)
    scroll.set_min_content_height(160)
    scroll.set_child(view)
    expander.set_child(scroll)
    outer.append(expander)
    close_btn = Gtk.Button(label="Close")
    close_btn.set_sensitive(False)
    close_btn.connect("clicked", lambda *_: dialog.close())
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    row.append(Gtk.Box(hexpand=True))
    row.append(close_btn)
    outer.append(row)
    dialog.set_child(outer)
    dialog.present()
    pump = ProgressPump()

    def paint(frame: ProgressFrame) -> None:
        bar.set_fraction(max(0.0, min(frame.fraction, 1.0)))
        if frame.english.startswith("Downloading "):
            bar.set_text(frame.english)
        else:
            bar.set_text(f"{int(frame.fraction * 100)}%")
        english.set_label(frame.english)
        for line in frame.technical:
            if line:
                buf.insert(buf.get_end_iter(), line + "\n")
        if frame.done or frame.failed:
            close_btn.set_sensitive(True)
        english.queue_draw()
        bar.queue_draw()

    def tick() -> bool:
        frame = pump.take()
        if frame is not None:
            paint(frame)
            if frame.done or frame.failed:
                return False
        return True

    GLib.timeout_add(PAINT_MS, tick)
    return dialog, pump.push


def _scroller(child: Gtk.Widget) -> Gtk.ScrolledWindow:
    scroller = Gtk.ScrolledWindow(vexpand=True)
    scroller.set_overlay_scrolling(False)
    scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scroller.set_child(child)
    return scroller


def _workflows_page(win, hw, user, workflows, checks, status, show_chat_download) -> Gtk.Widget:
    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    helpers = helper_workflow_ids(workflows)
    rec = recommended_ids(workflows, hw)

    def fill_row(wf) -> Gtk.Widget:
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
        title = workflow_row_title(wf.title, wf.id, helpers)
        return _check_row(title, summary, cb)

    mains = [wf for wf in workflows if wf.id not in helpers]
    helper_wfs = [wf for wf in workflows if wf.id in helpers]
    listbox = Gtk.ListBox()
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    listbox.add_css_class("boxed-list")
    for wf in mains:
        listbox.append(fill_row(wf))
    page.append(_scroller(listbox))
    if helper_wfs:
        helpers_head = Gtk.Label(label=HELPERS_SECTION, xalign=0)
        helpers_head.add_css_class("heading")
        page.append(helpers_head)
        helpers_blurb = Gtk.Label(label=HELPERS_BLURB, xalign=0, wrap=True)
        helpers_blurb.add_css_class("dim-label")
        page.append(helpers_blurb)
        helper_box = Gtk.ListBox()
        helper_box.set_selection_mode(Gtk.SelectionMode.NONE)
        helper_box.add_css_class("boxed-list")
        for wf in helper_wfs:
            helper_box.append(fill_row(wf))
        page.append(_scroller(helper_box))
    cta = Gtk.Button(label=CHAT_MODEL_CTA)
    cta.add_css_class("suggested-action")
    cta.connect("clicked", lambda *_: show_chat_download())
    if no_chat_gguf(target_for(user).model_root):
        cta.set_visible(True)
    else:
        cta.set_visible(False)
    page.append(cta)
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

        def start_apply() -> None:
            _dialog, update = _open_apply_progress(win)

            def work() -> None:
                try:
                    execute_plan(
                        actions,
                        t,
                        hw=hw,
                        dry_run=False,
                        on_progress=update,
                    )
                except ApplyError as exc:
                    GLib.idle_add(status.set_text, exc.english)
                    GLib.idle_add(_show_failure, win, exc.english, exc.technical)
                    return
                except Exception as exc:  # noqa: BLE001
                    GLib.idle_add(status.set_text, f"Apply failed. {exc}")
                    GLib.idle_add(_show_failure, win, "Apply failed.", str(exc))
                    return
                record_installed(user, expand_selection(selected_ids(), workflows, hw))
                report = collect(t, hw)
                extra = ""
                if any(c.name.startswith("group:") for c in report):
                    extra = "\nLog out and back in if group membership just changed."
                need_model = no_chat_gguf(t.model_root)
                if need_model:
                    extra += "\n" + CHAT_MODEL_NEEDED
                GLib.idle_add(status.set_text, format_checks(report) + extra)
                if need_model:
                    GLib.idle_add(cta.set_visible, True)
                    GLib.idle_add(_show_chat_model_cta, win, show_chat_download)
                else:
                    GLib.idle_add(cta.set_visible, False)
                if worst(report) == "fail":
                    GLib.idle_add(apply_btn.add_css_class, "destructive-action")

            threading.Thread(target=work, daemon=True).start()

        if plan_publishes_lemonade(actions):
            _confirm_load_warn(win, warn_for_publish(t, hw), start_apply)
        else:
            start_apply()

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
            f"Search known folders plus any you add, then copy into {t0.model_root}. "
            "Move removes the original after the checksum matches. "
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
    copy = Gtk.CheckButton(label="Copy")
    move = Gtk.CheckButton(label="Move", group=copy)
    copy.set_active(True)
    remove_src = Gtk.CheckButton(label="Remove original after checksum match")
    mode_row.append(copy)
    mode_row.append(move)
    page.append(mode_row)
    page.append(remove_src)

    def on_mode(*_args) -> None:
        if move.get_active():
            remove_src.set_sensitive(False)
            remove_src.set_active(True)
        else:
            remove_src.set_sensitive(True)

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
        mode = "move" if move.get_active() else "copy"
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
                    english = (
                        "A downloaded model file did not match its published checksum. "
                        "The broken file was not kept."
                        if "mismatch" in str(exc).lower()
                        else "Could not download a selected model file."
                    )
                    lines.append(english)
                    GLib.idle_add(_show_failure, win, english, str(exc))
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

    def prepare_chat_download() -> None:
        refill_downloads()
        needed = chat_weight_ids()
        for model in load_weight_catalog():
            want = model.id in needed or (
                model.default and "ubuntuai-chat" in model.workflows
            )
            if not want:
                continue
            cb = dl_checks.get(model.id)
            if cb is not None and cb.get_sensitive():
                cb.set_active(True)
        dl_btn.grab_focus()
        status.set_text(CHAT_MODEL_NEEDED)

    refill_folders()
    refill_found()
    refill_downloads()
    page.prepare_chat_download = prepare_chat_download
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
            except ApplyError as exc:
                GLib.idle_add(status.set_text, exc.english)
                GLib.idle_add(_show_failure, win, exc.english, exc.technical)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(status.set_text, f"Repair failed. {exc}")
                GLib.idle_add(_show_failure, win, "Repair failed.", str(exc))

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


def _upgrade_page(win, user, status) -> Gtk.Widget:
    import threading

    page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    hint = Gtk.Label(
        label=(
            "Upgrade vendor runtimes that are not apt or snap, refresh catalog "
            "models that drifted, and tune Lemonade for very large GGUF files. "
            "Diagnose writes a plan. Nothing is applied until you approve."
        ),
        xalign=0,
        wrap=True,
    )
    hint.add_css_class("dim-label")
    page.append(hint)
    plan_view = Gtk.TextView()
    plan_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    plan_view.set_editable(False)
    page.append(_scroller(plan_view))
    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    exit_btn = Gtk.Button(label="Exit")
    diag_btn = Gtk.Button(label="Diagnose")
    approve_btn = Gtk.Button(label="Approve")
    approve_btn.add_css_class("suggested-action")
    buttons.append(exit_btn)
    buttons.append(Gtk.Box(hexpand=True))
    buttons.append(diag_btn)
    buttons.append(approve_btn)
    page.append(buttons)

    def set_plan_text(text: str) -> None:
        plan_view.get_buffer().set_text(text)

    def do_diagnose() -> None:
        def work() -> None:
            try:
                _diag, plan = build_upgrade_plan(user)
                text = format_upgrade_plan(plan)
            except Exception as exc:  # noqa: BLE001
                text = f"Diagnose failed. {exc}"
            GLib.idle_add(set_plan_text, text)
            GLib.idle_add(
                status.set_text,
                "Review the update plan. Approve only if you accept every step.",
            )

        threading.Thread(target=work, daemon=True).start()

    def do_approve() -> None:
        saved = load_upgrade_plan(user)
        if saved is None:
            status.set_text("No update plan to approve. Diagnose first.")
            return

        def work() -> None:
            try:
                log = execute_upgrade_steps(user, saved.get("plan"))
                GLib.idle_add(set_plan_text, "\n".join(log))
                GLib.idle_add(status.set_text, "Updates finished. Diagnose again if you want a new plan.")
            except ApplyError as exc:
                GLib.idle_add(status.set_text, exc.english)
                GLib.idle_add(_show_failure, win, exc.english, exc.technical)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(status.set_text, f"Update failed. {exc}")
                GLib.idle_add(_show_failure, win, "Update failed.", str(exc))

        threading.Thread(target=work, daemon=True).start()

    exit_btn.connect("clicked", lambda *_: win.close())
    diag_btn.connect("clicked", lambda *_: do_diagnose())
    approve_btn.connect("clicked", lambda *_: do_approve())
    saved = load_upgrade_plan(user)
    if saved and saved.get("plan"):
        set_plan_text(str(saved.get("english") or format_upgrade_plan(saved["plan"])))
    return page


def _combo(pairs: list[tuple[str, str]], current: str) -> Gtk.ComboBoxText:
    box = Gtk.ComboBoxText()
    ids = []
    for key, label in pairs:
        box.append(key, label)
        ids.append(key)
    if current in ids:
        box.set_active_id(current)
    elif ids:
        box.set_active_id(ids[0])
    return box


def _config_box(win: Adw.ApplicationWindow) -> Gtk.Widget:
    user = guess_user()
    t = target_for(user)
    data = load_config(user)
    hw = probe()
    apps = app_statuses(user, t)
    outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    outer.set_margin_top(16)
    outer.set_margin_bottom(16)
    outer.set_margin_start(16)
    outer.set_margin_end(16)
    header = Gtk.Label(label="Ubuntu AI Configuration", xalign=0)
    header.add_css_class("title-1")
    outer.append(header)
    hint = Gtk.Label(
        label=(
            "Settings for the apps Ubuntu AI Installer put on this computer. "
            "Overview is the short path. Advanced has paths and API fields."
        ),
        xalign=0,
        wrap=True,
    )
    hint.add_css_class("dim-label")
    outer.append(hint)
    outer.append(_banner(hw))

    stack = Adw.ViewStack()
    switcher = Adw.ViewSwitcher()
    switcher.set_policy(Adw.ViewSwitcherPolicy.WIDE)
    switcher.set_stack(stack)
    outer.append(switcher)

    overview = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    helpers = helper_workflow_ids(load_workflows())
    installed = [a for a in apps if a.present]
    products = [a for a in installed if a.id not in helpers]
    helper_apps = [a for a in installed if a.id in helpers]
    missing = [
        a
        for a in apps
        if not a.present and a.id != "ubuntuai-core" and a.id not in helpers
    ]
    inst_label = Gtk.Label(label="Installed", xalign=0)
    inst_label.add_css_class("heading")
    overview.append(inst_label)
    if products:
        for app in products:
            line = app.title
            if app.endpoint:
                line += f"\n{app.endpoint}"
            overview.append(Gtk.Label(label=line, xalign=0, wrap=True))
    elif not helper_apps:
        overview.append(
            Gtk.Label(
                label="Nothing from the installer is on this computer yet.",
                xalign=0,
                wrap=True,
            )
        )
    if helper_apps:
        help_head = Gtk.Label(label=HELPERS_SECTION, xalign=0)
        help_head.add_css_class("heading")
        overview.append(help_head)
        overview.append(Gtk.Label(label=HELPERS_BLURB, xalign=0, wrap=True))
        for app in helper_apps:
            line = workflow_row_title(app.title, app.id, helpers)
            if app.endpoint:
                line += f"\n{app.endpoint}"
            overview.append(Gtk.Label(label=line, xalign=0, wrap=True))
    if missing:
        miss = Gtk.Label(
            label="Not installed. Open Ubuntu AI Installer to add: "
            + ", ".join(a.title for a in missing)
            + ".",
            xalign=0,
            wrap=True,
        )
        miss.add_css_class("dim-label")
        overview.append(miss)

    listen_head = Gtk.Label(label="Who can connect", xalign=0)
    listen_head.add_css_class("heading")
    overview.append(listen_head)
    this_pc = Gtk.CheckButton(label="This computer only")
    network = Gtk.CheckButton(label="Also on my home network", group=this_pc)
    if data.get("bind") == "0.0.0.0":
        network.set_active(True)
    else:
        this_pc.set_active(True)
    overview.append(this_pc)
    overview.append(network)
    warn = Gtk.Label(
        label="Home network lets other devices on your LAN reach local models. Leave this computer only unless you mean it.",
        xalign=0,
        wrap=True,
    )
    warn.add_css_class("dim-label")
    overview.append(warn)
    health_view = Gtk.Label(label="", xalign=0, wrap=True)
    overview.append(health_view)
    stack.add_titled(_scroller(overview), "overview", "Overview")

    settings = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    models = chat_models(t.model_root)
    chat_pairs = [("auto", "Choose for me")] + [(n, n) for n in models]
    settings.append(Gtk.Label(label="Chat model", xalign=0))
    chat_combo = _combo(chat_pairs, str(data.get("chat_model") or "auto"))
    settings.append(chat_combo)
    if not models:
        none = Gtk.Label(label=CHAT_MODEL_NEEDED_CONFIG, xalign=0, wrap=True)
        settings.append(none)
        cta = Gtk.Button(label=CHAT_MODEL_CTA)
        cta.add_css_class("suggested-action")
        cta.connect(
            "clicked",
            lambda *_: _show_chat_model_cta(
                win, lambda: None, CHAT_MODEL_NEEDED_CONFIG
            ),
        )
        settings.append(cta)

    tts_apps = [a for a in installed if a.role == "tts"]
    stt_apps = [a for a in installed if a.role == "stt"]
    tts_combo = None
    stt_combo = None
    if tts_apps:
        settings.append(Gtk.Label(label="Speech out (TTS)", xalign=0))
        tts_combo = _combo(
            [("auto", "Choose for me")] + [(a.id, a.title) for a in tts_apps],
            str(data.get("tts_engine") or "auto"),
        )
        settings.append(tts_combo)
    if stt_apps:
        settings.append(Gtk.Label(label="Speech in (STT)", xalign=0))
        stt_combo = _combo(
            [("auto", "Choose for me")] + [(a.id, a.title) for a in stt_apps],
            str(data.get("stt_engine") or "auto"),
        )
        settings.append(stt_combo)

    settings.append(Gtk.Label(label="GPU mode", xalign=0))
    backend_combo = _combo(
        [(b, BACKEND_LABELS.get(b, b)) for b in backend_choices(hw)],
        str(data.get("primary_backend") or "auto"),
    )
    settings.append(backend_combo)
    stack.add_titled(_scroller(settings), "settings", "Settings")

    advanced = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    advanced.append(Gtk.Label(label="Model folder", xalign=0))
    root_entry = Gtk.Entry(text=str(data.get("model_root") or t.model_root), hexpand=True)
    advanced.append(root_entry)
    advanced.append(
        Gtk.Label(
            label=f"Technical bind is 127.0.0.1 or 0.0.0.0. Current: {data.get('bind') or t.bind}.",
            xalign=0,
            wrap=True,
        )
    )
    advanced.append(Gtk.Label(label="OpenAI-compatible URI", xalign=0))
    uri_entry = Gtk.Entry(
        text=str(data.get("openai_base_url") or ""),
        hexpand=True,
        placeholder_text="http://127.0.0.1:8080/v1",
    )
    advanced.append(uri_entry)
    advanced.append(Gtk.Label(label="API key", xalign=0))
    key_entry = Gtk.Entry(
        text=str(data.get("openai_api_key") or ""),
        hexpand=True,
        placeholder_text="API key if the URI needs one",
    )
    key_entry.set_visibility(False)
    advanced.append(key_entry)
    coding = next((a for a in apps if a.id == "ubuntuai-coding" and a.present), None)
    if coding:
        advanced.append(
            Gtk.Label(
                label="Coding apps should use the Chat URI above.",
                xalign=0,
                wrap=True,
            )
        )
    folders = saved_scan_folders(user)
    advanced.append(Gtk.Label(label="Extra scan folders", xalign=0))
    if folders:
        for folder in folders:
            advanced.append(Gtk.Label(label=str(folder), xalign=0))
    else:
        advanced.append(
            Gtk.Label(
                label="None. Add folders in the installer Weights tab.",
                xalign=0,
                wrap=True,
            )
        )
    raw_view = Gtk.Label(label="", xalign=0, wrap=True)
    advanced.append(raw_view)
    stack.add_titled(_scroller(advanced), "advanced", "Advanced")

    status = Gtk.Label(label="", xalign=0, wrap=True)
    outer.append(stack)
    outer.append(status)
    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    exit_btn = Gtk.Button(label="Exit")
    save_btn = Gtk.Button(label="Save")
    val_btn = Gtk.Button(label="Check health")
    val_btn.add_css_class("suggested-action")
    buttons.append(exit_btn)
    buttons.append(Gtk.Box(hexpand=True))
    buttons.append(save_btn)
    buttons.append(val_btn)
    outer.append(buttons)

    def combo_value(box: Gtk.ComboBoxText | None) -> str:
        if box is None:
            return ""
        value = box.get_active_id() or ""
        if value == "auto":
            return ""
        return value

    def refresh_health() -> None:
        checks = collect(target_for(user), hw)
        health_view.set_label(format_health(checks))
        raw_view.set_label(format_checks(checks))

    def do_save() -> None:
        bind = "0.0.0.0" if network.get_active() else "127.0.0.1"
        uri = uri_entry.get_text().strip()
        if not uri and any(a.id == "ubuntuai-chat" and a.present for a in apps):
            host = "127.0.0.1" if bind != "0.0.0.0" else bind
            uri = f"http://{host}:8080/v1"
        chat = combo_value(chat_combo)

        def write() -> None:
            try:
                path = save_config(
                    user,
                    {
                        "bind": bind,
                        "model_root": root_entry.get_text().strip(),
                        "chat_model": chat,
                        "primary_backend": combo_value(backend_combo),
                        "tts_engine": combo_value(tts_combo),
                        "stt_engine": combo_value(stt_combo),
                        "openai_base_url": uri,
                        "openai_api_key": key_entry.get_text().strip(),
                    },
                )
            except ValueError as exc:
                _show_failure(win, str(exc), "")
                status.set_text(str(exc))
                return
            status.set_text(f"Saved {path}")

        if lemonade_detect():
            _confirm_load_warn(
                win,
                warn_for_chat_model(target_for(user), hw, chat),
                write,
            )
            return
        write()

    def do_validate() -> None:
        refresh_health()
        status.set_text("Health updated.")

    exit_btn.connect("clicked", lambda *_: win.close())
    save_btn.connect("clicked", lambda *_: do_save())
    val_btn.connect("clicked", lambda *_: do_validate())
    refresh_health()
    return outer
