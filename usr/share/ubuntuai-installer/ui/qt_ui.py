"""Qt6 UI for Plasma desktops."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from apply import ApplyError, build_plan, execute_plan, format_plan
from progress import ProgressEvent
from catalog import expand_selection, load_workflows, recommended_ids
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
    app_statuses,
    backend_choices,
    chat_models,
)
from validate import collect, format_checks, format_health
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
    app = QApplication([])
    app.setApplicationName("Ubuntu AI Installer")
    win = QMainWindow()
    win.setWindowTitle(
        "Ubuntu AI Configuration" if mode == "config" else "Ubuntu AI Installer"
    )
    win.resize(800, 720)
    if mode == "config":
        win.setCentralWidget(_config_widget(win))
    else:
        win.setCentralWidget(_installer_widget(win))
    win.show()
    return app.exec()


def _banner(hw) -> QLabel:
    backends = ", ".join(sorted(hw.backends()))
    ram = hw.ram_bytes // (1024**3)
    hybrid = "Hybrid available." if hw.hybrid_ok() else "Hybrid not detected."
    lab = QLabel(
        f"{hw.cpu_name}\n{ram} GiB RAM. Backends: {backends}. {hybrid}"
    )
    lab.setWordWrap(True)
    return lab


def _installer_widget(win: QMainWindow) -> QWidget:
    hw = probe()
    user = guess_user()
    target = target_for(user)
    workflows = load_workflows()
    root = QWidget()
    layout = QVBoxLayout(root)
    title = QLabel("Ubuntu AI Installer")
    font = title.font()
    font.setPointSize(18)
    title.setFont(font)
    layout.addWidget(title)
    layout.addWidget(_banner(hw))
    layout.addWidget(
        QLabel(
            f"Acting for {target.name}. Model root {target.model_root}. Bind {target.bind}."
        )
    )

    tabs = QTabWidget()
    status = QLabel("")
    status.setWordWrap(True)
    tabs.addTab(_qt_workflows(win, hw, user, workflows, status), "Workflows")
    tabs.addTab(_qt_weights(win, user, status), "Weights")
    tabs.addTab(_qt_upgrade(win, user, status), "Updates")
    tabs.addTab(_qt_repair(win, user, status), "Repair")
    layout.addWidget(tabs, 1)
    layout.addWidget(status)
    return root


def _open_apply_progress(win) -> tuple[QDialog, object]:
    dialog = QDialog(win)
    dialog.setWindowTitle("Installing")
    dialog.resize(580, 400)
    layout = QVBoxLayout(dialog)
    english = QLabel("Starting.")
    english.setWordWrap(True)
    layout.addWidget(english)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    layout.addWidget(bar)
    hint = QLabel(
        "This can take a while for large model files. "
        "Open Technical details for apt and download lines."
    )
    hint.setWordWrap(True)
    layout.addWidget(hint)
    layout.addWidget(QLabel("Technical details"))
    log = QTextEdit()
    log.setReadOnly(True)
    layout.addWidget(log, 1)
    close_btn = QPushButton("Close")
    close_btn.setEnabled(False)
    close_btn.clicked.connect(dialog.close)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(close_btn)
    layout.addLayout(row)
    dialog.show()

    def update(ev: object) -> None:
        if not isinstance(ev, ProgressEvent):
            ev = ProgressEvent(str(ev), str(ev))
        bar.setValue(int(max(0.0, min(ev.fraction, 1.0)) * 100))
        english.setText(ev.english)
        if ev.technical:
            log.append(ev.technical)
        if ev.done or ev.failed:
            close_btn.setEnabled(True)

    return dialog, update


def _show_failure(win, english: str, technical: str = "") -> None:
    box = QMessageBox(win)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Install failed")
    box.setText(english)
    if technical.strip():
        box.setDetailedText(technical)
    box.exec()


def _qt_workflows(win, hw, user, workflows, status) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    checks: dict[str, QCheckBox] = {}
    inner = QWidget()
    inner_l = QVBoxLayout(inner)
    inner_l.setContentsMargins(8, 8, 24, 8)
    rec = recommended_ids(workflows, hw)
    for wf in workflows:
        offered = wf.offered(hw) and wf.satisfied(hw)
        summary = wf.summary
        if wf.id in rec and wf.role:
            summary = "Recommended on this machine. " + wf.summary
        if not offered:
            summary = wf.summary + " Unavailable on this hardware."
        elif not wf.ready(hw):
            summary = wf.summary + " Apply will install the missing GPU packages."
        box = QCheckBox(f"{wf.title}\n{summary}")
        box.setChecked(wf.id in rec)
        box.setEnabled(offered and not wf.always)
        checks[wf.id] = box
        inner_l.addWidget(box)
    scroller = QScrollArea()
    scroller.setWidgetResizable(True)
    scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroller.setWidget(inner)
    layout.addWidget(scroller, 1)
    row = QHBoxLayout()
    exit_btn = QPushButton("Exit")
    dry = QPushButton("Dry run")
    apply_btn = QPushButton("Apply")
    row.addWidget(exit_btn)
    row.addStretch(1)
    row.addWidget(dry)
    row.addWidget(apply_btn)
    layout.addLayout(row)

    def do_apply(dry_run: bool) -> None:
        t = target_for(user)
        ids = tuple(wf.id for wf in workflows if checks[wf.id].isChecked())
        actions = build_plan(ids, hw, t)
        if dry_run:
            status.setText("Dry run\n" + format_plan(actions))
            return
        import threading

        _dialog, update = _open_apply_progress(win)
        pending: list[object] = []
        timer = QTimer(_dialog)
        timer.setInterval(80)

        def drain() -> None:
            while pending:
                ev = pending.pop(0)
                update(ev)
                if isinstance(ev, ProgressEvent) and ev.failed:
                    status.setText(ev.english)
                    _show_failure(win, ev.english, ev.technical)
                elif isinstance(ev, ProgressEvent) and ev.done:
                    status.setText(ev.english)

        timer.timeout.connect(drain)
        timer.start()

        def work() -> None:
            try:
                execute_plan(
                    actions,
                    t,
                    hw=hw,
                    dry_run=False,
                    on_progress=pending.append,
                )
                record_installed(user, expand_selection(ids, workflows, hw))
                report = collect(t, hw)
                extra = "\nLog out and back in if group membership just changed."
                pending.append(
                    ProgressEvent(
                        "Finished.",
                        format_checks(report) + extra,
                        1.0,
                        done=True,
                    )
                )
            except ApplyError as exc:
                pending.append(
                    ProgressEvent(exc.english, exc.technical, failed=True)
                )
            except Exception as exc:  # noqa: BLE001
                pending.append(ProgressEvent("Apply failed.", str(exc), failed=True))

        threading.Thread(target=work, daemon=True).start()

    exit_btn.clicked.connect(win.close)
    dry.clicked.connect(lambda: do_apply(True))
    apply_btn.clicked.connect(lambda: do_apply(False))
    return page


def _qt_weights(win, user, status) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    found_layout = QVBoxLayout()
    found_inner = QWidget()
    found_inner.setLayout(found_layout)
    found_scroll = QScrollArea()
    found_scroll.setWidgetResizable(True)
    found_scroll.setWidget(found_inner)
    dl_layout = QVBoxLayout()
    dl_inner = QWidget()
    dl_inner.setLayout(dl_layout)
    dl_scroll = QScrollArea()
    dl_scroll.setWidgetResizable(True)
    dl_scroll.setWidget(dl_inner)
    found_checks: dict[str, QCheckBox] = {}
    found_items: dict = {}
    dl_checks: dict[str, QCheckBox] = {}

    def _clear(box: QVBoxLayout) -> None:
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def refill_found() -> None:
        t = target_for(user)
        _clear(found_layout)
        found_checks.clear()
        found_items.clear()
        roots = scan_roots(t.home, t.model_root, saved_scan_folders(user))
        found = scan(roots, t.model_root)
        new_n = sum(1 for f in found if f.state == "new")
        status.setText(
            f"Found {len(found)} weight files in {len(roots)} folders. {new_n} new."
        )
        for item in found:
            if item.state == "already":
                continue
            key = str(item.path)
            box = QCheckBox(
                f"{item.dest_name} ({human_bytes(item.size)})\n"
                f"{item.state} · {item.fmt} · {item.kind} · {item.subdir} · {item.path}"
            )
            box.setChecked(item.state == "new")
            box.setEnabled(item.state == "new")
            found_checks[key] = box
            found_items[key] = item
            found_layout.addWidget(box)

    def refill_downloads() -> None:
        t = target_for(user)
        _clear(dl_layout)
        dl_checks.clear()
        for model in load_weight_catalog():
            dest = catalog_dest(model, t.model_root)
            have = dest.exists() or dest.is_symlink()
            box = QCheckBox(
                f"{model.title}\n"
                f"{'already in store' if have else human_bytes(model.bytes)}. {model.summary}"
            )
            box.setChecked(model.default and not have)
            box.setEnabled(not have)
            dl_checks[model.id] = box
            dl_layout.addWidget(box)

    folder_list = QVBoxLayout()
    folder_wrap = QWidget()
    folder_wrap.setLayout(folder_list)
    folder_scroll = QScrollArea()
    folder_scroll.setWidgetResizable(True)
    folder_scroll.setWidget(folder_wrap)
    folder_entry = QLineEdit()
    folder_entry.setPlaceholderText("Path to a folder of weights")
    add_folder_btn = QPushButton("Add")
    browse_btn = QPushButton("Browse")
    add_row = QHBoxLayout()
    add_row.addWidget(folder_entry, 1)
    add_row.addWidget(browse_btn)
    add_row.addWidget(add_folder_btn)
    layout.addWidget(QLabel("Folders"))
    layout.addWidget(folder_scroll)
    layout.addLayout(add_row)
    layout.addWidget(QLabel("On disk"))
    layout.addWidget(found_scroll, 1)
    layout.addWidget(QLabel("Download"))
    layout.addWidget(dl_scroll, 1)
    link = QRadioButton("Symlink")
    copy = QRadioButton("Copy")
    move = QRadioButton("Move")
    link.setChecked(True)
    remove_src = QCheckBox("Remove original after checksum match")
    mode_row = QHBoxLayout()
    mode_row.addWidget(link)
    mode_row.addWidget(copy)
    mode_row.addWidget(move)
    mode_row.addStretch(1)
    layout.addLayout(mode_row)
    layout.addWidget(remove_src)

    def on_mode(_checked: bool = False) -> None:
        if link.isChecked():
            remove_src.setEnabled(False)
            remove_src.setChecked(False)
        elif move.isChecked():
            remove_src.setEnabled(False)
            remove_src.setChecked(True)
        else:
            remove_src.setEnabled(True)

    link.toggled.connect(on_mode)
    copy.toggled.connect(on_mode)
    move.toggled.connect(on_mode)
    on_mode()
    row = QHBoxLayout()
    exit_btn = QPushButton("Exit")
    scan_btn = QPushButton("Scan")
    org_btn = QPushButton("Organize")
    dl_btn = QPushButton("Download selected")
    row.addWidget(exit_btn)
    row.addStretch(1)
    row.addWidget(scan_btn)
    row.addWidget(org_btn)
    row.addWidget(dl_btn)
    layout.addLayout(row)

    def refill_folders() -> None:
        _clear(folder_list)
        saved = saved_scan_folders(user)
        if not saved:
            folder_list.addWidget(
                QLabel("No extra folders yet. Built-in locations are always searched.")
            )
            return
        for folder in saved:
            row = QHBoxLayout()
            row.addWidget(QLabel(str(folder)), 1)
            btn = QPushButton("Remove")
            btn.clicked.connect(lambda _=False, path=folder: do_remove_folder(path))
            wrap = QWidget()
            wrap.setLayout(row)
            row.addWidget(btn)
            folder_list.addWidget(wrap)

    def do_add_folder(raw: str) -> None:
        text = raw.strip()
        if not text:
            status.setText("Enter a folder path or use Browse.")
            return
        try:
            folder = add_scan_folder(user, text)
        except ValueError as exc:
            status.setText(str(exc))
            return
        folder_entry.clear()
        status.setText(f"Added {folder}")
        refill_folders()
        refill_found()

    def do_remove_folder(path) -> None:
        remove_scan_folder(user, path)
        status.setText(f"Removed {path}")
        refill_folders()
        refill_found()

    def do_browse() -> None:
        picked = QFileDialog.getExistingDirectory(win, "Folder to scan")
        if picked:
            folder_entry.setText(picked)
            do_add_folder(picked)

    def do_organize() -> None:
        t = target_for(user)
        picked = tuple(
            found_items[k] for k, cb in found_checks.items() if cb.isChecked()
        )
        if not picked:
            status.setText("No new weights selected.")
            return
        if copy.isChecked():
            mode = "copy"
        elif move.isChecked():
            mode = "move"
        else:
            mode = "link"
        log = organize(
            picked,
            t.model_root,
            mode=mode,
            remove_source=remove_src.isChecked(),
            uid=t.uid,
            gid=t.gid,
        )
        status.setText("\n".join(log))
        refill_found()

    def do_download() -> None:
        t = target_for(user)
        catalog = {m.id: m for m in load_weight_catalog()}
        ids = [mid for mid, cb in dl_checks.items() if cb.isChecked()]
        if not ids:
            status.setText("No catalog weights selected.")
            return

        lines = []
        for mid in ids:
            try:
                lines.append(
                    download(catalog[mid], t.model_root, uid=t.uid, gid=t.gid)
                )
            except Exception as exc:  # noqa: BLE001
                english = (
                    "A downloaded model file did not match its published checksum. "
                    "The broken file was not kept."
                    if "mismatch" in str(exc).lower()
                    else "Could not download a selected model file."
                )
                lines.append(english)
                _show_failure(win, english, str(exc))
        status.setText("\n".join(lines))
        refill_downloads()
        refill_found()

    exit_btn.clicked.connect(win.close)
    scan_btn.clicked.connect(refill_found)
    org_btn.clicked.connect(do_organize)
    dl_btn.clicked.connect(do_download)
    add_folder_btn.clicked.connect(lambda: do_add_folder(folder_entry.text()))
    folder_entry.returnPressed.connect(lambda: do_add_folder(folder_entry.text()))
    browse_btn.clicked.connect(do_browse)
    refill_folders()
    refill_found()
    refill_downloads()
    return page


def _qt_repair(win, user, status) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    cfg = load_config(user)
    hint = QLabel(
        "Diagnose with local checks and the checksum published on each model page. "
        "The OpenAI path needs a server URI. Many providers also need an API key. "
        "The plan is shown in plain language. Nothing is applied until you approve."
    )
    hint.setWordWrap(True)
    layout.addWidget(hint)
    classical = QRadioButton("Classical")
    openai = QRadioButton("OpenAI URI")
    local = QRadioButton("Local model")
    classical.setChecked(True)
    row = QHBoxLayout()
    row.addWidget(classical)
    row.addWidget(openai)
    row.addWidget(local)
    row.addStretch(1)
    layout.addLayout(row)
    uri = QLineEdit(
        str(cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or "")
    )
    uri.setPlaceholderText("https://api.openai.com/v1")
    key = QLineEdit(
        str(cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "")
    )
    key.setEchoMode(QLineEdit.EchoMode.Password)
    key.setPlaceholderText("API key for the URI")
    layout.addWidget(QLabel("URI"))
    layout.addWidget(uri)
    layout.addWidget(QLabel("API key"))
    layout.addWidget(key)
    plan_view = QTextEdit()
    plan_view.setReadOnly(True)
    layout.addWidget(plan_view, 1)
    comment = QLineEdit()
    comment.setPlaceholderText("Request changes to the plan")
    layout.addWidget(comment)
    btns = QHBoxLayout()
    exit_btn = QPushButton("Exit")
    diag_btn = QPushButton("Diagnose")
    revise_btn = QPushButton("Request changes")
    approve_btn = QPushButton("Approve")
    btns.addWidget(exit_btn)
    btns.addStretch(1)
    btns.addWidget(diag_btn)
    btns.addWidget(revise_btn)
    btns.addWidget(approve_btn)
    layout.addLayout(btns)

    def advisor_name() -> str:
        if openai.isChecked():
            return "openai"
        if local.isChecked():
            return "local"
        return "classical"

    def do_diagnose(note: str) -> None:
        save_config(
            user,
            {
                "openai_base_url": uri.text().strip(),
                "openai_api_key": key.text().strip(),
                "repair_advisor": advisor_name(),
            },
        )
        try:
            _diag, plan = build_plan_for(
                user,
                advisor_name(),
                comment=note,
                openai_base_url=uri.text().strip(),
                openai_api_key=key.text().strip(),
            )
            plan_view.setPlainText(format_repair_plan(plan))
            status.setText("Review the plan. Approve only if you accept every step.")
        except Exception as exc:  # noqa: BLE001
            plan_view.setPlainText(f"Diagnose failed. {exc}")

    def do_approve() -> None:
        saved = load_saved_plan(user)
        if saved is None:
            status.setText("No plan to approve. Diagnose first.")
            return
        try:
            log = execute_plan_steps(user, saved.get("plan"))
            plan_view.setPlainText("\n".join(log))
            status.setText("Repair steps finished.")
        except ApplyError as exc:
            status.setText(exc.english)
            _show_failure(win, exc.english, exc.technical)
        except Exception as exc:  # noqa: BLE001
            status.setText(f"Repair failed. {exc}")
            _show_failure(win, "Repair failed.", str(exc))

    exit_btn.clicked.connect(win.close)
    diag_btn.clicked.connect(lambda: do_diagnose(""))
    revise_btn.clicked.connect(lambda: do_diagnose(comment.text().strip()))
    approve_btn.clicked.connect(do_approve)
    saved = load_saved_plan(user)
    if saved and saved.get("plan"):
        plan_view.setPlainText(
            str(saved.get("english") or format_repair_plan(saved["plan"]))
        )
    return page


def _qt_upgrade(win, user, status) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    hint = QLabel(
        "Upgrade vendor runtimes that are not apt or snap, refresh catalog "
        "models that drifted, and tune Lemonade for very large GGUF files. "
        "Diagnose writes a plan. Nothing is applied until you approve."
    )
    hint.setWordWrap(True)
    layout.addWidget(hint)
    plan_view = QTextEdit()
    plan_view.setReadOnly(True)
    layout.addWidget(plan_view, 1)
    btns = QHBoxLayout()
    exit_btn = QPushButton("Exit")
    diag_btn = QPushButton("Diagnose")
    approve_btn = QPushButton("Approve")
    btns.addWidget(exit_btn)
    btns.addStretch(1)
    btns.addWidget(diag_btn)
    btns.addWidget(approve_btn)
    layout.addLayout(btns)

    def do_diagnose() -> None:
        try:
            _diag, plan = build_upgrade_plan(user)
            plan_view.setPlainText(format_upgrade_plan(plan))
            status.setText("Review the update plan. Approve only if you accept every step.")
        except Exception as exc:  # noqa: BLE001
            plan_view.setPlainText(f"Diagnose failed. {exc}")

    def do_approve() -> None:
        saved = load_upgrade_plan(user)
        if saved is None:
            status.setText("No update plan to approve. Diagnose first.")
            return
        try:
            log = execute_upgrade_steps(user, saved.get("plan"))
            plan_view.setPlainText("\n".join(log))
            status.setText("Updates finished.")
        except ApplyError as exc:
            status.setText(exc.english)
            _show_failure(win, exc.english, exc.technical)
        except Exception as exc:  # noqa: BLE001
            status.setText(f"Update failed. {exc}")
            _show_failure(win, "Update failed.", str(exc))

    exit_btn.clicked.connect(win.close)
    diag_btn.clicked.connect(do_diagnose)
    approve_btn.clicked.connect(do_approve)
    saved = load_upgrade_plan(user)
    if saved and saved.get("plan"):
        plan_view.setPlainText(
            str(saved.get("english") or format_upgrade_plan(saved["plan"]))
        )
    return page


def _qcombo(pairs: list[tuple[str, str]], current: str) -> QComboBox:
    box = QComboBox()
    ids: list[str] = []
    for key, label in pairs:
        box.addItem(label, key)
        ids.append(key)
    if current in ids:
        box.setCurrentIndex(ids.index(current))
    return box


def _combo_value(box: QComboBox | None) -> str:
    if box is None:
        return ""
    value = str(box.currentData() or "")
    if value == "auto":
        return ""
    return value


def _config_widget(win: QMainWindow) -> QWidget:
    user = guess_user()
    t = target_for(user)
    data = load_config(user)
    hw = probe()
    apps = app_statuses(user, t)
    root = QWidget()
    layout = QVBoxLayout(root)
    title = QLabel("Ubuntu AI Configuration")
    font = title.font()
    font.setPointSize(18)
    title.setFont(font)
    layout.addWidget(title)
    hint = QLabel(
        "Settings for the apps Ubuntu AI Installer put on this computer. "
        "Overview is the short path. Advanced has paths and API fields."
    )
    hint.setWordWrap(True)
    layout.addWidget(hint)
    layout.addWidget(_banner(hw))
    tabs = QTabWidget()

    overview = QWidget()
    ov = QVBoxLayout(overview)
    installed = [a for a in apps if a.present]
    missing = [a for a in apps if not a.present]
    ov.addWidget(QLabel("Installed"))
    if installed:
        for app in installed:
            line = app.title
            if app.endpoint:
                line += f"\n{app.endpoint}"
            lab = QLabel(line)
            lab.setWordWrap(True)
            ov.addWidget(lab)
    else:
        ov.addWidget(QLabel("Nothing from the installer is on this computer yet."))
    if missing:
        miss = QLabel(
            "Not installed. Open Ubuntu AI Installer to add: "
            + ", ".join(a.title for a in missing)
            + "."
        )
        miss.setWordWrap(True)
        ov.addWidget(miss)
    ov.addWidget(QLabel("Who can connect"))
    this_pc = QRadioButton("This computer only")
    network = QRadioButton("Also on my home network")
    group = QButtonGroup(overview)
    group.addButton(this_pc)
    group.addButton(network)
    if data.get("bind") == "0.0.0.0":
        network.setChecked(True)
    else:
        this_pc.setChecked(True)
    ov.addWidget(this_pc)
    ov.addWidget(network)
    warn = QLabel(
        "Home network lets other devices on your LAN reach local models. "
        "Leave this computer only unless you mean it."
    )
    warn.setWordWrap(True)
    ov.addWidget(warn)
    health_view = QLabel("")
    health_view.setWordWrap(True)
    ov.addWidget(health_view, 1)
    tabs.addTab(overview, "Overview")

    settings = QWidget()
    st = QVBoxLayout(settings)
    models = chat_models(t.model_root)
    chat_combo = _qcombo(
        [("auto", "Choose for me")] + [(n, n) for n in models],
        str(data.get("chat_model") or "auto"),
    )
    st.addWidget(QLabel("Chat model"))
    st.addWidget(chat_combo)
    if not models:
        none = QLabel(
            "No GGUF chat files in the model folder yet. Use the installer Weights tab."
        )
        none.setWordWrap(True)
        st.addWidget(none)
    tts_combo = None
    stt_combo = None
    tts_apps = [a for a in installed if a.role == "tts"]
    stt_apps = [a for a in installed if a.role == "stt"]
    if tts_apps:
        st.addWidget(QLabel("Speech out (TTS)"))
        tts_combo = _qcombo(
            [("auto", "Choose for me")] + [(a.id, a.title) for a in tts_apps],
            str(data.get("tts_engine") or "auto"),
        )
        st.addWidget(tts_combo)
    if stt_apps:
        st.addWidget(QLabel("Speech in (STT)"))
        stt_combo = _qcombo(
            [("auto", "Choose for me")] + [(a.id, a.title) for a in stt_apps],
            str(data.get("stt_engine") or "auto"),
        )
        st.addWidget(stt_combo)
    st.addWidget(QLabel("GPU mode"))
    backend_combo = _qcombo(
        [(b, BACKEND_LABELS.get(b, b)) for b in backend_choices(hw)],
        str(data.get("primary_backend") or "auto"),
    )
    st.addWidget(backend_combo)
    st.addStretch(1)
    tabs.addTab(settings, "Settings")

    advanced = QWidget()
    adv = QVBoxLayout(advanced)
    adv.addWidget(QLabel("Model folder"))
    root_entry = QLineEdit(str(data.get("model_root") or t.model_root))
    adv.addWidget(root_entry)
    adv.addWidget(
        QLabel(
            f"Technical bind is 127.0.0.1 or 0.0.0.0. Current: {data.get('bind') or t.bind}."
        )
    )
    adv.addWidget(QLabel("OpenAI-compatible URI"))
    uri_entry = QLineEdit(str(data.get("openai_base_url") or ""))
    uri_entry.setPlaceholderText("http://127.0.0.1:8080/v1")
    adv.addWidget(uri_entry)
    adv.addWidget(QLabel("API key"))
    key_entry = QLineEdit(str(data.get("openai_api_key") or ""))
    key_entry.setEchoMode(QLineEdit.EchoMode.Password)
    key_entry.setPlaceholderText("API key if the URI needs one")
    adv.addWidget(key_entry)
    folders = saved_scan_folders(user)
    adv.addWidget(QLabel("Extra scan folders"))
    if folders:
        for folder in folders:
            adv.addWidget(QLabel(str(folder)))
    else:
        adv.addWidget(QLabel("None. Add folders in the installer Weights tab."))
    raw_view = QLabel("")
    raw_view.setWordWrap(True)
    adv.addWidget(raw_view, 1)
    tabs.addTab(advanced, "Advanced")
    layout.addWidget(tabs, 1)

    status = QLabel("")
    status.setWordWrap(True)
    layout.addWidget(status)
    btns = QHBoxLayout()
    exit_btn = QPushButton("Exit")
    save_btn = QPushButton("Save")
    val_btn = QPushButton("Check health")
    btns.addWidget(exit_btn)
    btns.addStretch(1)
    btns.addWidget(save_btn)
    btns.addWidget(val_btn)
    layout.addLayout(btns)

    def refresh_health() -> None:
        checks = collect(target_for(user), hw)
        health_view.setText(format_health(checks))
        raw_view.setText(format_checks(checks))

    def do_save() -> None:
        bind = "0.0.0.0" if network.isChecked() else "127.0.0.1"
        uri = uri_entry.text().strip()
        if not uri and any(a.id == "ubuntuai-chat" and a.present for a in apps):
            host = "127.0.0.1" if bind != "0.0.0.0" else bind
            uri = f"http://{host}:8080/v1"
        try:
            path = save_config(
                user,
                {
                    "bind": bind,
                    "model_root": root_entry.text().strip(),
                    "chat_model": _combo_value(chat_combo),
                    "primary_backend": _combo_value(backend_combo),
                    "tts_engine": _combo_value(tts_combo),
                    "stt_engine": _combo_value(stt_combo),
                    "openai_base_url": uri,
                    "openai_api_key": key_entry.text().strip(),
                },
            )
        except ValueError as exc:
            _show_failure(win, str(exc), "")
            status.setText(str(exc))
            return
        status.setText(f"Saved {path}")

    def do_validate() -> None:
        refresh_health()
        status.setText("Health updated.")

    exit_btn.clicked.connect(win.close)
    save_btn.clicked.connect(do_save)
    val_btn.clicked.connect(do_validate)
    refresh_health()
    return root
