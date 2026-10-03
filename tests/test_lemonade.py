from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import unittest
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from support import PKG  # noqa: F401

from domain import Action, Device, Hardware, UserTarget
from lemonade import (
    APPLY_PUBLISH_VERB,
    BindMount,
    HUGE_GGUF_BYTES,
    INSTALLER_WRITTEN_VALUES,
    LLAMACPP_MMAP_ARGS,
    LOAD_RISK_POLICY,
    CONFIG_READY_TIMEOUT,
    OWNED_UNIT_DESC,
    STAGE_DIR,
    DEFAULT_TUNING_LEDGER_DIR,
    _daemon_was_running,
    _installer_wrote_value,
    _values_equal,
    _drop_bind_unit,
    _mkdir_nofollow,
    _run,
    _snap_mount_plan,
    _wait_for_config,
    _unit_is_ours,
    _where_from_unit,
    bind_mounts,
    cli_tuning_parts,
    detect,
    apply_tuning,
    cover_unit_text,
    embeddings_cover,
    embeddings_mount,
    extra_dir,
    gguf_sources,
    is_owned_lemonade_where,
    largest_gguf_bytes,
    leftover_owned_binds,
    model_gguf_bytes,
    load_risk,
    load_tuning,
    mount_unit_path,
    mount_unit_text,
    owned_bind_wheres,
    propagation_command,
    publish,
    publish_plan,
    quote_unit_path,
    read_tuning_ledger,
    report_load_tuning,
    risk_english,
    tuning_ledger_path,
    verify_tuning,
    write_tuning_ledger,
)
from paths import (
    SNAP_LEMONADE_MODELS,
    is_home_path,
    lemonade_extra_models_dir,
)


_REAL_SUBPROCESS_RUN = subprocess.run


def _refuse_host_systemctl(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
    cmd = args[0] if args else kwargs.get("args", ())
    argv = list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)]
    if argv and Path(str(argv[0])).name == "systemctl":
        raise AssertionError(
            "a test reached the real systemctl: " + " ".join(str(part) for part in argv)
        )
    return _REAL_SUBPROCESS_RUN(*args, **kwargs)  # type: ignore[arg-type]


_REAL_URLOPEN = urllib.request.urlopen
_REAL_CREATE_CONNECTION = socket.create_connection


def _request_url(req: object) -> str:
    return str(getattr(req, "full_url", "") or req)


def _refuse_lemonade_http(req: object, *args: object, **kwargs: object) -> object:
    url = _request_url(req)
    port = getattr(req, "port", None)
    if port == 13305 or ":13305" in url:
        raise AssertionError(f"a test reached Lemonade on port 13305: {url}")
    return _REAL_URLOPEN(req, *args, **kwargs)  # type: ignore[arg-type]


def _refuse_lemonade_socket(address: object, *args: object, **kwargs: object) -> socket.socket:
    port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
    if port == 13305:
        raise AssertionError(f"a test reached Lemonade socket {address}")
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)  # type: ignore[arg-type]


def setUpModule() -> None:
    global _systemctl_guard, _http_guard, _socket_guard, _state_guard
    _systemctl_guard = patch("lemonade.subprocess.run", side_effect=_refuse_host_systemctl)
    _http_guard = patch("lemonade.urllib.request.urlopen", side_effect=_refuse_lemonade_http)
    _socket_guard = patch("socket.create_connection", side_effect=_refuse_lemonade_socket)
    # Publish creates /var/lib/ubuntuai before it unmounts. Unit tests are not root.
    _state_guard = patch("lemonade._prepare_state_dirs")
    _systemctl_guard.start()
    _http_guard.start()
    _socket_guard.start()
    _state_guard.start()


def tearDownModule() -> None:
    _state_guard.stop()
    _socket_guard.stop()
    _http_guard.stop()
    _systemctl_guard.stop()


@contextmanager
def _quiet_daemon():
    """Keep publish tests off the host lemonade daemon."""
    with (
        patch("lemonade._daemon_was_running", return_value=False),
        patch("lemonade._stop_daemon"),
        patch("lemonade._start_daemon"),
        patch("lemonade._wait_for_config"),
    ):
        yield


def _target(
    home: Path,
    extra: tuple[Path, ...] | None = None,
    model_root: Path | None = None,
) -> UserTarget:
    return UserTarget(
        name="owner",
        uid=1000,
        gid=1000,
        home=home,
        model_root=model_root or (home / "Models"),
        extra_model_paths=extra if extra is not None else (home / "AI models",),
        bind="127.0.0.1",
    )


def _write_gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"G" * 2048)
    return path


def _merged_factory() -> dict:
    """Shape of GET /internal/config after defaults.json is merged in."""
    return {
        "ctx_size": -1,
        "global_timeout": 600,
        "max_loaded_models": 1,
        "llamacpp": {"backend": "auto", "args": "", "vulkan_args": ""},
        "port": 13305,
        "extra_models_dir": "",
        "vulkan_bin": "",
    }


def _seed_unit(
    unit_dir: Path,
    what: Path,
    where: Path,
    name: str,
    ours: bool = True,
) -> Path:
    text = mount_unit_text(what, where)
    if not ours:
        text = text.replace(f"Description={OWNED_UNIT_DESC}", "Description=User bind")
    path = unit_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def _systemd_analyze_verify(
    analyze: str,
    escape: str,
    text: str,
    where: Path,
) -> tuple[int, str]:
    """Verify a mount unit in one transaction with local-fs.target."""
    escaped = subprocess.run(
        [escape, "-p", "--suffix=mount", str(where)],
        capture_output=True,
        text=True,
        check=False,
    )
    name = (escaped.stdout or "").strip()
    if escaped.returncode != 0 or not name:
        raise RuntimeError(escaped.stderr or "systemd-escape failed")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [analyze, "--man=no", "verify", str(path), "local-fs.target"],
            capture_output=True,
            text=True,
            check=False,
        )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


class LemonadePublishTests(unittest.TestCase):
    def test_symlink_store_is_not_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            real.mkdir()
            blob = real / "tiny.gguf"
            blob.write_bytes(b"G" * 2048)
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "tiny.gguf").symlink_to(blob)
            src = gguf_sources(_target(home))
            self.assertEqual(src, (real.resolve(),))
            self.assertNotIn((home / "Models" / "gguf").resolve(), src)

    def test_real_store_gguf_is_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "tiny.gguf").write_bytes(b"G" * 2048)
            src = gguf_sources(_target(home))
            self.assertIn(store.resolve(), src)

    def test_store_symlink_outside_extra_paths_is_followed(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            downloads = home / "Downloads"
            blob = _write_gguf(downloads / "outside.gguf")
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "outside.gguf").symlink_to(blob)
            src = gguf_sources(_target(home, extra=()))
            self.assertEqual(src, (downloads.resolve(),))
            self.assertNotIn(store.resolve(), src)

    def test_models_symlink_to_ai_models_top_level_gguf(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            real.mkdir()
            _write_gguf(real / "tiny.gguf")
            models = home / "Models"
            models.symlink_to(real)
            src = gguf_sources(_target(home, extra=(), model_root=models))
            self.assertEqual(src, (real.resolve(),))

    def test_hf_tree_is_not_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            hf = extra / "some-model"
            hf.mkdir(parents=True)
            (hf / "config.json").write_text("{}", encoding="utf-8")
            (hf / "model.safetensors").write_bytes(b"S" * 64)
            src = gguf_sources(_target(home, extra=(extra,)))
            self.assertEqual(src, ())

    def test_diffusers_tree_is_not_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            diff = extra / "pipe"
            (diff / "unet").mkdir(parents=True)
            (diff / "model_index.json").write_text("{}", encoding="utf-8")
            src = gguf_sources(_target(home, extra=(extra,)))
            self.assertEqual(src, ())

    def test_shard_folder_stays_one_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            shard = extra / "Qwen"
            _write_gguf(shard / "qwen-00001-of-00002.gguf")
            _write_gguf(shard / "qwen-00002-of-00002.gguf")
            src = gguf_sources(_target(home, extra=(extra,)))
            self.assertEqual(src, (extra.resolve(),))

    def test_nested_store_under_extra_collapses(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            _write_gguf(extra / "a.gguf")
            store = extra / "gguf"
            _write_gguf(store / "b.gguf")
            src = gguf_sources(_target(home, extra=(extra,), model_root=extra))
            self.assertEqual(src, (extra.resolve(),))

    def test_nested_models_gguf_under_ai_models_collapses(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            _write_gguf(extra / "outer.gguf")
            store = extra / "Models"
            _write_gguf(store / "gguf" / "inner.gguf")
            src = gguf_sources(_target(home, extra=(extra,), model_root=store))
            self.assertEqual(src, (extra.resolve(),))
            dest = extra_dir("snap")
            self.assertEqual(
                bind_mounts(src, dest),
                (BindMount(extra.resolve(), dest / "chat" / "src0"),),
            )

    def test_ai_models_root_outside_gguf_is_one_source(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "gguf" / "store.gguf")
            _write_gguf(root / "Gemma4-26B-A4B-GGUF" / "gemma.gguf")
            _write_gguf(root / "HauhauCS" / "hau.gguf")
            target = _target(home, extra=(), model_root=root)
            first = gguf_sources(target)
            second = gguf_sources(target)
            self.assertEqual(first, (root.resolve(),))
            self.assertEqual(first, second)
            self.assertNotIn((root / "gguf").resolve(), first)
            dest = Path(tmp) / "ubuntuai-models"
            mounts = bind_mounts(first, dest)
            self.assertEqual(
                mounts,
                (BindMount(root.resolve(), dest / "chat" / "src0"),),
            )
            self.assertEqual(bind_mounts(second, dest), mounts)

    def test_store_only_model_root_stays_on_gguf(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            store = root / "gguf"
            _write_gguf(store / "store.gguf")
            (root / "hf").mkdir(parents=True)
            src = gguf_sources(_target(home, extra=(), model_root=root))
            self.assertEqual(src, (store.resolve(),))
            self.assertNotIn(root.resolve(), src)

    def test_symlink_inside_gguf_does_not_widen_model_root(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            outside = _write_gguf(home / "Downloads" / "outside.gguf")
            store = root / "gguf"
            store.mkdir(parents=True)
            (store / "outside.gguf").symlink_to(outside)
            src = gguf_sources(_target(home, extra=(), model_root=root))
            self.assertEqual(src, ((home / "Downloads").resolve(),))
            self.assertNotIn(root.resolve(), src)
            self.assertNotIn(store.resolve(), src)

    def test_outside_gguf_srcn_stays_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            _write_gguf(extra / "Gemma4-26B-A4B-GGUF" / "gemma.gguf")
            _write_gguf(extra / "HauhauCS" / "hau.gguf")
            _write_gguf(extra / "gguf" / "store.gguf")
            other = home / "bucket"
            _write_gguf(other / "Gemma" / "a.gguf")
            _write_gguf(other / "gguf" / "b.gguf")
            target = _target(home, extra=(extra,), model_root=other)
            first = gguf_sources(target)
            second = gguf_sources(target)
            self.assertEqual(first, (other.resolve(), extra.resolve()))
            self.assertEqual(first, second)
            self.assertNotIn((extra / "gguf").resolve(), first)
            self.assertNotIn((other / "gguf").resolve(), first)
            dest = Path(tmp) / "ubuntuai-models"
            mounts = bind_mounts(first, dest)
            self.assertEqual(
                tuple((m.what, m.where.name) for m in mounts),
                (
                    (other.resolve(), "src0"),
                    (extra.resolve(), "src1"),
                ),
            )
            self.assertEqual(bind_mounts(gguf_sources(target), dest), mounts)

    def test_foreign_model_root_is_excluded(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "Gemma4-26B-A4B-GGUF" / "gemma.gguf")
            _write_gguf(root / "gguf" / "store.gguf")
            local = home / "local-weights"
            _write_gguf(local / "ok.gguf")
            with patch("weights._FOREIGN_PREFIXES", (root.resolve(),)):
                src = gguf_sources(
                    _target(home, extra=(local,), model_root=root)
                )
            self.assertEqual(src, (local.resolve(),))
            self.assertNotIn(root.resolve(), src)

    def test_foreign_lifted_tree_is_excluded(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            foreign = home / "usb"
            blob = _write_gguf(foreign / "outside.gguf")
            store = home / "Models" / "gguf"
            store.mkdir(parents=True)
            (store / "outside.gguf").symlink_to(blob)
            with patch("weights._FOREIGN_PREFIXES", (foreign.resolve(),)):
                src = gguf_sources(_target(home, extra=()))
            self.assertEqual(src, ())
            self.assertNotIn(foreign.resolve(), src)

    def test_too_wide_model_root_is_excluded(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            _write_gguf(home / "Gemma" / "a.gguf")
            _write_gguf(home / "gguf" / "b.gguf")
            src = gguf_sources(_target(home, extra=(), model_root=home))
            self.assertEqual(src, ())

    def test_snap_extra_dir(self) -> None:
        dest = extra_dir("snap")
        self.assertEqual(dest, Path("/var/snap/lemonade-server/common/ubuntuai-models"))
        self.assertEqual(dest, SNAP_LEMONADE_MODELS)
        self.assertEqual(dest, lemonade_extra_models_dir("snap"))
        self.assertFalse(is_home_path(dest))

    def test_cli_extra_dir_is_empty(self) -> None:
        self.assertEqual(extra_dir("cli"), Path())
        self.assertEqual(lemonade_extra_models_dir("deb"), Path())

    def test_home_path_helper(self) -> None:
        self.assertTrue(is_home_path(Path("/home/x/Models")))
        self.assertTrue(is_home_path(Path("/home/x/AI models")))
        self.assertFalse(is_home_path(SNAP_LEMONADE_MODELS))
        self.assertFalse(is_home_path(Path("/tmp/models")))

    def test_detect_snap_when_common_exists(self) -> None:
        with patch("lemonade.SNAP_COMMON") as common:
            common.is_dir.return_value = True
            self.assertEqual(detect(), "snap")

    def test_bind_mounts_single_source_uses_srcn(self) -> None:
        dest = SNAP_LEMONADE_MODELS
        src = Path("/var/tmp/ai-models")
        self.assertEqual(
            bind_mounts((src,), dest),
            (BindMount(src.resolve(), dest / "chat" / "src0"),),
        )

    def test_bind_mounts_many_sources_under_chat(self) -> None:
        dest = SNAP_LEMONADE_MODELS
        a = Path("/var/tmp/a")
        b = Path("/var/tmp/b")
        self.assertEqual(
            bind_mounts((a, b), dest),
            (
                BindMount(a.resolve(), dest / "chat" / "src0"),
                BindMount(b.resolve(), dest / "chat" / "src1"),
            ),
        )

    def test_bind_mounts_nested_src_is_dropped(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            extra.mkdir()
            nested = extra / "Models" / "gguf"
            nested.mkdir(parents=True)
            mounts = bind_mounts((extra, nested), dest)
            self.assertEqual(
                mounts, (BindMount(extra.resolve(), dest / "chat" / "src0"),)
            )
            self.assertFalse(any(part == "src1" for m in mounts for part in m.where.parts))

    def test_bind_mounts_symlink_nested_src_is_dropped(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            extra.mkdir()
            (extra / "gguf").mkdir()
            models = Path(tmp) / "Models"
            models.symlink_to(extra)
            mounts = bind_mounts((extra, models / "gguf"), dest)
            self.assertEqual(
                mounts, (BindMount(extra.resolve(), dest / "chat" / "src0"),)
            )
            self.assertEqual(len(mounts), 1)
            self.assertEqual(mounts[0].where, dest / "chat" / "src0")

    def test_bind_mounts_sibling_trees_keep_srcn(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            store = Path(tmp) / "Models" / "gguf"
            extra.mkdir()
            store.mkdir(parents=True)
            mounts = bind_mounts((extra, store), dest)
            self.assertEqual(
                mounts,
                (
                    BindMount(extra.resolve(), dest / "chat" / "src0"),
                    BindMount(store.resolve(), dest / "chat" / "src1"),
                ),
            )

    def test_mount_unit_writes_unquoted_paths(self) -> None:
        what = Path("/tmp/AI 100% models")
        where = SNAP_LEMONADE_MODELS / "chat" / "src0"
        text = mount_unit_text(what, where)
        self.assertEqual(mount_unit_path(what), "/tmp/AI 100%% models")
        self.assertIn("What=/tmp/AI 100%% models\n", text)
        self.assertIn('RequiresMountsFor="/tmp/AI 100%% models"', text)
        self.assertIn(f"Where={where}\n", text)
        self.assertNotIn('What="', text)
        self.assertNotIn('Where="', text)
        self.assertIn("Type=none", text)
        self.assertIn("Options=bind,nofail", text)
        self.assertEqual(_where_from_unit(text), where)
        self.assertEqual(
            _where_from_unit("Where=/tmp/100%% models\n"),
            Path("/tmp/100% models"),
        )

    def test_mount_unit_has_no_local_fs_ordering_cycle(self) -> None:
        what = Path("/tmp/AI models")
        where = SNAP_LEMONADE_MODELS / "chat" / "src0"
        text = mount_unit_text(what, where)
        # Default dependencies add Before=local-fs.target for a /var mount.
        self.assertNotIn("DefaultDependencies=no", text)
        self.assertNotIn("After=local-fs.target", text)
        self.assertIn(f"RequiresMountsFor={quote_unit_path(what)}", text)
        self.assertIn('RequiresMountsFor="/tmp/AI models"', text)
        self.assertIn("What=/tmp/AI models\n", text)
        self.assertIn(f"Where={where}\n", text)
        self.assertNotIn('What="', text)
        self.assertNotIn('Where="', text)
        self.assertIn("Options=bind,nofail", text)
        self.assertIn("Before=snap.lemonade-server.daemon.service", text)
        self.assertIn(f"Description={OWNED_UNIT_DESC}", text)
        self.assertIn("WantedBy=multi-user.target", text)

    def test_mount_unit_systemd_analyze_verify_offline(self) -> None:
        analyze = shutil.which("systemd-analyze")
        if not analyze:
            self.skipTest("systemd-analyze is not installed")
        escape = shutil.which("systemd-escape")
        if not escape:
            self.skipTest("systemd-escape is not installed")
        what = Path("/tmp/AI models")
        where = SNAP_LEMONADE_MODELS / "chat" / "src0"
        code, output = _systemd_analyze_verify(
            analyze, escape, mount_unit_text(what, where), where
        )
        self.assertNotIn("not absolute", output)
        self.assertNotIn("ordering cycle", output)
        self.assertEqual(code, 0, output)

    def test_legacy_quoted_unit_is_still_owned(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            source = Path(tmp) / "AI models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            legacy_where = dest / "chat" / "src0"
            new_where = dest / "chat" / "src1"
            legacy = (
                "[Unit]\n"
                f"Description={OWNED_UNIT_DESC}\n"
                "After=local-fs.target\n"
                "Before=snap.lemonade-server.daemon.service\n"
                "\n"
                "[Mount]\n"
                f"What={quote_unit_path(source)}\n"
                f"Where={quote_unit_path(legacy_where)}\n"
                "Type=none\n"
                "Options=bind\n"
                "\n"
                "[Install]\n"
                "WantedBy=multi-user.target\n"
            )
            legacy_path = unit_dir / "legacy.mount"
            legacy_path.write_text(legacy, encoding="utf-8")
            new_path = unit_dir / "new.mount"
            new_path.write_text(mount_unit_text(source, new_where), encoding="utf-8")
            self.assertTrue(_unit_is_ours(legacy_path))
            self.assertTrue(_unit_is_ours(new_path))
            legacy_parsed = _where_from_unit(legacy)
            new_parsed = _where_from_unit(new_path.read_text(encoding="utf-8"))
            self.assertEqual(legacy_parsed, legacy_where)
            self.assertEqual(new_parsed, new_where)
            self.assertTrue(is_owned_lemonade_where(dest, legacy_parsed))
            self.assertTrue(is_owned_lemonade_where(dest, new_parsed))
            leftovers = leftover_owned_binds(dest, (), unit_dir=unit_dir)
            self.assertEqual(
                set(leftovers),
                {legacy_where.resolve(), new_where.resolve()},
            )

    def test_publish_snap_sets_snap_common_not_home(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            real.mkdir()
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            binds: list[tuple[Path, Path]] = []
            extras: list[Path] = []

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir", side_effect=extras.append),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                msg = publish(_target(home))
            self.assertEqual(binds, [(real.resolve(), dest / "chat" / "src0")])
            self.assertEqual(extras, [dest])
            self.assertNotEqual(extras[0], real.resolve())
            self.assertTrue(str(extras[0]).endswith("ubuntuai-models"))
            self.assertIn(str(dest), msg)

    def test_publish_snap_nested_src_is_one_bind(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            _write_gguf(extra / "outer.gguf")
            store = extra / "Models"
            _write_gguf(store / "gguf" / "inner.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            binds: list[tuple[Path, Path]] = []

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                msg = publish(_target(home, extra=(extra,), model_root=store))
            self.assertEqual(binds, [(extra.resolve(), dest / "chat" / "src0")])
            self.assertFalse(any("src1" in str(where) for _, where in binds))
            self.assertIn(str(dest), msg)
            self.assertNotIn("2 trees", msg)

    def test_publish_snap_sibling_trees_use_srcn(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            store = home / "Models" / "gguf"
            _write_gguf(extra / "a.gguf")
            _write_gguf(store / "b.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            binds: list[tuple[Path, Path]] = []

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                msg = publish(_target(home, extra=(extra,), model_root=home / "Models"))
            self.assertEqual(
                binds,
                [
                    (store.resolve(), dest / "chat" / "src0"),
                    (extra.resolve(), dest / "chat" / "src1"),
                ],
            )
            self.assertIn("2 trees", msg)

    def test_publish_snap_refuses_home_extra_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            real.mkdir()
            _write_gguf(real / "tiny.gguf")
            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=Path("/home/x/Models")),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    publish(_target(home))
            self.assertIn("/var/snap/lemonade-server/common", str(ctx.exception))

    def test_publish_without_lemonade_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("lemonade.detect", return_value=""):
                msg = publish(_target(home))
            self.assertEqual(msg, "lemonade not installed")

    def test_owned_where_is_dest_or_srcn(self) -> None:
        dest = Path("/var/snap/lemonade-server/common/ubuntuai-models")
        self.assertTrue(is_owned_lemonade_where(dest, dest))
        self.assertTrue(is_owned_lemonade_where(dest, dest / "embeddings"))
        self.assertTrue(is_owned_lemonade_where(dest, dest / "chat" / "src0"))
        self.assertTrue(is_owned_lemonade_where(dest, dest / "chat" / "src12"))
        self.assertFalse(is_owned_lemonade_where(dest, dest / "embeddings" / "nested"))
        self.assertFalse(is_owned_lemonade_where(dest, dest / "chat"))
        self.assertFalse(is_owned_lemonade_where(dest, dest / "chat" / "src"))
        self.assertFalse(is_owned_lemonade_where(dest, dest / "chat" / "custom"))
        self.assertFalse(is_owned_lemonade_where(dest, dest / "chat" / "src0" / "nested"))
        self.assertTrue(
            is_owned_lemonade_where(dest, dest / "chat" / "src0" / "embeddings")
        )
        self.assertFalse(
            is_owned_lemonade_where(
                dest, dest / "chat" / "src0" / "embeddings" / "nested"
            )
        )
        self.assertFalse(is_owned_lemonade_where(dest, dest / "chat" / "src0" / "gguf"))
        self.assertFalse(is_owned_lemonade_where(dest, Path("/mnt/other")))
        stage = STAGE_DIR.resolve()
        self.assertTrue(is_owned_lemonade_where(dest, stage))
        self.assertTrue(is_owned_lemonade_where(dest, stage / "src0"))
        self.assertTrue(is_owned_lemonade_where(dest, stage / "src0" / "embeddings"))
        self.assertFalse(is_owned_lemonade_where(dest, stage / "src0" / "gguf"))
        self.assertFalse(is_owned_lemonade_where(dest, stage / "other"))

    def test_leftover_nested_src_under_collapsed_dest(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            nested = extra / "Models" / "gguf"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src0 = dest / "chat" / "src0"
            src1 = dest / "chat" / "src1"
            _seed_unit(unit_dir, extra, src0, "src0.mount")
            _seed_unit(unit_dir, nested, src1, "src1.mount")
            plan = bind_mounts((extra, nested), dest)
            self.assertEqual(
                plan, (BindMount(extra.resolve(), dest / "chat" / "src0"),)
            )
            leftovers = leftover_owned_binds(
                dest, plan, unit_dir=unit_dir, mounted=(src0, src1)
            )
            self.assertEqual(leftovers, (src1.resolve(),))
            self.assertNotIn(dest.resolve(), leftovers)
            self.assertNotIn(src0.resolve(), leftovers)

    def test_leftover_not_in_plan_goes_away(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            store = Path(tmp) / "Models" / "gguf"
            other = Path(tmp) / "Downloads"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src0 = dest / "chat" / "src0"
            src1 = dest / "chat" / "src1"
            src2 = dest / "chat" / "src2"
            _seed_unit(unit_dir, extra, src0, "src0.mount")
            _seed_unit(unit_dir, store, src1, "src1.mount")
            _seed_unit(unit_dir, other, src2, "src2.mount")
            plan = (
                BindMount(extra, src0),
                BindMount(store, src1),
            )
            leftovers = leftover_owned_binds(dest, plan, unit_dir=unit_dir)
            self.assertEqual(leftovers, (src2.resolve(),))
            self.assertNotIn(src0.resolve(), leftovers)
            self.assertNotIn(src1.resolve(), leftovers)

    def test_leftover_unrelated_mount_is_untouched(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src1 = dest / "chat" / "src1"
            foreign = dest / "chat" / "src9"
            custom = dest / "chat" / "custom"
            other = Path("/mnt/other")
            _seed_unit(unit_dir, extra, src1, "src1.mount")
            _seed_unit(unit_dir, extra, foreign, "src9.mount", ours=False)
            _seed_unit(unit_dir, extra, other, "other.mount")
            plan = (BindMount(extra, dest),)
            mounted = (src1, foreign, custom, other)
            leftovers = leftover_owned_binds(
                dest, plan, unit_dir=unit_dir, mounted=mounted
            )
            self.assertEqual(leftovers, (src1.resolve(),))
            owned = owned_bind_wheres(dest, unit_dir, mounted)
            self.assertNotIn(other.resolve(), owned)
            self.assertNotIn(foreign.resolve(), owned)
            self.assertNotIn(custom.resolve(), owned)

    def test_leftover_fallback_mount_without_unit(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            src1 = dest / "chat" / "src1"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            plan = (BindMount(Path(tmp) / "AI models", dest),)
            leftovers = leftover_owned_binds(
                dest, plan, unit_dir=unit_dir, mounted=(src1,)
            )
            self.assertEqual(leftovers, (src1.resolve(),))

    def test_leftover_dest_when_plan_uses_srcn(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            store = Path(tmp) / "Models" / "gguf"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            _seed_unit(unit_dir, extra, dest, "dest.mount")
            plan = bind_mounts((extra, store), dest)
            leftovers = leftover_owned_binds(dest, plan, unit_dir=unit_dir)
            self.assertEqual(leftovers, (dest.resolve(),))
            self.assertTrue(leftovers[0].parts[-1] != "src0")

    def test_leftover_innermost_first(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            extra = Path(tmp) / "AI models"
            store = Path(tmp) / "Models" / "gguf"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src0 = dest / "chat" / "src0"
            _seed_unit(unit_dir, extra, dest, "dest.mount")
            _seed_unit(unit_dir, store, src0, "src0.mount")
            leftovers = leftover_owned_binds(dest, (), unit_dir=unit_dir)
            self.assertEqual(leftovers, (src0.resolve(), dest.resolve()))

    def test_publish_snap_drops_nested_leftover_srcn(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            _write_gguf(extra / "outer.gguf")
            store = extra / "Models"
            _write_gguf(store / "gguf" / "inner.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src0 = dest / "chat" / "src0"
            src1 = dest / "chat" / "src1"
            _seed_unit(unit_dir, extra, src0, "src0.mount")
            _seed_unit(unit_dir, store / "gguf", src1, "src1.mount")
            _seed_unit(unit_dir, extra, Path("/mnt/other"), "other.mount")
            _seed_unit(unit_dir, extra, dest / "chat" / "src9", "src9.mount", ours=False)
            dropped: list[Path] = []
            binds: list[tuple[Path, Path]] = []

            def record_drop(where: Path, dest_arg: Path) -> None:
                dropped.append(where)

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch(
                    "lemonade._mounted_wheres",
                    return_value=(
                        src0,
                        src1,
                        dest / "chat" / "src9",
                        Path("/mnt/other"),
                        dest / "chat" / "custom",
                    ),
                ),
                patch("lemonade._drop_bind_unit", side_effect=record_drop),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(extra,), model_root=store))
            self.assertEqual(binds, [(extra.resolve(), dest / "chat" / "src0")])
            self.assertEqual(
                [path.resolve() for path in dropped],
                [src1.resolve()],
            )
            self.assertNotIn(dest.resolve(), {path.resolve() for path in dropped})
            self.assertNotIn(Path("/mnt/other"), dropped)
            self.assertFalse(any(path.name == "src9" for path in dropped))
            self.assertFalse(any(path.name == "custom" for path in dropped))

    def test_publish_snap_keeps_sibling_srcn_drops_extra(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            store = home / "Models" / "gguf"
            _write_gguf(extra / "a.gguf")
            _write_gguf(store / "b.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src0 = dest / "chat" / "src0"
            src1 = dest / "chat" / "src1"
            src2 = dest / "chat" / "src2"
            _seed_unit(unit_dir, store, src0, "src0.mount")
            _seed_unit(unit_dir, extra, src1, "src1.mount")
            _seed_unit(unit_dir, home / "Downloads", src2, "src2.mount")
            dropped: list[Path] = []

            def record_drop(where: Path, dest_arg: Path) -> None:
                dropped.append(where)

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=(src0, src1, src2)),
                patch("lemonade._drop_bind_unit", side_effect=record_drop),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(extra,), model_root=home / "Models"))
            self.assertEqual([path.resolve() for path in dropped], [src2.resolve()])
            self.assertNotIn(src0.resolve(), {path.resolve() for path in dropped})
            self.assertNotIn(src1.resolve(), {path.resolve() for path in dropped})

    def test_publish_snap_drops_obsolete_dest_when_siblings(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            store = home / "Models" / "gguf"
            _write_gguf(extra / "a.gguf")
            _write_gguf(store / "b.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            _seed_unit(unit_dir, extra, dest, "dest.mount")
            dropped: list[Path] = []
            binds: list[tuple[Path, Path]] = []

            def record_drop(where: Path, dest_arg: Path) -> None:
                dropped.append(where)

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=(dest,)),
                patch("lemonade._drop_bind_unit", side_effect=record_drop),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(extra,), model_root=home / "Models"))
            self.assertEqual([path.resolve() for path in dropped], [dest.resolve()])
            self.assertEqual(
                binds,
                [
                    (store.resolve(), dest / "chat" / "src0"),
                    (extra.resolve(), dest / "chat" / "src1"),
                ],
            )

    def test_drop_bind_unit_disables_ours_and_unmounts(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            src1 = dest / "chat" / "src1"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            unit = _seed_unit(unit_dir, Path(tmp) / "AI models", src1, "src1.mount")
            cmds: list[list[str]] = []

            def record_run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(cmd)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            backup = Path(tmp) / "backups"
            original = unit.read_text(encoding="utf-8")
            with (
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade.UNIT_BACKUP_DIR", backup),
                patch("lemonade._ledger_dir_is_trusted", return_value=True),
                patch("lemonade._run", side_effect=record_run),
            ):
                _drop_bind_unit(src1, dest)
            self.assertFalse(unit.exists())
            saved = list(backup.glob("src1.mount.*"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_text(encoding="utf-8"), original)
            self.assertIn(["systemctl", "disable", "--now", "src1.mount"], cmds)
            self.assertIn(["umount", str(src1)], cmds)

    def test_drop_bind_unit_skips_unrelated(self) -> None:
        cmds: list[list[str]] = []
        with patch("lemonade._run", side_effect=lambda cmd: cmds.append(cmd)):
            _drop_bind_unit(Path("/mnt/other"), Path("/tmp/ubuntuai-models"))
        self.assertEqual(cmds, [])

    def test_publish_drops_legacy_dest_root_for_srcn(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            _seed_unit(unit_dir, real, dest, "dest.mount")
            dropped: list[Path] = []
            binds: list[tuple[Path, Path]] = []

            def record_drop(where: Path, dest_arg: Path) -> None:
                dropped.append(where)

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=(dest,)),
                patch("lemonade._drop_bind_unit", side_effect=record_drop),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home))
            self.assertEqual(binds, [(real.resolve(), dest / "chat" / "src0")])
            self.assertEqual([path.resolve() for path in dropped], [dest.resolve()])
            self.assertTrue(is_owned_lemonade_where(dest, dest))

    def test_embeddings_bind_uses_category_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "gguf" / "store.gguf")
            _write_gguf(root / "Gemma4-26B-A4B-GGUF" / "gemma.gguf")
            _write_gguf(root / "embeddings" / "qwen3-embedding.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            target = _target(home, extra=(), model_root=root)
            self.assertEqual(
                embeddings_mount(target, dest),
                BindMount(
                    (root / "embeddings").resolve(),
                    dest / "embeddings",
                ),
            )
            text = mount_unit_text((root / "embeddings").resolve(), dest / "embeddings")
            self.assertNotIn("After=local-fs.target", text)
            self.assertIn("Options=bind,nofail", text)
            self.assertIn(
                f"RequiresMountsFor={quote_unit_path((root / 'embeddings').resolve())}",
                text,
            )
            self.assertIn(f"What={mount_unit_path((root / 'embeddings').resolve())}\n", text)
            self.assertIn(f"Where={mount_unit_path(dest / 'embeddings')}\n", text)
            self.assertNotIn('What="', text)
            self.assertNotIn('Where="', text)
            binds: list[tuple[Path, Path]] = []
            covers: list[tuple[Path, str]] = []
            order: list[str] = []

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                order.append(f"bind:{where.name}")
                return "unit.mount"

            def record_cover(where: Path, parent_unit: str) -> str:
                covers.append((where, parent_unit))
                order.append("cover")
                return "cover.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._write_cover_unit", side_effect=record_cover),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                msg = publish(target)
            stage = STAGE_DIR.resolve()
            self.assertEqual(
                binds,
                [
                    (stage, stage),
                    (root.resolve(), stage / "src0"),
                    ((root / "embeddings").resolve(), dest / "embeddings"),
                    (stage / "src0", dest / "chat" / "src0"),
                ],
            )
            self.assertIn("2 trees", msg)
            self.assertEqual(
                covers,
                [(stage / "src0" / "embeddings", "unit.mount")],
            )
            self.assertLess(order.index("bind:embeddings"), order.index("cover"))
            self.assertLess(order.index("cover"), order.index("bind:src0", order.index("cover")))
            self.assertTrue(is_owned_lemonade_where(dest, dest / "embeddings"))
            self.assertTrue(is_owned_lemonade_where(dest, stage / "src0" / "embeddings"))
            cover = embeddings_cover(root, bind_mounts(gguf_sources(target), dest))
            self.assertEqual(
                cover,
                BindMount(
                    Path("tmpfs"),
                    stage / "src0" / "embeddings",
                    "ro,nosuid,nodev,noexec,size=64k,mode=0555,nofail",
                ),
            )
            self.assertNotEqual(cover.where, dest / "chat" / "src0" / "embeddings")

    def test_embeddings_cover_unit_text_orders_after_parent(self) -> None:
        dest = SNAP_LEMONADE_MODELS
        where = STAGE_DIR / "src0" / "embeddings"
        parent = "var-lib-ubuntuai-stage-src0.mount"
        text = cover_unit_text(where, parent)
        self.assertIn("What=tmpfs\n", text)
        self.assertIn("Type=tmpfs", text)
        self.assertIn(
            "Options=ro,nosuid,nodev,noexec,size=64k,mode=0555,nofail", text
        )
        self.assertIn(f"After={parent}\n", text)
        self.assertNotIn("After=local-fs.target", text)
        self.assertNotIn("DefaultDependencies=no", text)
        self.assertIn("Before=snap.lemonade-server.daemon.service", text)
        self.assertIn(f"Description={OWNED_UNIT_DESC}", text)
        self.assertIn("WantedBy=multi-user.target", text)
        self.assertIn(f"RequiresMountsFor={quote_unit_path(where.parent)}", text)
        self.assertIn(f"Where={mount_unit_path(where)}\n", text)
        self.assertNotIn('What="', text)
        self.assertNotIn('Where="', text)
        self.assertTrue(is_owned_lemonade_where(dest, where))
        with TemporaryDirectory() as tmp:
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            local_dest = Path(tmp) / "ubuntuai-models"
            cover_where = local_dest / "chat" / "src0" / "embeddings"
            (unit_dir / "cover.mount").write_text(
                cover_unit_text(cover_where, parent), encoding="utf-8"
            )
            chat = BindMount(Path(tmp) / "AI models", local_dest / "chat" / "src0")
            bare = leftover_owned_binds(local_dest, (chat,), unit_dir=unit_dir)
            self.assertEqual(bare, (cover_where.resolve(),))
            kept = leftover_owned_binds(
                local_dest,
                (chat, BindMount(Path("tmpfs"), cover_where)),
                unit_dir=unit_dir,
            )
            self.assertEqual(kept, ())

    def test_embeddings_unit_is_before_the_stage_cover(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            other = home / "other"
            _write_gguf(root / "chat.gguf")
            _write_gguf(root / "embeddings" / "embed.gguf")
            _write_gguf(other / "side.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            target = _target(home, extra=(other,), model_root=root)
            plan = _snap_mount_plan(target, dest, gguf_sources(target))
            cover = next(mount for mount in plan if mount.what == Path("tmpfs"))
            emb = next(
                mount
                for mount in plan
                if mount.where.name == "embeddings" and mount.what != Path("tmpfs")
            )
            rbind = next(mount for mount in plan if mount.options.startswith("rbind"))
            stage_src = next(
                mount for mount in plan if mount.where == STAGE_DIR.resolve() / "src0"
            )
            direct = next(mount for mount in plan if mount.what == other.resolve())
            self.assertLess(plan.index(emb), plan.index(cover))
            self.assertLess(plan.index(cover), plan.index(rbind))
            self.assertEqual(emb.what, (root / "embeddings").resolve())
            self.assertEqual(emb.where, dest / "embeddings")
            self.assertEqual(cover.where, STAGE_DIR.resolve() / "src0" / "embeddings")
            self.assertEqual(direct.where, dest / "chat" / "src1")
            self.assertEqual(direct.options, "bind,nofail")
            self.assertIn("bind,rprivate", stage_src.options)
            self.assertEqual(
                propagation_command(stage_src),
                ("mount", "-o", "bind,rprivate", str(stage_src.what), str(stage_src.where)),
            )
            self.assertEqual(propagation_command(rbind)[:2], ("mount", "--rbind"))
            self.assertEqual(propagation_command(emb)[:2], ("mount", "--bind"))
            self.assertEqual(propagation_command(cover)[1:3], ("-t", "tmpfs"))
            import lemonade

            names = lemonade._set_unit_order(plan)
            emb_text = lemonade._render_unit(emb)
            cover_unit = names[cover.where.resolve()]
            self.assertIn(f"Before={cover_unit}\n", emb_text)
            self.assertNotIn(f"After={cover_unit}", emb_text)
            self.assertNotIn("After=", emb_text)
            rbind_text = lemonade._render_unit(rbind)
            self.assertIn("Options=rbind,nofail", rbind_text)
            self.assertIn(f"After={cover_unit}\n", rbind_text)
            self.assertIn("Options=bind,rprivate,nofail", lemonade._render_unit(stage_src))

    def test_publish_plan_lists_units_without_writes(self) -> None:
        import lemonade

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "chat.gguf")
            _write_gguf(root / "embeddings" / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            target = _target(home, extra=(), model_root=root)
            original = lemonade._run
            tools: list[str] = []

            def spy(cmd: list[str]) -> SimpleNamespace:
                tool = Path(str(cmd[0])).name
                tools.append(tool)
                if tool in {"systemctl", "mount", "umount", "pkexec"}:
                    raise AssertionError(tool)
                return original(cmd)

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._run", side_effect=spy),
                patch("lemonade._write_bind_unit", side_effect=AssertionError("write")),
                patch("lemonade._write_cover_unit", side_effect=AssertionError("cover")),
                patch("lemonade._prepare_stage_dir", side_effect=AssertionError("stage")),
                patch(
                    "lemonade.load_tuning",
                    return_value={
                        "ctx_size": 2048,
                        "global_timeout": 2400,
                        "max_loaded_models": 1,
                        "llamacpp_backend": "vulkan",
                        "llamacpp_args": "--load-mode mmap",
                    },
                ),
            ):
                text = publish_plan(target)
            self.assertIn("Options=bind,rprivate,nofail", text)
            self.assertIn("Options=rbind,nofail", text)
            self.assertIn("What=tmpfs", text)
            self.assertIn(f"Where={STAGE_DIR.resolve() / 'src0' / 'embeddings'}", text)
            self.assertIn(f"What={(root / 'embeddings').resolve()}", text)
            self.assertIn("ctx_size=2048", text)
            self.assertIn("global_timeout=2400", text)
            self.assertIn("llamacpp_backend=vulkan", text)
            self.assertIn("llamacpp_args=--load-mode mmap", text)
            self.assertNotIn("systemctl", tools)
            self.assertNotIn("mount", tools)
            self.assertNotIn("umount", tools)

    def test_cli_dry_run_does_not_call_the_helper(self) -> None:
        from main import installer_main

        with (
            patch("lemonade.publish_plan", return_value="What=/models\nctx_size=2048\n") as plan,
            patch("apply.run_privileged", side_effect=AssertionError("helper")),
        ):
            rc = installer_main(["--publish-lemonade", "--dry-run"])
        self.assertEqual(rc, 0)
        plan.assert_called_once()

    def test_wait_tells_the_user_and_allows_a_slow_start(self) -> None:
        from io import StringIO
        from contextlib import redirect_stdout

        self.assertGreaterEqual(CONFIG_READY_TIMEOUT, 90.0)
        buf = StringIO()
        with (
            patch("lemonade._config_answers", return_value=True),
            redirect_stdout(buf),
        ):
            _wait_for_config()
        self.assertIn("Waiting for Lemonade to answer.", buf.getvalue())

    def test_publish_after_timeout_finishes_on_the_next_run(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            seen = {"n": 0}
            binds: list[Path] = []
            events: list[str] = []

            def answers() -> bool:
                seen["n"] += 1
                return seen["n"] > 1

            def bind(what: Path, where: Path) -> str:
                binds.append(where)
                return "unit.mount"

            def record_set(path: Path) -> None:
                events.append("set")

            def tune(target: UserTarget, hw: object = None) -> str:
                events.append("tune")
                return "lemonade load settings updated"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", side_effect=bind),
                patch("lemonade._mounted_wheres", return_value=()),
                patch("lemonade._daemon_was_running", return_value=False),
                patch("lemonade._stop_daemon"),
                patch("lemonade._start_daemon"),
                patch("lemonade._config_answers", side_effect=answers),
                patch("lemonade.CONFIG_READY_TIMEOUT", 0),
                patch("lemonade._set_extra_models_dir", side_effect=record_set),
                patch("lemonade.report_load_tuning", side_effect=tune),
            ):
                with self.assertRaises(RuntimeError):
                    publish(_target(home))
                msg = publish(_target(home))
            self.assertEqual(binds, [dest / "chat" / "src0", dest / "chat" / "src0"])
            self.assertEqual(events, ["set", "tune"])
            self.assertIn("load settings updated", msg)

    def test_publish_removes_a_tmpfs_hiding_embeddings(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "models"
            emb = root / "embeddings"
            _write_gguf(root / "chat.gguf")
            _write_gguf(emb / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            state = {
                "text": (
                    f"12 1 0:46 / {emb} ro,nosuid,nodev,noexec - "
                    "tmpfs tmpfs ro,size=64k,mode=555\n"
                )
            }
            cmds: list[list[str]] = []

            def read() -> str:
                return state["text"]

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd and cmd[0] == "umount":
                    state["text"] = ""
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if cmd and Path(str(cmd[0])).name == "systemd-escape":
                    return SimpleNamespace(returncode=0, stdout="unit.mount\n", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._write_cover_unit", return_value="cover.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                msg = publish(_target(home, extra=(), model_root=root))
            self.assertIn("Removed a temporary filesystem", msg)
            self.assertIn(str(emb), msg)
            self.assertIn(["umount", str(emb)], cmds)

    def test_publish_stops_when_the_tmpfs_stays(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "models"
            emb = root / "embeddings"
            _write_gguf(root / "chat.gguf")
            _write_gguf(emb / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            text = (
                f"12 1 0:46 / {emb} ro,nosuid,nodev,noexec - "
                "tmpfs tmpfs ro,size=64k,mode=555\n"
            )

            def run(cmd: list[str]) -> SimpleNamespace:
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", return_value=text),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", side_effect=AssertionError("write")),
                patch("lemonade._set_extra_models_dir", side_effect=AssertionError("set")),
                _quiet_daemon(),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    publish(_target(home, extra=(), model_root=root))
            message = str(ctx.exception)
            self.assertIn("still hiding", message)
            self.assertIn(str(emb), message)
            self.assertIn("model files stay where they are", message)
            self.assertNotIn("Traceback", message)

    def test_publish_leaves_a_real_mount_on_embeddings(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "models"
            emb = root / "embeddings"
            _write_gguf(root / "chat.gguf")
            _write_gguf(emb / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            text = f"12 1 8:1 / {emb} rw - ext4 /dev/sda1 rw\n"
            cmds: list[list[str]] = []

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="src0.mount\n", stderr="")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", return_value=text),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._write_cover_unit", return_value="cover.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(), model_root=root))
            self.assertFalse(any(cmd and cmd[0] == "umount" for cmd in cmds))

    def test_unchanged_unit_is_not_backed_up_again(self) -> None:
        import lemonade

        with TemporaryDirectory() as tmp:
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            backup = Path(tmp) / "backups"
            where = Path(tmp) / "dest" / "chat" / "src0"
            what = Path(tmp) / "models"
            unit = unit_dir / "src0.mount"
            unit.write_text(mount_unit_text(what, where), encoding="utf-8")

            def run(cmd: list[str]) -> SimpleNamespace:
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade.UNIT_BACKUP_DIR", backup),
                patch("lemonade._ledger_dir_is_trusted", return_value=True),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._escape_mount", return_value="src0.mount"),
                patch("lemonade._mounted_wheres", return_value=()),
            ):
                lemonade._write_bind_unit(what, where)
                self.assertEqual(list(backup.glob("*")), [])
                unit.write_text(unit.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
                lemonade._write_bind_unit(what, where)
            saved = list(backup.glob("src0.mount.*"))
            self.assertEqual(len(saved), 1)
            self.assertIn(OWNED_UNIT_DESC, saved[0].read_text(encoding="utf-8"))

    def test_failed_backup_does_not_delete_the_unit(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            src1 = dest / "chat" / "src1"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            unit = _seed_unit(unit_dir, Path(tmp) / "models", src1, "src1.mount")

            def run(cmd: list[str]) -> SimpleNamespace:
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._mounted_wheres", return_value=()),
                patch(
                    "lemonade._ensure_ledger_dir",
                    side_effect=RuntimeError(
                        "The Ubuntu AI state folder is not safe to use."
                    ),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _drop_bind_unit(src1, dest)
            self.assertIn("not safe", str(ctx.exception))
            self.assertTrue(unit.is_file())

    def test_wants_cleanup_keeps_foreign_links(self) -> None:
        import lemonade

        with TemporaryDirectory() as tmp:
            unit_dir = Path(tmp) / "units"
            wants = unit_dir / "snap.lemonade-server.daemon.service.wants"
            wants.mkdir(parents=True)
            stale_where = Path(tmp) / "dest" / "chat" / "src9"
            current_where = Path(tmp) / "dest" / "chat" / "src0"
            owned = unit_dir / "old.mount"
            current = unit_dir / "current.mount"
            foreign = unit_dir / "user.mount"
            owned.write_text(mount_unit_text(Path(tmp) / "old", stale_where), encoding="utf-8")
            current.write_text(
                mount_unit_text(Path(tmp) / "keep", current_where), encoding="utf-8"
            )
            foreign.write_text(
                mount_unit_text(Path(tmp) / "user", Path(tmp) / "user-where").replace(
                    f"Description={OWNED_UNIT_DESC}", "Description=User bind"
                ),
                encoding="utf-8",
            )
            (wants / "old.mount").symlink_to("../old.mount")
            (wants / "current.mount").symlink_to("../current.mount")
            (wants / "user.mount").symlink_to("../user.mount")
            (wants / "missing.mount").symlink_to("../missing.mount")
            lemonade._action_notes.clear()
            with patch("lemonade.SYSTEM_UNIT_DIR", unit_dir):
                lemonade._clean_daemon_wants((BindMount(Path(tmp) / "keep", current_where),))
            self.assertFalse((wants / "old.mount").exists())
            self.assertTrue((wants / "current.mount").is_symlink())
            self.assertTrue((wants / "user.mount").is_symlink())
            self.assertTrue((wants / "missing.mount").is_symlink())
            self.assertTrue(foreign.is_file())
            self.assertTrue(owned.is_file())
            notes = "\n".join(lemonade._action_notes)
            self.assertIn("Removed an old Lemonade mount link old.mount.", notes)
            self.assertIn("does not own", notes)
            self.assertIn("user.mount", notes)
            self.assertIn("missing.mount", notes)
            self.assertIn("unit file is missing", notes)

    def test_model_root_is_src0_then_lifted_trees_sorted(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "local.gguf")
            zeta = home / "zeta"
            mu = home / "mu"
            _write_gguf(zeta / "z.gguf")
            _write_gguf(mu / "m.gguf")
            (root / "to-zeta.gguf").symlink_to(zeta / "z.gguf")
            (root / "to-mu.gguf").symlink_to(mu / "m.gguf")
            extra = home / "extra"
            _write_gguf(extra / "e.gguf")
            target = _target(home, extra=(extra,), model_root=root)
            first = gguf_sources(target)
            second = gguf_sources(target)
            self.assertEqual(first, second)
            self.assertEqual(first[0], root.resolve())
            rest = (
                extra.resolve(),
                mu.resolve(),
                zeta.resolve(),
            )
            self.assertEqual(first[1:], tuple(sorted(rest, key=str)))

    def test_rewrites_unit_when_what_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            src0 = dest / "chat" / "src0"
            old = home / "Models" / "lemonade" / "chat"
            _seed_unit(unit_dir, old, src0, "src0.mount")
            mounted = {src0.resolve()}
            events: list[str] = []
            binds: list[tuple[Path, Path]] = []

            def wheres() -> tuple[Path, ...]:
                return tuple(mounted)

            def record_run(cmd: list[str]) -> SimpleNamespace:
                if cmd[:3] == ["systemctl", "is-active", "--quiet"]:
                    events.append("active")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if cmd[:3] == ["systemctl", "stop", "snap.lemonade-server.daemon"]:
                    events.append("stop-daemon")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if cmd[:2] == ["systemctl", "stop"]:
                    events.append("stop-unit")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if cmd[:2] == ["systemctl", "start"]:
                    events.append("start-daemon")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if cmd and cmd[0] == "umount":
                    events.append("umount")
                    mounted.discard(Path(cmd[1]).resolve())
                    return SimpleNamespace(returncode=1, stdout="", stderr="busy")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def record_bind(what: Path, where: Path) -> str:
                events.append("rewrite")
                binds.append((what, where))
                _seed_unit(unit_dir, what, where, "src0.mount")
                return "src0.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", side_effect=wheres),
                patch("lemonade._run", side_effect=record_run),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._wait_for_config"),
                patch("lemonade._set_extra_models_dir"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(), model_root=root))
            self.assertEqual(binds, [(root.resolve(), src0)])
            self.assertLess(events.index("stop-daemon"), events.index("umount"))
            self.assertLess(events.index("stop-unit"), events.index("umount"))
            self.assertLess(events.index("umount"), events.index("rewrite"))
            self.assertLess(events.index("rewrite"), events.index("start-daemon"))
            text = (unit_dir / "src0.mount").read_text(encoding="utf-8")
            self.assertIn(f"What={mount_unit_path(root.resolve())}\n", text)
            self.assertNotIn("lemonade/chat", text)

    def test_publish_aborts_when_unmount_stays_busy(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            _seed_unit(unit_dir, real, dest, "dest.mount")
            cmds: list[list[str]] = []

            def record_run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd[:3] == ["systemctl", "is-active", "--quiet"]:
                    return SimpleNamespace(returncode=0, stdout="active", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def fail_write(*args: object) -> str:
                raise AssertionError(args)

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=(dest,)),
                patch("lemonade._run", side_effect=record_run),
                patch("lemonade._write_bind_unit", side_effect=fail_write),
                patch("lemonade._write_cover_unit", side_effect=fail_write),
                patch(
                    "lemonade._set_extra_models_dir",
                    side_effect=AssertionError("set"),
                ),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    publish(_target(home))
            message = str(ctx.exception)
            self.assertIn("still mounted", message)
            self.assertIn(str(dest), message)
            self.assertIn("model files stay where they are", message)
            self.assertFalse((dest / "chat").exists())
            self.assertFalse((dest / "embeddings").exists())
            flat = [" ".join(cmd) for cmd in cmds]
            stop_at = flat.index("systemctl stop snap.lemonade-server.daemon")
            umount_at = flat.index(f"umount {dest}")
            start_at = flat.index("systemctl start snap.lemonade-server.daemon")
            self.assertLess(stop_at, umount_at)
            self.assertLess(umount_at, start_at)

    def test_publish_sets_and_tunes_while_daemon_is_up(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            events: list[str] = []
            up = {"value": True}

            def was_running() -> bool:
                return True

            def stop() -> None:
                events.append("stop")
                self.assertTrue(up["value"])
                up["value"] = False

            def start() -> None:
                events.append("start")
                self.assertFalse(up["value"])
                up["value"] = True

            def bind(what: Path, where: Path) -> str:
                events.append("bind")
                self.assertFalse(up["value"])
                return "unit.mount"

            def wait() -> None:
                events.append("wait")
                self.assertTrue(up["value"])

            def record_set(path: Path) -> None:
                events.append("set")
                self.assertTrue(up["value"])

            def tune(target: UserTarget, hw: object = None) -> str:
                events.append("tune")
                self.assertTrue(up["value"])
                return "lemonade load settings kept"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=()),
                patch("lemonade._daemon_was_running", side_effect=was_running),
                patch("lemonade._stop_daemon", side_effect=stop),
                patch("lemonade._start_daemon", side_effect=start),
                patch("lemonade._wait_for_config", side_effect=wait),
                patch("lemonade._write_bind_unit", side_effect=bind),
                patch("lemonade._set_extra_models_dir", side_effect=record_set),
                patch("lemonade.report_load_tuning", side_effect=tune),
            ):
                publish(_target(home))
            self.assertEqual(
                events, ["stop", "bind", "start", "wait", "set", "tune"]
            )
            self.assertTrue(up["value"])

    def test_publish_stops_daemon_again_when_it_was_not_running(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            events: list[str] = []
            up = {"value": False}

            def stop() -> None:
                events.append("stop")
                self.assertTrue(up["value"])
                up["value"] = False

            def start() -> None:
                events.append("start")
                self.assertFalse(up["value"])
                up["value"] = True

            def wait() -> None:
                events.append("wait")
                self.assertTrue(up["value"])

            def record_set(path: Path) -> None:
                events.append("set")
                self.assertTrue(up["value"])

            def tune(target: UserTarget, hw: object = None) -> str:
                events.append("tune")
                self.assertTrue(up["value"])
                return "lemonade load settings kept"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=()),
                patch("lemonade._daemon_was_running", return_value=False),
                patch("lemonade._stop_daemon", side_effect=stop),
                patch("lemonade._start_daemon", side_effect=start),
                patch("lemonade._wait_for_config", side_effect=wait),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._set_extra_models_dir", side_effect=record_set),
                patch("lemonade.report_load_tuning", side_effect=tune),
            ):
                publish(_target(home))
            self.assertEqual(events, ["start", "wait", "set", "tune", "stop"])
            self.assertFalse(up["value"])

    def test_publish_times_out_when_lemonade_does_not_answer(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            events: list[str] = []
            up = {"value": False}

            def stop() -> None:
                events.append("stop")
                self.assertTrue(up["value"])
                up["value"] = False

            def start() -> None:
                events.append("start")
                self.assertFalse(up["value"])
                up["value"] = True

            def fail_set(path: Path) -> None:
                events.append("set")
                raise AssertionError("set before lemonade answered")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=()),
                patch("lemonade._daemon_was_running", return_value=False),
                patch("lemonade._stop_daemon", side_effect=stop),
                patch("lemonade._start_daemon", side_effect=start),
                patch("lemonade._config_answers", return_value=False),
                patch("lemonade.CONFIG_READY_TIMEOUT", 0),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._set_extra_models_dir", side_effect=fail_set),
                patch(
                    "lemonade.report_load_tuning",
                    side_effect=AssertionError("tune"),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    publish(_target(home))
            message = str(ctx.exception)
            self.assertIn("in place", message)
            self.assertIn("models folder was not set", message)
            self.assertIn("Run publish again", message)
            self.assertNotIn("Traceback", message)
            self.assertEqual(events, ["start", "stop"])
            self.assertFalse(up["value"])

    def test_unreadable_ledger_does_not_stop_publish(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            _write_gguf(real / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"

            def urlopen(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("tuning ran after an unreadable ledger")

            with (
                _quiet_daemon(),
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._set_extra_models_dir"),
                patch(
                    "lemonade._load_ledger",
                    side_effect=RuntimeError("The Lemonade settings folder is not safe to use."),
                ),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.read_config", return_value=_merged_factory()),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.largest_gguf_bytes", return_value=105 * 1024**3),
            ):
                msg = publish(_target(home))
            self.assertIn(str(dest), msg)
            self.assertIn("could not be read", msg)
            self.assertIn("still published", msg)
            self.assertNotIn("Traceback", msg)
            self.assertNotIn("did not accept", msg)

    def test_symlink_embeddings_skips_tmpfs_cover(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "chat.gguf")
            real = home / "real-embeddings"
            _write_gguf(real / "embed.gguf")
            (root / "embeddings").symlink_to(real, target_is_directory=True)
            dest = Path(tmp) / "ubuntuai-models"
            target = _target(home, extra=(), model_root=root)
            self.assertIsNone(
                embeddings_cover(root, bind_mounts(gguf_sources(target), dest))
            )
            covers: list[Path] = []

            def record_cover(where: Path, parent_unit: str) -> str:
                covers.append(where)
                return "cover.mount"

            with (
                _quiet_daemon(),
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._write_cover_unit", side_effect=record_cover),
                patch("lemonade._set_extra_models_dir"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(target)
            self.assertEqual(covers, [])
            self.assertTrue((real / "embed.gguf").is_file())

    def test_busy_unmount_keeps_the_unit_file(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            src1 = dest / "chat" / "src1"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            unit = _seed_unit(unit_dir, Path(tmp) / "AI models", src1, "src1.mount")
            cmds: list[list[str]] = []

            def record_run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._run", side_effect=record_run),
                patch("lemonade._mounted_wheres", return_value=(src1,)),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _drop_bind_unit(src1, dest)
            self.assertIn("still mounted", str(ctx.exception))
            self.assertTrue(unit.is_file())
            self.assertIn(["systemctl", "disable", "--now", "src1.mount"], cmds)
            self.assertIn(["umount", str(src1)], cmds)
            self.assertNotIn(["systemctl", "daemon-reload"], cmds)

    def test_missing_systemctl_is_plain_english(self) -> None:
        with patch(
            "lemonade.subprocess.run",
            side_effect=FileNotFoundError(2, "No such file", "systemctl"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _daemon_was_running()
        text = str(ctx.exception)
        self.assertIn("systemctl is not installed", text)
        self.assertIn("model files stay where they are", text)
        self.assertNotIn("Errno", text)
        self.assertNotIn("Traceback", text)

    def test_missing_mount_and_umount_are_plain_english(self) -> None:
        for tool in ("mount", "umount"):
            with self.subTest(tool=tool):
                with patch(
                    "lemonade.subprocess.run",
                    side_effect=FileNotFoundError(2, "No such file", tool),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        _run([tool, "/tmp/nowhere"])
                text = str(ctx.exception)
                self.assertIn(f"{tool} is not installed", text)
                self.assertIn("model files stay where they are", text)
                self.assertNotIn("Errno", text)
                self.assertNotIn("Traceback", text)

    def test_live_lemonade_port_is_refused(self) -> None:
        req = urllib.request.Request("http://127.0.0.1:13305/internal/config")
        with self.assertRaises(AssertionError) as http_ctx:
            urllib.request.urlopen(req, timeout=1)
        self.assertIn("13305", str(http_ctx.exception))
        with self.assertRaises(AssertionError) as sock_ctx:
            socket.create_connection(("127.0.0.1", 13305), timeout=1)
        self.assertIn("13305", str(sock_ctx.exception))

    def test_bind_mkdir_refuses_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            victim = home / "victim"
            victim.mkdir()
            (victim / "keep").write_bytes(b"stay")
            dest = home / "dest"
            dest.mkdir()
            (dest / "chat").symlink_to(victim, target_is_directory=True)
            with self.assertRaises(RuntimeError) as ctx:
                _mkdir_nofollow(dest / "chat" / "src0")
            self.assertIn("symlink", str(ctx.exception))
            self.assertFalse((victim / "src0").exists())
            self.assertEqual((victim / "keep").read_bytes(), b"stay")

    def test_host_systemctl_is_refused(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            _run(["systemctl", "is-active", "--quiet", "snap.lemonade-server.daemon"])
        self.assertIn("systemctl", str(ctx.exception))

    def test_missing_embeddings_adds_no_bind(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "tiny.gguf")
            (root / "embeddings").mkdir()
            dest = Path(tmp) / "ubuntuai-models"
            target = _target(home, extra=(), model_root=root)
            self.assertIsNone(embeddings_mount(target, dest))
            binds: list[tuple[Path, Path]] = []

            def record_bind(what: Path, where: Path) -> str:
                binds.append((what, where))
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", side_effect=record_bind),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(target)
            self.assertEqual(binds, [(root.resolve(), dest / "chat" / "src0")])

    def test_foreign_embeddings_are_not_bound(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "tiny.gguf")
            emb = root / "embeddings"
            _write_gguf(emb / "qwen3-embedding.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            target = _target(home, extra=(), model_root=root)
            with patch("weights._FOREIGN_PREFIXES", (emb.resolve(),)):
                self.assertIsNone(embeddings_mount(target, dest))

    def test_obsolete_embeddings_bind_is_dropped(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "tiny.gguf")
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            emb_where = dest / "embeddings"
            _seed_unit(unit_dir, root / "embeddings", emb_where, "emb.mount")
            dropped: list[Path] = []

            def record_drop(where: Path, dest_arg: Path) -> None:
                dropped.append(where)

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._mounted_wheres", return_value=(emb_where,)),
                patch("lemonade._drop_bind_unit", side_effect=record_drop),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(), model_root=root))
            self.assertEqual([path.resolve() for path in dropped], [emb_where.resolve()])

    def test_apply_publish_verb(self) -> None:
        self.assertEqual(APPLY_PUBLISH_VERB, "lemonade-publish")
        action = Action("lemonade", "publish GGUF files to Lemonade", ("owner",))
        self.assertEqual(action.kind, "lemonade")

    def test_systemd_path_escape_matches_known_names(self) -> None:
        import lemonade

        cases = {
            Path("/var/lib/ubuntuai/stage"): "var-lib-ubuntuai-stage.mount",
            Path("/tmp/AI models"): "tmp-AI\\x20models.mount",
            Path("/tmp/100% models"): "tmp-100\\x25\\x20models.mount",
            Path("/"): "-.mount",
            Path("/tmp/foo-bar_baz.txt"): "tmp-foo\\x2dbar_baz.txt.mount",
            Path("/tmp//foo/"): "tmp-foo.mount",
            Path("/tmp/./foo"): "tmp-foo.mount",
        }
        for path, expected in cases.items():
            self.assertEqual(lemonade._escape_mount(path), expected)
        with self.assertRaises(RuntimeError) as ctx:
            lemonade._escape_mount(Path("/tmp/foo/../bar"))
        self.assertIn("systemd-escape failed", str(ctx.exception))

    def test_second_publish_keeps_the_carried_cover(self) -> None:
        import lemonade

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "chat.gguf")
            _write_gguf(root / "embeddings" / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            backup = Path(tmp) / "backups"
            stage = STAGE_DIR.resolve()
            dest_cover = dest / "chat" / "src0" / "embeddings"
            stage_cover = stage / "src0" / "embeddings"
            carried = "\n".join(
                (
                    f"10 1 0:1 / {stage} rw - ext4 /dev/sda1 rw",
                    f"11 10 0:1 / {stage / 'src0'} rw - ext4 /dev/sda1 rw",
                    (
                        f"12 11 0:46 / {stage_cover} ro,nosuid,nodev,noexec shared:5 - "
                        "tmpfs tmpfs ro,size=64k,mode=555"
                    ),
                    f"13 1 0:1 / {dest / 'chat' / 'src0'} rw - ext4 /dev/sda1 rw",
                    (
                        f"14 13 0:46 / {dest_cover} ro,nosuid,nodev,noexec master:5 - "
                        "tmpfs tmpfs ro,size=64k,mode=555"
                    ),
                )
            )
            phase = {"second": False}
            cmds: list[list[str]] = []
            dropped: list[Path] = []

            def read() -> str:
                return carried if phase["second"] else ""

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def record_drop(where: Path, dest_arg: Path) -> None:
                dropped.append(where)

            target = _target(home, extra=(), model_root=root)
            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade.UNIT_BACKUP_DIR", backup),
                patch("lemonade._ledger_dir_is_trusted", return_value=True),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._drop_bind_unit", side_effect=record_drop),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._write_cover_unit", return_value="cover.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(target)
                (unit_dir / "carried.mount").write_text(
                    cover_unit_text(dest_cover, "parent.mount"),
                    encoding="utf-8",
                )
                phase["second"] = True
                cmds.clear()
                dropped.clear()
                publish(target)
                self.assertTrue(
                    lemonade._is_carried_rbind_mount(
                        dest_cover,
                        _snap_mount_plan(target, dest, gguf_sources(target)),
                    )
                )
            self.assertEqual(dropped, [])
            self.assertIn(["systemctl", "disable", "carried.mount"], cmds)
            self.assertNotIn(["systemctl", "disable", "--now", "carried.mount"], cmds)
            umounted = [
                cmd for cmd in cmds if cmd and cmd[0] == "umount" and str(dest_cover) in cmd
            ]
            self.assertEqual(umounted, [])
            self.assertFalse((unit_dir / "carried.mount").exists())

    def test_publish_remounts_embeddings_captured_as_tmpfs(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "chat.gguf")
            _write_gguf(root / "embeddings" / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            emb_where = dest / "embeddings"
            _seed_unit(unit_dir, root / "embeddings", emb_where, "emb.mount")
            state = {
                "text": (
                    f"20 1 0:46 / {emb_where} ro,nosuid,nodev,noexec - "
                    "tmpfs tmpfs ro,size=64k,mode=555\n"
                )
            }
            cmds: list[list[str]] = []
            wrote: list[Path] = []

            def read() -> str:
                return state["text"]

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd[:1] == ["umount"] and len(cmd) > 1 and cmd[1] == str(emb_where):
                    state["text"] = ""
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def bind(what: Path, where: Path) -> str:
                wrote.append(where)
                return "unit.mount"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", side_effect=bind),
                patch("lemonade._write_cover_unit", return_value="cover.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(), model_root=root))
            self.assertIn(["systemctl", "stop", "emb.mount"], cmds)
            self.assertIn(["umount", str(emb_where)], cmds)
            self.assertLess(
                cmds.index(["umount", str(emb_where)]),
                len(cmds),
            )
            self.assertIn(emb_where, wrote)
            stop_at = cmds.index(["systemctl", "stop", "emb.mount"])
            umount_at = cmds.index(["umount", str(emb_where)])
            self.assertLess(stop_at, umount_at)

    def test_publish_remounts_embeddings_when_the_directory_differs(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "AI models"
            _write_gguf(root / "chat.gguf")
            _write_gguf(root / "embeddings" / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            emb_where = dest / "embeddings"
            emb_where.mkdir(parents=True)
            state = {"text": f"20 1 8:1 /other {emb_where} rw - ext4 /dev/sdb1 rw\n"}
            cmds: list[list[str]] = []

            def read() -> str:
                return state["text"]

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd[:1] == ["umount"] and len(cmd) > 1 and cmd[1] == str(emb_where):
                    state["text"] = ""
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._write_cover_unit", return_value="cover.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                publish(_target(home, extra=(), model_root=root))
            self.assertIn(["umount", str(emb_where)], cmds)

    def test_publish_leaves_a_user_tmpfs_on_embeddings(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "models"
            emb = root / "embeddings"
            kept = _write_gguf(emb / "embed.gguf")
            _write_gguf(root / "chat.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            text = f"12 1 0:46 / {emb} rw,relatime - tmpfs tmpfs rw,size=1024k\n"
            cmds: list[list[str]] = []

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", return_value=text),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", side_effect=AssertionError("write")),
                patch("lemonade._set_extra_models_dir", side_effect=AssertionError("set")),
                _quiet_daemon(),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    publish(_target(home, extra=(), model_root=root))
            message = str(ctx.exception)
            self.assertIn("not the empty cover", message)
            self.assertIn("stays as it is", message)
            self.assertNotIn("Traceback", message)
            self.assertFalse(any(cmd and cmd[0] == "umount" for cmd in cmds))
            self.assertEqual(kept.read_bytes(), b"G" * 2048)

    def test_publish_refuses_a_cover_shared_only_with_a_foreign_mount(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "models"
            emb = root / "embeddings"
            _write_gguf(root / "chat.gguf")
            _write_gguf(emb / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            foreign = Path(tmp) / "foreign"
            text = "\n".join(
                (
                    (
                        f"12 1 0:46 / {emb} ro,nosuid,nodev,noexec shared:9 - "
                        "tmpfs tmpfs ro,size=64k,mode=555"
                    ),
                    f"13 1 0:46 / {foreign} rw shared:9 - ext4 /dev/sdb1 rw",
                )
            )
            cmds: list[list[str]] = []

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", return_value=text),
                patch("lemonade._run", side_effect=run),
                patch("lemonade._write_bind_unit", side_effect=AssertionError("write")),
                _quiet_daemon(),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    publish(_target(home, extra=(), model_root=root))
            self.assertIn("not the empty cover", str(ctx.exception))
            self.assertFalse(any(cmd and cmd[0] == "umount" for cmd in cmds))

            owned = "\n".join(
                (
                    (
                        f"12 1 0:46 / {emb} ro,nosuid,nodev,noexec shared:9 - "
                        "tmpfs tmpfs ro,size=64k,mode=555"
                    ),
                    (
                        f"13 1 0:46 / {dest / 'chat' / 'src0' / 'embeddings'} "
                        "ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,size=64k,mode=555"
                    ),
                )
            )
            cleared = {"text": owned}

            def read() -> str:
                return cleared["text"]

            def clear(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd and cmd[0] == "umount":
                    cleared["text"] = ""
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            cmds.clear()
            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=clear),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._write_cover_unit", return_value="cover.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade.report_load_tuning", return_value=""),
            ):
                msg = publish(_target(home, extra=(), model_root=root))
            self.assertIn("Removed a temporary filesystem", msg)
            self.assertIn(["umount", str(emb)], cmds)

    def test_publish_creates_the_state_folder_before_unmount(self) -> None:
        _state_guard.stop()
        try:
            with TemporaryDirectory() as tmp:
                home = Path(tmp)
                real = home / "AI models"
                _write_gguf(real / "tiny.gguf")
                dest = Path(tmp) / "ubuntuai-models"
                unit_dir = Path(tmp) / "units"
                unit_dir.mkdir()
                _seed_unit(unit_dir, real, dest, "dest.mount")
                state = Path(tmp) / "state"
                stage = state / "stage"
                backup = state / "backups"
                seen = {"umount": False}

                def run(cmd: list[str]) -> SimpleNamespace:
                    if cmd and cmd[0] == "umount":
                        self.assertTrue(stage.is_dir())
                        self.assertTrue(backup.is_dir())
                        seen["umount"] = True
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                def wheres() -> tuple[Path, ...]:
                    if seen["umount"]:
                        return ()
                    return (dest,)

                with (
                    patch("lemonade.detect", return_value="snap"),
                    patch("lemonade.extra_dir", return_value=dest),
                    patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                    patch("lemonade.STAGE_DIR", stage),
                    patch("lemonade.UNIT_BACKUP_DIR", backup),
                    patch("lemonade._ledger_dir_is_trusted", return_value=True),
                    patch("lemonade._mounted_wheres", side_effect=wheres),
                    patch("lemonade._run", side_effect=run),
                    patch("lemonade._write_bind_unit", return_value="unit.mount"),
                    patch("lemonade._set_extra_models_dir"),
                    _quiet_daemon(),
                    patch("lemonade.report_load_tuning", return_value=""),
                ):
                    publish(_target(home))
                self.assertTrue(seen["umount"])
                self.assertTrue(stage.is_dir())
        finally:
            _state_guard.start()

    def test_publish_stops_before_unmount_when_the_state_folder_is_unsafe(self) -> None:
        _state_guard.stop()
        try:
            with TemporaryDirectory() as tmp:
                home = Path(tmp)
                root = home / "models"
                emb = root / "embeddings"
                _write_gguf(root / "chat.gguf")
                _write_gguf(emb / "embed.gguf")
                dest = Path(tmp) / "ubuntuai-models"
                unit_dir = Path(tmp) / "units"
                unit_dir.mkdir()
                unit = _seed_unit(unit_dir, root, dest, "dest.mount")
                state = Path(tmp) / "state"
                text = (
                    f"12 1 0:46 / {emb} ro,nosuid,nodev,noexec - "
                    "tmpfs tmpfs ro,size=64k,mode=555\n"
                )
                cmds: list[list[str]] = []

                def run(cmd: list[str]) -> SimpleNamespace:
                    cmds.append(list(cmd))
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                with (
                    patch("lemonade.detect", return_value="snap"),
                    patch("lemonade.extra_dir", return_value=dest),
                    patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                    patch("lemonade.STAGE_DIR", state / "stage"),
                    patch("lemonade.UNIT_BACKUP_DIR", state / "backups"),
                    patch("lemonade._ledger_dir_is_trusted", return_value=False),
                    patch("lemonade._read_mountinfo_text", return_value=text),
                    patch("lemonade._run", side_effect=run),
                    patch("lemonade._write_bind_unit", side_effect=AssertionError("write")),
                    _quiet_daemon(),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        publish(_target(home, extra=(), model_root=root))
                self.assertIn("not safe", str(ctx.exception))
                self.assertTrue(unit.is_file())
                self.assertFalse(any(cmd and cmd[0] == "umount" for cmd in cmds))
        finally:
            _state_guard.start()

    def test_publish_plan_mentions_the_embeddings_cover_leak(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "models"
            emb = root / "embeddings"
            _write_gguf(root / "chat.gguf")
            _write_gguf(emb / "embed.gguf")
            dest = Path(tmp) / "ubuntuai-models"
            cover = (
                f"12 1 0:46 / {emb} ro,nosuid,nodev,noexec - "
                "tmpfs tmpfs ro,size=64k,mode=555\n"
            )
            foreign = f"12 1 0:46 / {emb} rw,relatime - tmpfs tmpfs rw,size=1024k\n"
            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", return_value=cover),
                patch("lemonade.load_tuning", return_value={}),
            ):
                text = publish_plan(_target(home, extra=(), model_root=root))
            self.assertIn(f"Remove the empty cover that is hiding {emb}.", text)
            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._read_mountinfo_text", return_value=foreign),
                patch("lemonade.load_tuning", return_value={}),
            ):
                text = publish_plan(_target(home, extra=(), model_root=root))
            self.assertIn("not the empty cover", text)
            self.assertIn("stays as it is", text)
            self.assertNotIn("Remove the empty cover", text)

    def test_unit_backups_keep_the_latest_ten(self) -> None:
        import lemonade

        with TemporaryDirectory() as tmp:
            backup = Path(tmp) / "backups"
            backup.mkdir()
            unit = Path(tmp) / "src0.mount"
            unit.write_text(mount_unit_text(Path(tmp) / "models", Path(tmp) / "dest"), encoding="utf-8")
            for index in range(12):
                (backup / f"src0.mount.20200101T000000{index:06d}Z").write_text(
                    "old", encoding="utf-8"
                )
            (backup / "src0.mount.keep-link").symlink_to(unit.name)
            with (
                patch("lemonade.UNIT_BACKUP_DIR", backup),
                patch("lemonade._ledger_dir_is_trusted", return_value=True),
            ):
                lemonade._backup_unit_file(unit)
            kept = sorted(
                path.name
                for path in backup.iterdir()
                if path.is_file() and path.name.startswith("src0.mount.")
            )
            self.assertEqual(len(kept), 10)
            self.assertTrue(kept[-1].startswith("src0.mount.20"))
            self.assertGreater(kept[-1], "src0.mount.20200101T000000000011Z")
            self.assertTrue((backup / "src0.mount.keep-link").is_symlink())
            self.assertFalse((backup / "src0.mount.20200101T000000000000Z").exists())

    def test_prepare_stage_dir_closes_directory_fds(self) -> None:
        import lemonade

        def fd_count() -> int:
            return len(os.listdir("/proc/self/fd"))

        _state_guard.stop()
        try:
            before = fd_count()
            with TemporaryDirectory() as tmp:
                state = Path(tmp) / "state"
                with (
                    patch("lemonade.STAGE_DIR", state / "stage"),
                    patch("lemonade.UNIT_BACKUP_DIR", state / "backups"),
                    patch("lemonade._ledger_dir_is_trusted", return_value=True),
                ):
                    lemonade._prepare_stage_dir()
                    lemonade._prepare_stage_dir()
                    self.assertEqual(fd_count(), before)
        finally:
            _state_guard.start()

    def test_mount_fallback_replaces_a_wrong_mount_and_skips_a_current_one(self) -> None:
        import lemonade

        calls: list[list[str]] = []

        def run(cmd: list[str]) -> SimpleNamespace:
            calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch("lemonade._run", side_effect=run),
            patch("lemonade._mount_is_current", return_value=True),
        ):
            self.assertIsNone(
                lemonade._mount_with_command(Path("/src"), Path("/dst"), "bind,nofail")
            )
        self.assertEqual(calls, [])
        mounted = {"on": True}

        def current(what: Path, where: Path, options: str) -> bool:
            return False

        def is_mount(where: Path) -> bool:
            return mounted["on"] and where == Path("/dst")

        def replace(cmd: list[str]) -> SimpleNamespace:
            calls.append(list(cmd))
            if cmd and cmd[0] == "umount":
                mounted["on"] = False
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        calls.clear()
        with (
            patch("lemonade._run", side_effect=replace),
            patch("lemonade._mount_is_current", side_effect=current),
            patch("lemonade._is_mountpoint", side_effect=is_mount),
        ):
            lemonade._mount_with_command(Path("/src"), Path("/dst"), "bind,nofail")
        self.assertEqual(calls[0][:1], ["umount"])
        self.assertEqual(calls[1][:1], ["mount"])

    def test_busy_unmount_puts_the_embeddings_cover_back(self) -> None:
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ubuntuai-models"
            where = dest / "chat" / "src0"
            child = where / "embeddings"
            unit_dir = Path(tmp) / "units"
            unit_dir.mkdir()
            unit = _seed_unit(unit_dir, Path(tmp) / "AI models", where, "src0.mount")
            state = {"child": True}

            def read() -> str:
                lines = [f"30 1 0:1 / {where} rw - ext4 /dev/sda1 rw"]
                if state["child"]:
                    lines.append(
                        f"31 30 0:46 / {child} ro,nosuid,nodev,noexec - "
                        "tmpfs tmpfs ro,size=64k,mode=555"
                    )
                return "\n".join(lines) + "\n"

            cmds: list[list[str]] = []

            def run(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd[:1] == ["umount"] and len(cmd) > 1 and cmd[1] == str(child):
                    state["child"] = False
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if cmd[:1] == ["mount"]:
                    state["child"] = True
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=run),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _drop_bind_unit(where, dest)
            self.assertIn("still mounted", str(ctx.exception))
            self.assertNotIn("was removed", str(ctx.exception))
            self.assertTrue(unit.is_file())
            self.assertTrue(any(cmd[:1] == ["mount"] for cmd in cmds))
            self.assertTrue(state["child"])

            state["child"] = True
            cmds.clear()

            def fail_restore(cmd: list[str]) -> SimpleNamespace:
                cmds.append(list(cmd))
                if cmd[:1] == ["umount"] and len(cmd) > 1 and cmd[1] == str(child):
                    state["child"] = False
                if cmd[:1] == ["mount"]:
                    return SimpleNamespace(returncode=32, stdout="", stderr="busy")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("lemonade.SYSTEM_UNIT_DIR", unit_dir),
                patch("lemonade._read_mountinfo_text", side_effect=read),
                patch("lemonade._run", side_effect=fail_restore),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _drop_bind_unit(where, dest)
            message = str(ctx.exception)
            self.assertIn("was removed", message)
            self.assertIn("Run publish again", message)
            self.assertIn(str(child), message)


def _vulkan_igpu(ram_gib: int = 122) -> Hardware:
    return Hardware(
        cpu_name="strix",
        ram_bytes=ram_gib * 1024**3,
        devices=(
            Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
            Device("cpu", "cpu", "cpu", None, "cpu"),
        ),
    )


class LemonadeLoadTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._trust = patch("lemonade._ledger_dir_is_trusted", return_value=True)
        self._trust.start()
        self.addCleanup(self._trust.stop)

    def test_large_vulkan_igpu_lands_mmap(self) -> None:
        hw = _vulkan_igpu(122)
        huge = 105 * 1024**3
        tun = load_tuning(hw, huge)
        self.assertEqual(tun["llamacpp_args"], LLAMACPP_MMAP_ARGS)
        self.assertEqual(tun["llamacpp_backend"], "vulkan")
        self.assertEqual(tun["max_loaded_models"], 1)
        self.assertEqual(tun["ctx_size"], 2048)
        self.assertGreaterEqual(int(tun["global_timeout"]), 2400)

    def test_small_vulkan_igpu_skips_mmap(self) -> None:
        hw = _vulkan_igpu(122)
        tun = load_tuning(hw, 2 * 1024**3)
        self.assertEqual(tun["llamacpp_backend"], "vulkan")
        self.assertEqual(tun["ctx_size"], -1)
        self.assertNotIn("llamacpp_args", tun)

    def test_huge_shard_on_big_ram_still_mmaps(self) -> None:
        hw = _vulkan_igpu(512)
        huge = 105 * 1024**3
        tun = load_tuning(hw, huge)
        self.assertLess(huge / hw.ram_bytes, 0.35)
        self.assertEqual(tun["llamacpp_args"], LLAMACPP_MMAP_ARGS)
        self.assertEqual(tun["max_loaded_models"], 1)

    def test_navi_rocm_small_has_no_mmap(self) -> None:
        hw = Hardware(
            cpu_name="amd",
            ram_bytes=32 * 1024**3,
            devices=(
                Device("dgpu", "amd", "Radeon RX 7900 XT", "/dev/dri/renderD128", "rocm"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        tun = load_tuning(hw, 2 * 1024**3)
        self.assertEqual(tun["llamacpp_backend"], "rocm")
        self.assertNotIn("llamacpp_args", tun)

    def test_cli_parts_set_vulkan_args(self) -> None:
        parts = cli_tuning_parts(
            {
                "max_loaded_models": 1,
                "llamacpp_backend": "vulkan",
                "llamacpp_args": LLAMACPP_MMAP_ARGS,
            }
        )
        self.assertIn("llamacpp.backend=vulkan", parts)
        self.assertIn(f"llamacpp.args={LLAMACPP_MMAP_ARGS}", parts)
        self.assertIn(f"llamacpp.vulkan_args={LLAMACPP_MMAP_ARGS}", parts)

    def test_apply_tuning_keeps_user_values_and_still_warns(self) -> None:
        state = {
            "max_loaded_models": 2,
            "ctx_size": 16384,
            "global_timeout": 900,
            "llamacpp_backend": "vulkan",
            "llamacpp_args": "--threads 8",
        }
        reads: list[str] = []

        def read() -> dict:
            reads.append("read")
            return dict(state)

        def urlopen(req: object, timeout: int = 3) -> object:
            raise AssertionError(getattr(req, "data", b""))

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger(
                {
                    "max_loaded_models": 1,
                    "ctx_size": 4096,
                    "global_timeout": 2400,
                    "llamacpp_backend": "rocm",
                    "llamacpp_args": LLAMACPP_MMAP_ARGS,
                    "llamacpp_vulkan_args": LLAMACPP_MMAP_ARGS,
                },
                home,
            )
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.largest_gguf_bytes", return_value=105 * 1024**3),
                patch("lemonade.shutil.which", return_value=None),
            ):
                text = report_load_tuning(
                    _target(home), _vulkan_igpu(122), ledger_dir=home
                )
        self.assertEqual(reads, ["read"])
        self.assertIn("lemonade load settings kept", text)
        self.assertIn("Strong warning", text)
        self.assertIn("2 models", text)
        self.assertNotIn("keep one model", text)
        self.assertEqual(LOAD_RISK_POLICY, "warn_only")
        risk = load_risk(0.60, 105 * 1024**3, "vulkan")
        self.assertEqual(risk["action"], "warn")

    def test_apply_tuning_fills_only_unset_values(self) -> None:
        state: dict = {
            "max_loaded_models": 2,
            "llamacpp": {"backend": "rocm", "args": "--threads 4"},
        }
        order: list[str] = []
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            order.append("read")
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            order.append("write")
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger(
                {
                    "max_loaded_models": 1,
                    "llamacpp_backend": "vulkan",
                    "llamacpp_args": "",
                    "llamacpp_vulkan_args": "",
                },
                home,
            )
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
        self.assertEqual(order, ["read", "write", "read"])
        self.assertEqual(msg, "lemonade load settings updated")
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertNotIn("max_loaded_models", body)
        self.assertNotIn("llamacpp_args", body)
        self.assertNotIn("llamacpp_backend", body)
        self.assertEqual(body["ctx_size"], defaults["ctx_size"])
        self.assertEqual(body["global_timeout"], defaults["global_timeout"])
        self.assertEqual(state["max_loaded_models"], 2)
        self.assertEqual(state["llamacpp"]["args"], "--threads 4")

    def test_apply_tuning_sets_defaults_when_unset(self) -> None:
        state: dict = {
            "max_loaded_models": "  ",
            "ctx_size": "",
            "global_timeout": None,
        }
        order: list[str] = []
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            order.append("read")
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            order.append("write")
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
        self.assertEqual(order[0], "read")
        self.assertLess(order.index("read"), order.index("write"))
        self.assertEqual(msg, "lemonade load settings updated")
        body = bodies[0]
        self.assertEqual(body["max_loaded_models"], 1)
        self.assertEqual(body["ctx_size"], defaults["ctx_size"])
        self.assertEqual(body["global_timeout"], defaults["global_timeout"])
        self.assertEqual(body["llamacpp_backend"], defaults["llamacpp_backend"])
        self.assertEqual(body["llamacpp_args"], LLAMACPP_MMAP_ARGS)

    def test_string_max_loaded_models_is_kept(self) -> None:
        state = {
            "max_loaded_models": "2",
            "ctx_size": "16384",
            "global_timeout": "900",
            "llamacpp_backend": "vulkan",
            "llamacpp_vulkan_args": "--load-mode mmap",
        }

        def read() -> dict:
            return dict(state)

        def urlopen(req: object, timeout: int = 3) -> object:
            raise AssertionError(getattr(req, "data", b""))

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger(
                {
                    "max_loaded_models": 1,
                    "ctx_size": 2048,
                    "global_timeout": 2400,
                    "llamacpp_backend": "rocm",
                    "llamacpp_args": "",
                    "llamacpp_vulkan_args": "",
                },
                home,
            )
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.shutil.which", return_value=None),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
        self.assertEqual(msg, "lemonade load settings kept")

    def test_fresh_merged_defaults_get_our_tuning(self) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
        self.assertEqual(msg, "lemonade load settings updated")
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertEqual(body["ctx_size"], defaults["ctx_size"])
        self.assertEqual(body["global_timeout"], defaults["global_timeout"])
        self.assertNotIn("max_loaded_models", body)
        self.assertEqual(state["max_loaded_models"], 1)
        self.assertEqual(body["llamacpp_backend"], "vulkan")
        self.assertEqual(body["llamacpp_args"], LLAMACPP_MMAP_ARGS)

    def test_no_ledger_keeps_max_loaded_and_retunes_ctx(self) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        state["max_loaded_models"] = 2
        state["ctx_size"] = 4096
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.largest_gguf_bytes", return_value=105 * 1024**3),
            ):
                text = report_load_tuning(
                    _target(home), _vulkan_igpu(122), ledger_dir=home
                )
            ledger = read_tuning_ledger(home)
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertNotIn("max_loaded_models", body)
        self.assertEqual(body["ctx_size"], 2048)
        self.assertEqual(body["global_timeout"], 2400)
        self.assertEqual(body["llamacpp_backend"], "vulkan")
        self.assertEqual(body["llamacpp_args"], LLAMACPP_MMAP_ARGS)
        self.assertEqual(state["max_loaded_models"], 2)
        self.assertEqual(state["ctx_size"], 2048)
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(ledger["ctx_size"], 2048)
        self.assertNotIn("max_loaded_models", ledger)
        self.assertIn("2 models", text)
        self.assertNotIn("keep one model", text)

    def _drive(self, state: dict, defaults: dict) -> tuple[str, list[dict], dict | None]:
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
            ledger = read_tuning_ledger(home)
        return msg, bodies, ledger

    def test_no_ledger_keeps_ctx_262144(self) -> None:
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        for ctx in (262144, "262144"):
            with self.subTest(ctx=ctx):
                state = _merged_factory()
                state["ctx_size"] = ctx
                msg, bodies, _ledger = self._drive(state, defaults)
                self.assertIn("updated", msg)
                self.assertTrue(bodies)
                for body in bodies:
                    self.assertNotIn("ctx_size", body)
                self.assertEqual(state["ctx_size"], ctx)

    def test_no_ledger_retunes_ctx_4096(self) -> None:
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        for ctx in (4096, "4096", "  4096  "):
            with self.subTest(ctx=ctx):
                state = _merged_factory()
                state["ctx_size"] = ctx
                _msg, bodies, ledger = self._drive(state, defaults)
                self.assertEqual(len(bodies), 1)
                self.assertEqual(bodies[0]["ctx_size"], 2048)
                self.assertEqual(state["ctx_size"], 2048)
                self.assertIsNotNone(ledger)
                assert ledger is not None
                self.assertEqual(ledger["ctx_size"], 2048)

    def test_no_ledger_matching_vulkan_backend_is_not_posted(self) -> None:
        state = _merged_factory()
        state["llamacpp"]["backend"] = "vulkan"
        small = load_tuning(_vulkan_igpu(32), 1 * 1024**3)
        msg, bodies, ledger = self._drive(state, small)
        self.assertEqual(msg, "lemonade load settings kept")
        self.assertEqual(bodies, [])
        self.assertIsNone(ledger)
        self.assertEqual(state["llamacpp"]["backend"], "vulkan")

        state = _merged_factory()
        state["llamacpp"]["backend"] = "vulkan"
        state["ctx_size"] = 4096
        large = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        self.assertEqual(large["llamacpp_backend"], "vulkan")
        _msg, bodies, ledger = self._drive(state, large)
        self.assertEqual(len(bodies), 1)
        self.assertNotIn("llamacpp_backend", bodies[0])
        self.assertEqual(bodies[0]["ctx_size"], 2048)
        self.assertEqual(state["llamacpp"]["backend"], "vulkan")
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(ledger["llamacpp_backend"], "vulkan")

    def test_no_ledger_keeps_custom_llamacpp_args(self) -> None:
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        self.assertEqual(defaults["llamacpp_args"], LLAMACPP_MMAP_ARGS)
        for args in ("--threads 8", "  --threads 8  "):
            with self.subTest(args=args):
                state = _merged_factory()
                state["llamacpp"]["args"] = args
                _msg, bodies, ledger = self._drive(state, defaults)
                self.assertTrue(bodies)
                for body in bodies:
                    self.assertNotIn("llamacpp_args", body)
                    self.assertNotIn("llamacpp_vulkan_args", body)
                    self.assertNotIn(LLAMACPP_MMAP_ARGS, json.dumps(body))
                self.assertEqual(state["llamacpp"]["args"], args)
                self.assertIsNotNone(ledger)
                assert ledger is not None
                self.assertNotIn("llamacpp_args", ledger)
                self.assertNotIn("llamacpp_vulkan_args", ledger)

    def test_no_ledger_trimmed_mmap_args_stay_ours(self) -> None:
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        state = _merged_factory()
        state["ctx_size"] = 4096
        state["llamacpp"]["args"] = "  --load-mode mmap  "
        _msg, bodies, ledger = self._drive(state, defaults)
        self.assertEqual(len(bodies), 1)
        self.assertNotIn("llamacpp_args", bodies[0])
        self.assertNotIn(LLAMACPP_MMAP_ARGS, json.dumps(bodies[0]))
        self.assertEqual(state["llamacpp"]["args"], "  --load-mode mmap  ")
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(ledger["llamacpp_args"], LLAMACPP_MMAP_ARGS)

    def test_no_ledger_keeps_max_loaded_models_2(self) -> None:
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        for count in (2, "2"):
            with self.subTest(count=count):
                state = _merged_factory()
                state["max_loaded_models"] = count
                msg, bodies, ledger = self._drive(state, defaults)
                self.assertIn("updated", msg)
                for body in bodies:
                    self.assertNotIn("max_loaded_models", body)
                self.assertEqual(state["max_loaded_models"], count)
                self.assertIsNotNone(ledger)
                assert ledger is not None
                self.assertNotIn("max_loaded_models", ledger)

    def test_load_tuning_values_stay_in_installer_written(self) -> None:
        gib = 1024**3
        ram_gibs = (8, 16, 32, 64, 122, 256, 512)
        fracs = (0.0, 0.10, 0.34, 0.35, 0.49, 0.50, 0.69, 0.70, 0.90)
        absolutes = (0, 2 * gib, 79 * gib, 80 * gib, 105 * gib)

        def machines(ram_gib: int) -> tuple[Hardware, ...]:
            ram = ram_gib * gib
            return (
                Hardware(
                    cpu_name="strix",
                    ram_bytes=ram,
                    devices=(
                        Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="strix",
                    ram_bytes=ram,
                    gfx="gfx1151",
                    devices=(
                        Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "rocm"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="amd",
                    ram_bytes=ram,
                    devices=(
                        Device("dgpu", "amd", "Radeon RX 7600", "/dev/dri/renderD128", "rocm"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="amd",
                    ram_bytes=ram,
                    devices=(
                        Device("dgpu", "amd", "Radeon RX 7900 XT", "/dev/dri/renderD128", "vulkan"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="intel",
                    ram_bytes=ram,
                    devices=(
                        Device("igpu", "intel", "Arc", "/dev/dri/renderD128", "vulkan"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="intel",
                    ram_bytes=ram,
                    devices=(
                        Device("igpu", "intel", "UHD", None, "cpu"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="nvidia",
                    ram_bytes=ram,
                    devices=(
                        Device("dgpu", "nvidia", "RTX 3070", "/dev/dri/renderD128", "cuda"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
                Hardware(
                    cpu_name="cpu",
                    ram_bytes=ram,
                    devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
                ),
                Hardware(
                    cpu_name="amd",
                    ram_bytes=ram,
                    devices=(
                        Device("igpu", "amd", "Raphael", "/dev/dri/renderD128", "rocm"),
                        Device("cpu", "cpu", "cpu", None, "cpu"),
                    ),
                ),
            )

        misses: set[tuple[str, str]] = set()
        saw_mmap = False
        saw_plain = False
        backends: set[object] = set()
        ctxs: set[object] = set()
        for ram_gib in ram_gibs:
            for hw in machines(ram_gib):
                sizes = [int(hw.ram_bytes * frac) for frac in fracs]
                sizes.extend(int(hw.ram_bytes * frac) + 1 for frac in fracs)
                sizes.extend(absolutes)
                for size in sizes:
                    tun = load_tuning(hw, size)
                    if tun.get("llamacpp_args") == LLAMACPP_MMAP_ARGS:
                        saw_mmap = True
                    else:
                        saw_plain = True
                    backends.add(tun.get("llamacpp_backend"))
                    ctxs.add(tun.get("ctx_size"))
                    for key, value in tun.items():
                        allowed = INSTALLER_WRITTEN_VALUES.get(key)
                        if allowed is None or not any(
                            _values_equal(value, item) for item in allowed
                        ):
                            misses.add((key, repr(value)))
        self.assertEqual(misses, set())
        self.assertTrue(saw_mmap)
        self.assertTrue(saw_plain)
        self.assertEqual(backends, {"auto", "vulkan", "rocm"})
        self.assertTrue({-1, 2048, 4096, 8192} <= ctxs)
        self.assertTrue(_installer_wrote_value("ctx_size", "4096"))
        self.assertTrue(_installer_wrote_value("ctx_size", "  4096"))
        self.assertFalse(_installer_wrote_value("ctx_size", 262144))
        self.assertFalse(_installer_wrote_value("ctx_size", "262144"))
        self.assertTrue(_installer_wrote_value("global_timeout", "2400"))
        self.assertTrue(_installer_wrote_value("llamacpp_args", "  --load-mode mmap  "))
        self.assertFalse(_installer_wrote_value("llamacpp_args", " --threads 8 "))
        self.assertTrue(_installer_wrote_value("llamacpp_vulkan_args", ""))
        self.assertFalse(_installer_wrote_value("max_loaded_models", 2))
        self.assertTrue(_installer_wrote_value("max_loaded_models", "1"))

    def test_ledger_round_trip_is_atomic(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger({"ctx_size": 4096, "max_loaded_models": 1}, home)
            path = tuning_ledger_path(home)
            self.assertEqual(read_tuning_ledger(home), {"ctx_size": 4096, "max_loaded_models": 1})
            calls: list[tuple[str, str]] = []
            real_rename = os.rename
            real_open = os.open
            real_fsync = os.fsync
            syncs = {"n": 0}

            def spy_open(file: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
                if isinstance(file, str) and file.endswith(".tmp"):
                    required = os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY | os.O_CLOEXEC
                    self.assertEqual(flags & required, required)
                    self.assertEqual(mode, 0o644)
                if dir_fd is None:
                    return real_open(file, flags, mode)  # type: ignore[arg-type]
                return real_open(file, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

            def spy_rename(
                src: str,
                dst: str,
                *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None,
            ) -> None:
                calls.append((src, dst))
                self.assertNotEqual(src, dst)
                self.assertTrue(src.endswith(".tmp"))
                self.assertEqual(dst, path.name)
                fd = real_open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=src_dir_fd)
                try:
                    text = os.read(fd, 4096).decode("utf-8")
                finally:
                    os.close(fd)
                self.assertIn("2048", text)
                real_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

            def spy_fsync(fd: int) -> None:
                syncs["n"] += 1
                real_fsync(fd)

            with (
                patch("lemonade.os.open", side_effect=spy_open),
                patch("lemonade.os.rename", side_effect=spy_rename),
                patch("lemonade.os.fsync", side_effect=spy_fsync),
                patch("lemonade.os.chown") as chown,
                patch("lemonade.os.lchown") as lchown,
            ):
                write_tuning_ledger({"ctx_size": 2048, "max_loaded_models": 1}, home)
            chown.assert_not_called()
            lchown.assert_not_called()
            self.assertEqual(len(calls), 1)
            self.assertGreaterEqual(syncs["n"], 2)
            self.assertEqual(read_tuning_ledger(home)["ctx_size"], 2048)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_installer_owned_ctx_retunes(self) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        state["ctx_size"] = 4096
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger({"ctx_size": 4096, "max_loaded_models": 1}, home)
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
            ledger = read_tuning_ledger(home)
        self.assertEqual(msg, "lemonade load settings updated")
        self.assertEqual(bodies[0]["ctx_size"], 2048)
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(ledger["ctx_size"], 2048)
        self.assertEqual(ledger["max_loaded_models"], 1)

    def test_user_changed_value_survives_ledger(self) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        state["max_loaded_models"] = 2
        state["ctx_size"] = 16384
        state["global_timeout"] = 900
        state["llamacpp"] = {"backend": "vulkan", "args": "--threads 8", "vulkan_args": ""}

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> object:
            raise AssertionError(getattr(req, "data", b""))

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger(
                {
                    "max_loaded_models": 1,
                    "ctx_size": 4096,
                    "global_timeout": 2400,
                    "llamacpp_backend": "rocm",
                    "llamacpp_args": LLAMACPP_MMAP_ARGS,
                    "llamacpp_vulkan_args": LLAMACPP_MMAP_ARGS,
                },
                home,
            )
            before = tuning_ledger_path(home).read_bytes()
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.shutil.which", return_value=None),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
            self.assertEqual(tuning_ledger_path(home).read_bytes(), before)
        self.assertEqual(msg, "lemonade load settings kept")
        self.assertEqual(state["max_loaded_models"], 2)
        self.assertEqual(state["ctx_size"], 16384)

    def test_unknown_config_writes_nothing_and_keeps_ledger(self) -> None:
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger({"ctx_size": 4096, "max_loaded_models": 2}, home)
            before = tuning_ledger_path(home).read_bytes()

            def urlopen(req: object, timeout: int = 3) -> object:
                raise AssertionError("wrote")

            with (
                patch("lemonade.read_config", return_value={}),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.shutil.which", return_value=None),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
            self.assertEqual(tuning_ledger_path(home).read_bytes(), before)
        self.assertIn("did not report its load settings", msg)
        self.assertIn("left them unchanged", msg)

    def test_failed_set_does_not_update_ledger(self) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        state["ctx_size"] = 4096
        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> object:
            raise OSError("down")

        def run(cmd: list[str]) -> SimpleNamespace:
            return SimpleNamespace(returncode=1, stdout="", stderr="lemonade config set failed")

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_tuning_ledger({"ctx_size": 4096}, home)
            before = tuning_ledger_path(home).read_bytes()
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.shutil.which", return_value="/bin/lemonade-server"),
                patch("lemonade._run", side_effect=run),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    apply_tuning(defaults, ledger_dir=home)
            self.assertIn("config set failed", str(ctx.exception))
            self.assertEqual(tuning_ledger_path(home).read_bytes(), before)

    def test_default_ledger_path_is_var_lib(self) -> None:
        self.assertEqual(DEFAULT_TUNING_LEDGER_DIR, Path("/var/lib/ubuntuai"))
        self.assertEqual(
            tuning_ledger_path(),
            Path("/var/lib/ubuntuai/lemonade-tuning.json"),
        )

    def _victim(self, directory: Path) -> Path:
        victim = directory / "victim"
        victim.write_bytes(b"do-not-touch")
        os.chmod(victim, 0o640)
        return victim

    def _stamp(self, path: Path) -> tuple[bytes, int, int, int]:
        st = os.lstat(path)
        return (path.read_bytes(), st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode))

    def test_symlink_temp_final_and_parent_are_refused(self) -> None:
        cases = ("temp", "final", "parent")
        for kind in cases:
            with self.subTest(kind=kind):
                with TemporaryDirectory() as tmp:
                    home = Path(tmp)
                    victim = self._victim(home)
                    before = self._stamp(victim)
                    ledger_dir = home / "ledger"
                    if kind == "parent":
                        real = home / "real"
                        real.mkdir()
                        victim = self._victim(real)
                        before = self._stamp(victim)
                        ledger_dir.symlink_to(real, target_is_directory=True)
                    else:
                        ledger_dir.mkdir()
                    if kind == "final":
                        tuning_ledger_path(ledger_dir).symlink_to(victim)
                    token = "abcd1234abcd1234"
                    with (
                        patch("lemonade.secrets.token_hex", return_value=token),
                        patch("lemonade.os.chown") as chown,
                        patch("lemonade.os.lchown") as lchown,
                    ):
                        if kind == "temp":
                            planted = ledger_dir / f".lemonade-tuning.json.{token}.tmp"
                            planted.symlink_to(victim)
                        with self.assertRaises(RuntimeError):
                            write_tuning_ledger({"ctx_size": 2048}, ledger_dir)
                    chown.assert_not_called()
                    lchown.assert_not_called()
                    self.assertEqual(self._stamp(victim), before)
                    if kind == "parent":
                        self.assertFalse((home / "real" / "lemonade-tuning.json").exists())

    def _corrupt_migrates(self, payload: bytes) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        state["max_loaded_models"] = 2
        state["ctx_size"] = 4096
        bodies: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            bodies.append(body)
            state.update(body)
            return _Resp()

        defaults = load_tuning(_vulkan_igpu(122), 105 * 1024**3)
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            tuning_ledger_path(home).write_bytes(payload)
            self.assertIsNone(read_tuning_ledger(home))
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
            ):
                msg = apply_tuning(defaults, ledger_dir=home)
        self.assertIn("could not be read", msg)
        self.assertIn("set again", msg)
        self.assertEqual(len(bodies), 1)
        self.assertNotIn("max_loaded_models", bodies[0])
        self.assertEqual(bodies[0]["ctx_size"], 2048)
        self.assertEqual(state["max_loaded_models"], 2)
        self.assertEqual(state["ctx_size"], 2048)

    def test_corrupt_json_is_no_ledger(self) -> None:
        self._corrupt_migrates(b"{")

    def test_non_utf8_ledger_is_no_ledger(self) -> None:
        self._corrupt_migrates(b"\xff\xfe")

    def test_non_dict_ledger_is_no_ledger(self) -> None:
        self._corrupt_migrates(b"[1, 2]")

    def test_ledger_write_failure_is_plain_english(self) -> None:
        factory = _merged_factory()
        state = json.loads(json.dumps(factory))
        posts: list[dict] = []

        class _Resp:
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def read() -> dict:
            return json.loads(json.dumps(state))

        def urlopen(req: object, timeout: int = 3) -> _Resp:
            body = json.loads(req.data.decode())  # type: ignore[attr-defined]
            posts.append(body)
            state.update(body)
            return _Resp()

        def boom(*_args: object, **_kwargs: object) -> None:
            raise PermissionError(13, "Permission denied", "/var/lib/ubuntuai/lemonade-tuning.json")

        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("lemonade.read_config", side_effect=read),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade.urllib.request.urlopen", side_effect=urlopen),
                patch("lemonade.largest_gguf_bytes", return_value=105 * 1024**3),
                patch("lemonade.write_tuning_ledger", side_effect=boom),
            ):
                msg = report_load_tuning(_target(home), _vulkan_igpu(122), ledger_dir=home)
        self.assertTrue(posts)
        self.assertIn("could not be saved", msg)
        self.assertIn("still published", msg)
        self.assertNotIn("did not accept", msg)
        self.assertNotIn("Errno", msg)
        self.assertNotIn("Traceback", msg)


class LemonadeLoadRiskTests(unittest.TestCase):
    def test_policy_is_warn_only(self) -> None:
        self.assertEqual(LOAD_RISK_POLICY, "warn_only")
        risk = load_risk(0.60, 70 * 1024**3, "vulkan")
        self.assertEqual(risk["level"], "strong")
        self.assertEqual(risk["action"], "warn")
        self.assertEqual(risk["policy"], "warn_only")
        self.assertIn("continue", risk_english(risk, ram_bytes=122 * 1024**3).lower())

    def test_frac_warn(self) -> None:
        risk = load_risk(0.40, 40 * 1024**3, "vulkan")
        self.assertEqual(risk["level"], "warn")
        self.assertIn("Warning.", risk_english(risk, ram_bytes=100 * 1024**3))

    def test_absolute_huge_is_strong(self) -> None:
        risk = load_risk(0.20, 105 * 1024**3, "vulkan")
        self.assertGreaterEqual(105 * 1024**3, HUGE_GGUF_BYTES)
        self.assertEqual(risk["level"], "strong")
        self.assertEqual(risk["action"], "warn")

    def test_small_is_ok(self) -> None:
        risk = load_risk(0.10, 2 * 1024**3, "vulkan")
        self.assertEqual(risk["level"], "ok")
        self.assertEqual(risk_english(risk, ram_bytes=122 * 1024**3), "")


class LemonadeVerifyTests(unittest.TestCase):
    def test_verify_accepts_nested_mmap(self) -> None:
        expected = {
            "ctx_size": 4096,
            "global_timeout": 1800,
            "max_loaded_models": 1,
            "llamacpp_args": LLAMACPP_MMAP_ARGS,
        }
        ok, miss = verify_tuning(
            expected,
            {
                "ctx_size": 4096,
                "global_timeout": 1800,
                "max_loaded_models": 1,
                "llamacpp": {"args": "", "vulkan_args": LLAMACPP_MMAP_ARGS},
            },
        )
        self.assertTrue(ok)
        self.assertEqual(miss, "")

    def test_verify_miss_is_soft_english(self) -> None:
        ok, miss = verify_tuning(
            {
                "ctx_size": 4096,
                "global_timeout": 1800,
                "max_loaded_models": 1,
                "llamacpp_args": LLAMACPP_MMAP_ARGS,
            },
            {"ctx_size": 4096, "global_timeout": 1800, "max_loaded_models": 8},
        )
        self.assertFalse(ok)
        self.assertIn("max_loaded_models", miss)
        self.assertIn("still published", miss)

    def test_verify_accepts_preserved_max_loaded_models(self) -> None:
        ok, miss = verify_tuning(
            {
                "ctx_size": 2048,
                "global_timeout": 2400,
                "llamacpp_backend": "vulkan",
            },
            {
                "max_loaded_models": 2,
                "ctx_size": 2048,
                "global_timeout": 2400,
                "llamacpp_backend": "vulkan",
            },
        )
        self.assertTrue(ok, miss)
        self.assertEqual(miss, "")

    def test_verify_skips_mmap_when_key_absent(self) -> None:
        ok, miss = verify_tuning(
            {
                "ctx_size": 4096,
                "global_timeout": 1800,
                "max_loaded_models": 1,
                "llamacpp_args": LLAMACPP_MMAP_ARGS,
            },
            {
                "ctx_size": 4096,
                "global_timeout": 1800,
                "max_loaded_models": 1,
            },
        )
        self.assertTrue(ok)
        self.assertEqual(miss, "")


class LemonadePublishTuningTests(unittest.TestCase):
    def test_publish_applies_load_tuning(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            real = home / "AI models"
            real.mkdir()
            (real / "tiny.gguf").write_bytes(b"G" * 2048)
            dest = Path(tmp) / "snap-common" / "ubuntuai-models"
            hw = _vulkan_igpu(122)
            captured: list[dict] = []

            def record_apply(settings: dict, *_args: object, **_kwargs: object) -> str:
                captured.append(settings)
                return "lemonade load settings updated"

            with (
                patch("lemonade.detect", return_value="snap"),
                patch("lemonade.extra_dir", return_value=dest),
                patch("lemonade._write_bind_unit", return_value="unit.mount"),
                patch("lemonade._set_extra_models_dir"),
                _quiet_daemon(),
                patch("lemonade._restart_snap"),
                patch("lemonade.probe", return_value=hw),
                patch("lemonade.largest_gguf_bytes", return_value=105 * 1024**3),
                patch("lemonade.read_config", return_value=_merged_factory()),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade._load_ledger", return_value=(None, "")),
                patch("lemonade.apply_tuning", side_effect=record_apply),
            ):
                msg = publish(_target(home))
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["llamacpp_args"], LLAMACPP_MMAP_ARGS)
            self.assertEqual(captured[0]["max_loaded_models"], 1)
            self.assertIn("Strong warning", msg)
            self.assertIn(str(dest), msg)

    def test_report_continues_when_verify_misses(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            hw = _vulkan_igpu(122)
            with (
                patch(
                    "lemonade.apply_tuning",
                    return_value=(
                        "Lemonade did not keep the load settings the installer wrote "
                        "(max_loaded_models). Models are still published."
                    ),
                ),
                patch("lemonade.largest_gguf_bytes", return_value=105 * 1024**3),
                patch("lemonade.read_config", return_value=_merged_factory()),
                patch("lemonade.read_factory_defaults", side_effect=_merged_factory),
                patch("lemonade._load_ledger", return_value=(None, "")),
            ):
                text = report_load_tuning(_target(home), hw)
            self.assertIn("still published", text)
            self.assertIn("Strong warning", text)

    def test_shard_bytes_sum_as_one_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            extra = home / "AI models"
            shard = extra / "Huge"
            shard.mkdir(parents=True)
            for i in range(1, 5):
                (shard / f"huge-0000{i}-of-00004.gguf").write_bytes(b"G" * 1000)
            (shard / "mmproj.gguf").write_bytes(b"M" * 100)
            size = largest_gguf_bytes(_target(home, extra=(extra,)))
            self.assertEqual(size, 4000)

    def test_model_gguf_bytes_follows_selected_shard_file(self) -> None:
        with TemporaryDirectory() as tmp:
            shard = Path(tmp) / "Huge"
            shard.mkdir()
            first = shard / "huge-00001-of-00002.gguf"
            first.write_bytes(b"G" * 1000)
            (shard / "huge-00002-of-00002.gguf").write_bytes(b"G" * 1500)
            (shard / "other.gguf").write_bytes(b"X" * 50)
            self.assertEqual(model_gguf_bytes(first), 2500)
            self.assertEqual(model_gguf_bytes(shard), 2500)


if __name__ == "__main__":
    unittest.main()
