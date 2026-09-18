from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from support import PKG  # noqa: F401

from domain import Action, UserTarget
from lemonade import (
    APPLY_PUBLISH_VERB,
    BindMount,
    bind_mounts,
    detect,
    extra_dir,
    gguf_sources,
    mount_unit_text,
    publish,
    quote_unit_path,
)
from paths import (
    SNAP_LEMONADE_MODELS,
    is_home_path,
    lemonade_extra_models_dir,
)


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
                (BindMount(extra.resolve(), dest),),
            )

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

    def test_bind_mounts_single_source_on_dest(self) -> None:
        dest = SNAP_LEMONADE_MODELS
        src = Path("/var/tmp/ai-models")
        self.assertEqual(bind_mounts((src,), dest), (BindMount(src.resolve(), dest),))

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
            self.assertEqual(mounts, (BindMount(extra.resolve(), dest),))
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
            self.assertEqual(mounts, (BindMount(extra.resolve(), dest),))
            self.assertEqual(len(mounts), 1)
            self.assertEqual(mounts[0].where, dest)

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

    def test_mount_unit_quotes_spaces(self) -> None:
        what = Path("/tmp/AI models")
        where = SNAP_LEMONADE_MODELS
        text = mount_unit_text(what, where)
        self.assertIn(f"What={quote_unit_path(what)}", text)
        self.assertIn(f"Where={quote_unit_path(where)}", text)
        self.assertIn('What="/tmp/AI models"', text)
        self.assertIn("Type=none", text)
        self.assertIn("Options=bind", text)

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
                patch("lemonade._restart_snap"),
            ):
                msg = publish(_target(home))
            self.assertEqual(binds, [(real.resolve(), dest)])
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
                patch("lemonade._restart_snap"),
            ):
                msg = publish(_target(home, extra=(extra,), model_root=store))
            self.assertEqual(binds, [(extra.resolve(), dest)])
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
                patch("lemonade._restart_snap"),
            ):
                msg = publish(_target(home, extra=(extra,), model_root=home / "Models"))
            self.assertEqual(
                binds,
                [
                    (extra.resolve(), dest / "chat" / "src0"),
                    (store.resolve(), dest / "chat" / "src1"),
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

    def test_apply_publish_verb(self) -> None:
        self.assertEqual(APPLY_PUBLISH_VERB, "lemonade-publish")
        action = Action("lemonade", "publish GGUF files to Lemonade", ("owner",))
        self.assertEqual(action.kind, "lemonade")


if __name__ == "__main__":
    unittest.main()
