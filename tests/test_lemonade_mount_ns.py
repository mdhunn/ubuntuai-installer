"""Shared-namespace check for the Lemonade embeddings cover.

The host keeps / rshared. A private unshare misses that, so this test
makes / rshared inside an outer unshare before it publishes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from support import PKG


def _observer_script() -> str:
    return textwrap.dedent(
        """
        import os
        import time
        from pathlib import Path

        base = Path(os.environ["UBUNTUAI_NS_BASE"])
        (base / "slave-ready").write_text("ready")

        def listing(path: Path) -> str:
            if not path.is_dir():
                return "MISSING"
            names = sorted(os.listdir(path))
            return ",".join(names) if names else "EMPTY"

        def keep_text(path: Path) -> str:
            file = path / "keep.gguf"
            if not file.is_file():
                return "MISSING"
            return file.read_bytes().decode()

        for tag in ("p1", "p2", "p3", "p4", "p5"):
            go = base / f"go-{tag}"
            while not go.exists():
                time.sleep(0.02)
            source = base / "AI models" / "embeddings"
            text = (
                f"SLAVE_{tag}_SOURCE={listing(source)}\\n"
                f"SLAVE_{tag}_COVER={listing(base / 'dest' / 'chat' / 'AI models' / 'embeddings')}\\n"
                f"SLAVE_{tag}_DESTEMB={listing(base / 'dest' / 'embeddings')}\\n"
                f"SLAVE_{tag}_KEEP={keep_text(source)}\\n"
            )
            (base / f"out-{tag}").write_text(text)
        """
    )


def namespace_child() -> None:
    import time

    import lemonade
    from domain import UserTarget

    def ns_ino(pid: str) -> int:
        return os.stat(f"/proc/{pid}/ns/mnt").st_ino

    if ns_ino("self") == ns_ino("1"):
        sys.stderr.write("unshare did not create a mount namespace\n")
        raise SystemExit(2)

    subprocess.run(["mount", "--make-rshared", "/"], check=True)
    base = Path(os.environ["UBUNTUAI_NS_BASE"])
    source = base / "AI models"
    emb = source / "embeddings"
    emb.mkdir(parents=True)
    from weights import gguf_architecture_header

    (emb / "file.gguf").write_bytes(b"G" * 32)
    (source / "chat.gguf").write_bytes(gguf_architecture_header("llama"))
    dest = base / "dest"
    (dest / "embeddings").mkdir(parents=True)
    (dest / "chat").mkdir(parents=True)
    home = base / "home"
    home.mkdir()
    units = base / "units"
    units.mkdir()
    state = base / "state"
    lemonade.STAGE_DIR = state / "stage"
    lemonade.UNIT_BACKUP_DIR = state / "unit-backups"
    lemonade.SYSTEM_UNIT_DIR = units

    real_run = subprocess.run

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        argv = [str(part) for part in cmd]
        tool = Path(argv[0]).name if argv else ""
        if tool == "systemctl":
            if len(argv) > 1 and argv[1] == "is-active":
                return subprocess.CompletedProcess(argv, 1, "", "")
            if "enable" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if tool in {"mount", "umount"}:
            return real_run(argv, check=False, capture_output=True, text=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    lemonade._run = fake_run
    lemonade.detect = lambda: "snap"
    lemonade.extra_dir = lambda kind: dest
    lemonade._config_answers = lambda: True
    lemonade._set_extra_models_dir = lambda path: None
    lemonade.report_load_tuning = lambda target, hw=None: ""

    target = UserTarget(
        name="owner",
        uid=0,
        gid=0,
        home=home,
        model_root=source,
        extra_model_paths=(),
        bind="127.0.0.1",
    )
    if state.exists():
        sys.stderr.write("state dir existed before publish\n")
        raise SystemExit(2)

    observer = base / "observer.py"
    observer.write_text(_observer_script())
    proc = subprocess.Popen(
        ["unshare", "-m", "--propagation", "slave", sys.executable, str(observer)],
        env={**os.environ, "UBUNTUAI_NS_BASE": str(base)},
    )

    def listing(path: Path) -> str:
        if not path.is_dir():
            return "MISSING"
        names = sorted(os.listdir(path))
        return ",".join(names) if names else "EMPTY"

    def keep_text(path: Path) -> str:
        file = path / "keep.gguf"
        if not file.is_file():
            return "MISSING"
        return file.read_bytes().decode()

    def unescape(raw: str) -> str:
        return raw.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")

    def mount_ids(path: Path) -> list[int]:
        wanted = str(path)
        found: list[int] = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and unescape(parts[4]) == wanted:
                found.append(int(parts[0]))
        return found

    def mounted(path: Path) -> bool:
        return bool(mount_ids(path))

    def mount_ok(argv: list[str]) -> None:
        result = real_run(argv, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(
                f"{' '.join(argv)} -> {result.returncode} {result.stderr}\n"
            )
            raise SystemExit(7)

    def wait_file(path: Path, label: str) -> None:
        for _ in range(300):
            if path.exists():
                return
            if proc.poll() is not None:
                sys.stderr.write(f"{label} observer exited\n")
                raise SystemExit(3)
            time.sleep(0.02)
        sys.stderr.write(f"{label} timed out\n")
        raise SystemExit(3)

    def fresh_lines(tag: str) -> str:
        code = (
            "import os\n"
            "from pathlib import Path\n"
            f"base = Path({str(base)!r})\n"
            "def listing(path):\n"
            "    path = Path(path)\n"
            "    if not path.is_dir():\n"
            "        return 'MISSING'\n"
            "    names = sorted(os.listdir(path))\n"
            "    return ','.join(names) if names else 'EMPTY'\n"
            "def keep_text(path):\n"
            "    file = Path(path) / 'keep.gguf'\n"
            "    if not file.is_file():\n"
            "        return 'MISSING'\n"
            "    return file.read_bytes().decode()\n"
            "source = base / 'AI models' / 'embeddings'\n"
            f"print('FRESH_{tag}_SOURCE=' + listing(source))\n"
            f"print('FRESH_{tag}_COVER=' + listing(base / 'dest' / 'chat' / 'AI models' / 'embeddings'))\n"
            f"print('FRESH_{tag}_DESTEMB=' + listing(base / 'dest' / 'embeddings'))\n"
            f"print('FRESH_{tag}_KEEP=' + keep_text(source))\n"
        )
        result = real_run(
            ["unshare", "-m", "--propagation", "slave", sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode or 4)
        return result.stdout

    def snapshot(tag: str) -> None:
        print(f"HOST_{tag}_SOURCE={listing(emb)}")
        print(f"HOST_{tag}_COVER={listing(dest / 'chat' / 'AI models' / 'embeddings')}")
        print(f"HOST_{tag}_DESTEMB={listing(dest / 'embeddings')}")
        print(f"HOST_{tag}_EMB_MOUNT={int(mounted(emb))}")
        print(f"HOST_{tag}_KEEP={keep_text(emb)}")
        (base / f"go-{tag}").write_text("go")
        wait_file(base / f"out-{tag}", tag)
        print((base / f"out-{tag}").read_text(), end="")
        print(fresh_lines(tag), end="")

    def clear_plan(plan: tuple) -> None:
        for _ in range(2):
            for cmd in lemonade.unmount_commands(plan):
                real_run(list(cmd), check=False, capture_output=True, text=True)

    try:
        wait_file(base / "slave-ready", "slave")
        plan = lemonade._snap_mount_plan(target, dest, lemonade.gguf_sources(target))
        msg = lemonade.publish(target)
        if msg.startswith("ERROR"):
            sys.stderr.write(msg + "\n")
            raise SystemExit(5)
        print("P1_STATE=1" if (state / "stage").is_dir() else "P1_STATE=0")
        print(f"P1_BACKUP={int((state / 'unit-backups').is_dir())}")
        snapshot("p1")

        def one_id(path: Path) -> int:
            found = mount_ids(path)
            return found[-1] if found else -1

        print(f"EMB_ID={one_id(dest / 'embeddings')}")
        print(f"COVER_ID={one_id(lemonade.STAGE_DIR / 'src0' / 'embeddings')}")
        print(f"DEST_COVER_ID={one_id(dest / 'chat' / 'AI models' / 'embeddings')}")
        source_text = real_run(
            ["findmnt", "-n", "-o", "SOURCE", str(dest / "embeddings")],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"DESTEMB_SRC={source_text}")
        prop = real_run(
            ["findmnt", "-n", "-o", "PROPAGATION", str(lemonade.STAGE_DIR / "src0")],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"STAGE_SRC_PROP={prop}")

        before = len(mount_ids(dest / "chat" / "AI models"))
        lemonade.publish(target)
        after = len(mount_ids(dest / "chat" / "AI models"))
        print(f"P2_SRC0_BEFORE={before}")
        print(f"P2_SRC0_AFTER={after}")
        snapshot("p2")

        clear_plan(plan)
        chat = dest / "chat" / "AI models"
        chat.mkdir(parents=True, exist_ok=True)
        mount_ok(["mount", "--bind", str(source), str(chat)])
        mount_ok(
            [
                "mount",
                "-t",
                "tmpfs",
                "-o",
                "ro,nosuid,nodev,noexec,size=64k,mode=0555",
                "tmpfs",
                str(chat / "embeddings"),
            ]
        )
        mount_ok(["mount", "--bind", str(emb), str(dest / "embeddings")])
        print(f"P3_BEFORE_DESTEMB={listing(dest / 'embeddings')}")
        print(f"P3_BEFORE_SOURCE={listing(emb)}")
        lemonade.publish(target)
        snapshot("p3")

        clear_plan(plan)
        mount_ok(["mount", "--bind", str(emb), str(dest / "embeddings")])
        mount_ok(["mount", "--bind", str(source), str(chat)])
        mount_ok(
            [
                "mount",
                "-t",
                "tmpfs",
                "-o",
                "ro,nosuid,nodev,noexec,size=64k,mode=0555",
                "tmpfs",
                str(chat / "embeddings"),
            ]
        )
        print(f"P4_BEFORE_DESTEMB={listing(dest / 'embeddings')}")
        print(f"P4_BEFORE_SOURCE={listing(emb)}")
        lemonade.publish(target)
        snapshot("p4")

        clear_plan(plan)
        mount_ok(
            ["mount", "-t", "tmpfs", "-o", "rw,size=16m", "tmpfs", str(emb)]
        )
        (emb / "keep.gguf").write_bytes(b"keep-data")
        try:
            lemonade.publish(target)
        except RuntimeError as exc:
            print(f"P5_ERR={exc}")
        else:
            print("P5_ERR=")
            sys.stderr.write("user tmpfs was not refused\n")
            raise SystemExit(8)
        snapshot("p5")
        print(f"P5_STILL={int(mounted(emb))}")

        real_run(["umount", str(emb)], check=False, capture_output=True, text=True)
        clear_plan(plan)

        def cover_tmpfs(path: Path, data: bytes) -> None:
            path.mkdir(parents=True, exist_ok=True)
            mount_ok(
                [
                    "mount",
                    "-t",
                    "tmpfs",
                    "-o",
                    "rw,nosuid,nodev,noexec,size=64k,mode=0555",
                    "tmpfs",
                    str(path),
                ]
            )
            (path / "keep.gguf").write_bytes(data)
            mount_ok(
                [
                    "mount",
                    "-o",
                    "remount,ro,nosuid,nodev,noexec,size=64k,mode=0555",
                    str(path),
                ]
            )

        def refused(tag: str) -> None:
            try:
                lemonade.publish(target)
            except RuntimeError as exc:
                print(f"{tag}_ERR={exc}")
            else:
                print(f"{tag}_ERR=")
                sys.stderr.write(f"{tag} exact cover was not refused\n")
                raise SystemExit(8)

        cover_tmpfs(emb, b"keep-exact")
        refused("P6")
        print(f"P6_STILL={int(mounted(emb))}")
        print(f"P6_KEEP={keep_text(emb)}")
        real_run(["umount", str(emb)], check=False, capture_output=True, text=True)

        cover_tmpfs(emb, b"keep-destpeer")
        mount_ok(["mount", "--make-shared", str(emb)])
        mount_ok(["mount", "--bind", str(emb), str(dest / "embeddings")])
        refused("P7")
        print(f"P7_STILL={int(mounted(emb))}")
        print(f"P7_KEEP={keep_text(emb)}")
        real_run(["umount", str(dest / "embeddings")], check=False, capture_output=True, text=True)
        real_run(["umount", str(emb)], check=False, capture_output=True, text=True)

        mount_ok(["mount", "-t", "tmpfs", "-o", "rw,size=64m", "tmpfs", str(source)])
        emb.mkdir(parents=True, exist_ok=True)
        (emb / "file.gguf").write_bytes(b"G" * 32)
        (source / "chat.gguf").write_bytes(b"C" * 32)
        tmpfs_plan = lemonade._snap_mount_plan(target, dest, lemonade.gguf_sources(target))
        msg = lemonade.publish(target)
        if str(msg).startswith("ERROR"):
            sys.stderr.write(str(msg) + "\n")
            raise SystemExit(9)
        before_ids = mount_ids(dest / "embeddings")
        lemonade.publish(target)
        after_ids = mount_ids(dest / "embeddings")
        print("P8_BEFORE=" + ",".join(str(item) for item in before_ids))
        print("P8_AFTER=" + ",".join(str(item) for item in after_ids))
        print(f"P8_COUNT={len(after_ids)}")
        clear_plan(tmpfs_plan)
        real_run(["umount", str(source)], check=False, capture_output=True, text=True)
        left: list[str] = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            mountpoint = unescape(parts[4])
            if mountpoint == str(base) or mountpoint.startswith(str(base) + "/"):
                left.append(mountpoint)
        print("LEFT=" + (",".join(left) if left else "none"))
        print(f"HOST_EMB_AFTER={listing(emb)}")
        print(f"HOST_EMB_MOUNT_AFTER={int(mounted(emb))}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


class LemonadeMountNamespaceTests(unittest.TestCase):
    def test_shared_namespace_cover_does_not_hide_embeddings(self) -> None:
        if os.geteuid() != 0:
            self.skipTest("namespace mount test needs root")
        if shutil.which("unshare") is None or shutil.which("mount") is None:
            self.skipTest("unshare or mount is not installed")
        if shutil.which("findmnt") is None:
            self.skipTest("findmnt is not installed")
        base = Path("/tmp") / f"ubuntuai-ns-{os.getpid()}"
        base.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env["UBUNTUAI_PKG"] = str(PKG)
        env["UBUNTUAI_TESTS"] = str(Path(__file__).resolve().parent)
        env["UBUNTUAI_NS_BASE"] = str(base)
        code = (
            "import os, sys\n"
            "sys.path[:0] = [os.environ['UBUNTUAI_TESTS'], os.environ['UBUNTUAI_PKG']]\n"
            "import test_lemonade_mount_ns as ns\n"
            "ns.namespace_child()\n"
        )
        try:
            result = subprocess.run(
                ["unshare", "-m", "--propagation", "private", sys.executable, "-c", code],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            lines = {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in result.stdout.splitlines()
                if "=" in item
            }
            for tag in ("p1", "p2", "p3", "p4"):
                for who in ("HOST", "SLAVE", "FRESH"):
                    self.assertEqual(lines[f"{who}_{tag}_SOURCE"], "file.gguf", tag)
                    self.assertEqual(lines[f"{who}_{tag}_COVER"], "EMPTY", tag)
                    self.assertEqual(lines[f"{who}_{tag}_DESTEMB"], "file.gguf", tag)
                    self.assertEqual(lines[f"{who}_{tag}_KEEP"], "MISSING", tag)
                self.assertEqual(lines[f"HOST_{tag}_EMB_MOUNT"], "0", tag)
            self.assertEqual(lines["P1_STATE"], "1")
            self.assertEqual(lines["P1_BACKUP"], "1")
            self.assertEqual(lines["P2_SRC0_BEFORE"], "1")
            self.assertEqual(lines["P2_SRC0_AFTER"], "1")
            self.assertEqual(lines["P3_BEFORE_DESTEMB"], "EMPTY")
            self.assertEqual(lines["P3_BEFORE_SOURCE"], "EMPTY")
            self.assertEqual(lines["P4_BEFORE_DESTEMB"], "EMPTY")
            self.assertEqual(lines["P4_BEFORE_SOURCE"], "EMPTY")
            self.assertIn(str(base / "AI models" / "embeddings"), lines["DESTEMB_SRC"])
            self.assertEqual(lines["STAGE_SRC_PROP"], "private")
            self.assertLess(int(lines["EMB_ID"]), int(lines["COVER_ID"]))
            self.assertLess(int(lines["EMB_ID"]), int(lines["DEST_COVER_ID"]))
            self.assertIn("not the empty cover", lines["P5_ERR"])
            self.assertIn("stays as it is", lines["P5_ERR"])
            for who in ("HOST", "SLAVE", "FRESH"):
                self.assertEqual(lines[f"{who}_p5_KEEP"], "keep-data")
                self.assertIn("keep.gguf", lines[f"{who}_p5_SOURCE"])
            self.assertEqual(lines["P5_STILL"], "1")
            self.assertEqual(lines["HOST_p5_EMB_MOUNT"], "1")
            self.assertIn("not the empty cover", lines["P6_ERR"])
            self.assertIn("stays as it is", lines["P6_ERR"])
            self.assertEqual(lines["P6_STILL"], "1")
            self.assertEqual(lines["P6_KEEP"], "keep-exact")
            self.assertIn("not the empty cover", lines["P7_ERR"])
            self.assertIn("stays as it is", lines["P7_ERR"])
            self.assertEqual(lines["P7_STILL"], "1")
            self.assertEqual(lines["P7_KEEP"], "keep-destpeer")
            self.assertEqual(lines["P8_BEFORE"], lines["P8_AFTER"])
            self.assertEqual(lines["P8_COUNT"], "1")
            self.assertNotEqual(lines["P8_BEFORE"], "")
            self.assertEqual(lines["LEFT"], "none")
            self.assertEqual(lines["HOST_EMB_AFTER"], "file.gguf")
            self.assertEqual(lines["HOST_EMB_MOUNT_AFTER"], "0")
            host_hits = [
                line
                for line in Path("/proc/self/mountinfo").read_text().splitlines()
                if str(base) in line
            ]
            self.assertEqual(host_hits, [])
        finally:
            if base.exists():
                shutil.rmtree(base, ignore_errors=True)

    def test_migrate_from_whole_tree_keeps_embeddings(self) -> None:
        if os.geteuid() != 0:
            self.skipTest("namespace mount test needs root")
        if shutil.which("unshare") is None or shutil.which("mount") is None:
            self.skipTest("unshare or mount is not installed")
        base = Path("/tmp") / f"ubuntuai-ns-migrate-{os.getpid()}"
        base.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env["UBUNTUAI_PKG"] = str(PKG)
        env["UBUNTUAI_TESTS"] = str(Path(__file__).resolve().parent)
        env["UBUNTUAI_NS_BASE"] = str(base)
        code = (
            "import os, sys\n"
            "sys.path[:0] = [os.environ['UBUNTUAI_TESTS'], os.environ['UBUNTUAI_PKG']]\n"
            "import test_lemonade_mount_ns as ns\n"
            "ns.namespace_migrate_child()\n"
        )
        try:
            result = subprocess.run(
                ["unshare", "-m", "--propagation", "private", sys.executable, "-c", code],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            lines = {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in result.stdout.splitlines()
                if "=" in item
            }
            self.assertEqual(lines["MIG_HOST_EMB"], "file.gguf")
            self.assertEqual(lines["MIG_HOST_EMB_MOUNT"], "0")
            self.assertEqual(lines["MIG_DEST_EMB"], "file.gguf")
            self.assertEqual(lines["MIG_CHAT"], "model.gguf")
            self.assertEqual(lines["MIG_VOICE_EXISTS"], "1")
            self.assertEqual(lines["MIG_GEMMA_EXISTS"], "1")
            self.assertEqual(lines["MIG_STAGE_MOUNTED"], "0")
            self.assertEqual(lines["MIG_BACKUP"], "1")
            self.assertEqual(lines["MIG_MSG_VOICE"], "1")
            self.assertEqual(lines["MIG_SLAVE_EMB"], "file.gguf")
            self.assertEqual(lines["LEFT"], "none")
            host_hits = [
                line
                for line in Path("/proc/self/mountinfo").read_text().splitlines()
                if str(base) in line
            ]
            self.assertEqual(host_hits, [])
        finally:
            if base.exists():
                shutil.rmtree(base, ignore_errors=True)


def namespace_migrate_child() -> None:
    import lemonade
    from domain import UserTarget
    from weights import gguf_architecture_header

    def ns_ino(pid: str) -> int:
        return os.stat(f"/proc/{pid}/ns/mnt").st_ino

    if ns_ino("self") == ns_ino("1"):
        sys.stderr.write("unshare did not create a mount namespace\n")
        raise SystemExit(2)

    subprocess.run(["mount", "--make-rshared", "/"], check=True)
    base = Path(os.environ["UBUNTUAI_NS_BASE"])
    source = base / "AI models"
    emb = source / "embeddings"
    emb.mkdir(parents=True)
    (emb / "file.gguf").write_bytes(b"G" * 32)
    (source / "chat.gguf").write_bytes(gguf_architecture_header("llama"))
    dest = base / "dest"
    (dest / "embeddings").mkdir(parents=True)
    (dest / "chat" / "src0").mkdir(parents=True)
    home = base / "home"
    home.mkdir()
    units = base / "units"
    units.mkdir()
    state = base / "state"
    lemonade.STAGE_DIR = state / "stage"
    lemonade.UNIT_BACKUP_DIR = state / "unit-backups"
    lemonade.SYSTEM_UNIT_DIR = units
    stage = lemonade.STAGE_DIR
    stage.mkdir(parents=True)
    (stage / "src0").mkdir()

    real_run = subprocess.run

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        argv = [str(part) for part in cmd]
        tool = Path(argv[0]).name if argv else ""
        if tool == "systemctl":
            if len(argv) > 1 and argv[1] == "is-active":
                return subprocess.CompletedProcess(argv, 1, "", "")
            if "enable" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if tool in {"mount", "umount"}:
            return real_run(argv, check=False, capture_output=True, text=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    lemonade._run = fake_run
    lemonade.detect = lambda: "snap"
    lemonade.extra_dir = lambda kind: dest
    lemonade._config_answers = lambda: True
    lemonade._set_extra_models_dir = lambda path: None
    lemonade.report_load_tuning = lambda target, hw=None: ""

    def mount_ok(argv: list[str]) -> None:
        result = real_run(argv, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(f"{' '.join(argv)} -> {result.returncode} {result.stderr}\n")
            raise SystemExit(7)

    def unescape(raw: str) -> str:
        return raw.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")

    def mounted(path: Path) -> bool:
        wanted = str(path)
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and unescape(parts[4]) == wanted:
                return True
        return False

    def listing(path: Path) -> str:
        if not path.is_dir():
            return "MISSING"
        names = sorted(os.listdir(path))
        return ",".join(names) if names else "EMPTY"

    mount_ok(["mount", "-o", "bind,rprivate", str(stage), str(stage)])
    mount_ok(["mount", "-o", "bind,rprivate", str(source), str(stage / "src0")])
    mount_ok(["mount", "--bind", str(emb), str(dest / "embeddings")])
    mount_ok(
        [
            "mount",
            "-t",
            "tmpfs",
            "-o",
            "ro,nosuid,nodev,noexec,size=64k,mode=0555",
            "tmpfs",
            str(stage / "src0" / "embeddings"),
        ]
    )
    mount_ok(["mount", "--rbind", str(stage / "src0"), str(dest / "chat" / "src0")])
    (units / "stage.mount").write_text(
        lemonade.mount_unit_text(stage, stage, options="bind,rprivate,nofail"),
        encoding="utf-8",
    )
    (units / "stage-src.mount").write_text(
        lemonade.mount_unit_text(
            source, stage / "src0", options="bind,rprivate,nofail"
        ),
        encoding="utf-8",
    )
    (units / "cover.mount").write_text(
        lemonade.cover_unit_text(stage / "src0" / "embeddings", "stage-src.mount"),
        encoding="utf-8",
    )
    (units / "rbind.mount").write_text(
        lemonade.mount_unit_text(
            stage / "src0", dest / "chat" / "src0", options="rbind,nofail"
        ),
        encoding="utf-8",
    )
    (units / "carried.mount").write_text(
        lemonade.cover_unit_text(dest / "chat" / "src0" / "embeddings", "rbind.mount"),
        encoding="utf-8",
    )
    (units / "emb.mount").write_text(
        lemonade.mount_unit_text(emb, dest / "embeddings"),
        encoding="utf-8",
    )
    (source / "chat.gguf").unlink()
    gemma = source / "gemma" / "model.gguf"
    gemma.parent.mkdir()
    gemma.write_bytes(gguf_architecture_header("llama"))
    voice = source / "voice" / "model.gguf"
    voice.parent.mkdir()
    voice.write_bytes(gguf_architecture_header("moss-tts-delay"))

    target = UserTarget(
        name="owner",
        uid=0,
        gid=0,
        home=home,
        model_root=source,
        extra_model_paths=(),
        bind="127.0.0.1",
    )
    try:
        msg = lemonade.publish(target)
    except Exception as exc:
        sys.stderr.write(f"publish failed: {exc}\n")
        raise SystemExit(5) from exc
    if msg.startswith("ERROR"):
        sys.stderr.write(msg + "\n")
        raise SystemExit(5)
    print(f"MIG_HOST_EMB={listing(emb)}")
    print(f"MIG_HOST_EMB_MOUNT={int(mounted(emb))}")
    print(f"MIG_DEST_EMB={listing(dest / 'embeddings')}")
    print(f"MIG_CHAT={listing(dest / 'chat' / 'gemma')}")
    print(f"MIG_VOICE_EXISTS={int(voice.is_file())}")
    print(f"MIG_GEMMA_EXISTS={int(gemma.is_file())}")
    print(f"MIG_STAGE_MOUNTED={int(mounted(stage))}")
    backups = list((state / "unit-backups").glob("*")) if (state / "unit-backups").is_dir() else []
    print(f"MIG_BACKUP={int(bool(backups))}")
    print(f"MIG_MSG_VOICE={int(str(voice) in msg)}")
    plan = lemonade._snap_mount_plan(target, dest, lemonade.gguf_sources(target))
    for _ in range(2):
        for cmd in lemonade.unmount_commands(plan):
            real_run(list(cmd), check=False, capture_output=True, text=True)
    left: list[str] = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        mountpoint = unescape(parts[4])
        if mountpoint == str(base) or mountpoint.startswith(str(base) + "/"):
            left.append(mountpoint)
    print("LEFT=" + (",".join(left) if left else "none"))
    # The slave namespace must still see the host embeddings files.
    code = (
        "import os\n"
        "from pathlib import Path\n"
        f"base = Path({str(base)!r})\n"
        "names = sorted(os.listdir(base / 'AI models' / 'embeddings'))\n"
        "print(','.join(names) if names else 'EMPTY')\n"
    )
    slave = real_run(
        ["unshare", "-m", "--propagation", "slave", sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    if slave.returncode != 0:
        sys.stderr.write(slave.stderr)
        raise SystemExit(slave.returncode or 4)
    print("MIG_SLAVE_EMB=" + slave.stdout.strip())


if __name__ == "__main__":
    unittest.main()
