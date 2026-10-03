"""Shared-namespace check for the Lemonade embeddings cover.

The host keeps / rshared. A private unshare misses that, so this test
makes / rshared inside an outer unshare before it publishes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path

from support import PKG


def _child_script() -> str:
    return textwrap.dedent(
        r'''
        import os
        import sys
        import time
        import subprocess
        from pathlib import Path

        sys.path.insert(0, os.environ["UBUNTUAI_PKG"])
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
        (emb / "file.gguf").write_bytes(b"G" * 32)
        (source / "chat.gguf").write_bytes(b"C" * 32)
        stage = base / "stage"
        dest = base / "dest"
        (dest / "embeddings").mkdir(parents=True)
        (dest / "chat").mkdir(parents=True)
        stage.mkdir()
        home = base / "home"
        home.mkdir()
        lemonade.STAGE_DIR = stage
        target = UserTarget(
            name="owner",
            uid=0,
            gid=0,
            home=home,
            model_root=source,
            extra_model_paths=(),
            bind="127.0.0.1",
        )
        plan = lemonade._snap_mount_plan(target, dest, lemonade.gguf_sources(target))
        ready = base / "slave-ready"
        flag = base / "go"
        slave_out = base / "slave-out"
        slave = base / "slave.py"
        slave.write_text(
            "import os, time\n"
            "from pathlib import Path\n"
            f"base = Path({str(base)!r})\n"
            "(base / 'slave-ready').write_text('ready')\n"
            "while not (base / 'go').exists():\n"
            "    time.sleep(0.02)\n"
            "def listing(path):\n"
            "    names = sorted(os.listdir(path))\n"
            "    return ','.join(names) if names else 'EMPTY'\n"
            "text = (\n"
            "    'SOURCE=' + listing(base / 'AI models' / 'embeddings') + '\\n'\n"
            "    + 'COVER=' + listing(base / 'dest' / 'chat' / 'src0' / 'embeddings') + '\\n'\n"
            "    + 'DESTEMB=' + listing(base / 'dest' / 'embeddings') + '\\n'\n"
            ")\n"
            "(base / 'slave-out').write_text(text)\n"
        )
        proc = subprocess.Popen(
            ["unshare", "-m", "--propagation", "slave", sys.executable, str(slave)]
        )
        try:
            for _ in range(200):
                if ready.exists():
                    break
                time.sleep(0.02)
            else:
                sys.stderr.write("slave namespace did not start\n")
                raise SystemExit(3)
            lemonade.apply_propagation(plan)

            def listing(path: Path) -> str:
                names = sorted(os.listdir(path))
                return ",".join(names) if names else "EMPTY"

            def mount_id(path: Path) -> int:
                wanted = str(path)
                for line in Path("/proc/self/mountinfo").read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and parts[4].replace("\\040", " ") == wanted:
                        return int(parts[0])
                return -1

            def mounted(path: Path) -> bool:
                return mount_id(path) >= 0

            print(f"HOST_EMB_MOUNT={int(mounted(emb))}")
            print(f"HOST_EMB_FILES={listing(emb)}")
            print(f"HOST_COVER={listing(dest / 'chat' / 'src0' / 'embeddings')}")
            print(f"HOST_DESTEMB={listing(dest / 'embeddings')}")
            print(f"EMB_ID={mount_id(dest / 'embeddings')}")
            print(f"COVER_ID={mount_id(stage / 'src0' / 'embeddings')}")
            print(f"DEST_COVER_ID={mount_id(dest / 'chat' / 'src0' / 'embeddings')}")
            source_text = subprocess.run(
                ["findmnt", "-n", "-o", "SOURCE", str(dest / "embeddings")],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            print(f"DESTEMB_SRC={source_text}")
            prop = subprocess.run(
                ["findmnt", "-n", "-o", "PROPAGATION", str(stage / "src0")],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            print(f"STAGE_SRC_PROP={prop}")
            flag.write_text("go")
            proc.wait(timeout=10)
            print(slave_out.read_text())
            fresh = subprocess.run(
                [
                    "unshare",
                    "-m",
                    "--propagation",
                    "slave",
                    sys.executable,
                    "-c",
                    "import os\n"
                    "from pathlib import Path\n"
                    f"base = Path({str(base)!r})\n"
                    "def listing(path):\n"
                    "    names = sorted(os.listdir(path))\n"
                    "    print(','.join(names) if names else 'EMPTY')\n"
                    "print('FRESH_SOURCE', end=' ')\n"
                    "listing(base / 'AI models' / 'embeddings')\n"
                    "print('FRESH_COVER', end=' ')\n"
                    "listing(base / 'dest' / 'chat' / 'src0' / 'embeddings')\n"
                    "print('FRESH_DESTEMB', end=' ')\n"
                    "listing(base / 'dest' / 'embeddings')\n",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if fresh.returncode != 0:
                sys.stderr.write(fresh.stderr)
                raise SystemExit(fresh.returncode)
            print(fresh.stdout, end="")
            for cmd in lemonade.unmount_commands(plan):
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    sys.stderr.write(f"{' '.join(cmd)} {result.returncode} {result.stderr}\n")
                    raise SystemExit(4)
            left = []
            for line in Path("/proc/self/mountinfo").read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                mountpoint = parts[4].replace("\\040", " ")
                if mountpoint == str(base) or mountpoint.startswith(str(base) + "/"):
                    left.append(mountpoint)
            print("LEFT=" + (",".join(left) if left else "none"))
            print(f"HOST_EMB_AFTER={listing(emb)}")
            print(f"HOST_EMB_MOUNT_AFTER={int(mounted(emb))}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        '''
    )


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
        env["UBUNTUAI_NS_BASE"] = str(base)
        try:
            result = subprocess.run(
                [
                    "unshare",
                    "-m",
                    "--propagation",
                    "private",
                    sys_executable(),
                    "-c",
                    _child_script(),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            lines = dict(
                item.split("=", 1)
                for item in result.stdout.splitlines()
                if "=" in item and not item.startswith("FRESH_")
            )
            fresh = {
                item.split(" ", 1)[0]: item.split(" ", 1)[1]
                for item in result.stdout.splitlines()
                if item.startswith("FRESH_")
            }
            self.assertEqual(lines["HOST_EMB_MOUNT"], "0")
            self.assertEqual(lines["HOST_EMB_FILES"], "file.gguf")
            self.assertEqual(lines["HOST_COVER"], "EMPTY")
            self.assertEqual(lines["HOST_DESTEMB"], "file.gguf")
            self.assertEqual(lines["SOURCE"], "file.gguf")
            self.assertEqual(lines["COVER"], "EMPTY")
            self.assertEqual(lines["DESTEMB"], "file.gguf")
            self.assertEqual(fresh["FRESH_SOURCE"], "file.gguf")
            self.assertEqual(fresh["FRESH_COVER"], "EMPTY")
            self.assertEqual(fresh["FRESH_DESTEMB"], "file.gguf")
            self.assertIn(str(base / "AI models" / "embeddings"), lines["DESTEMB_SRC"])
            self.assertEqual(lines["STAGE_SRC_PROP"], "private")
            self.assertLess(int(lines["EMB_ID"]), int(lines["COVER_ID"]))
            self.assertLess(int(lines["EMB_ID"]), int(lines["DEST_COVER_ID"]))
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


def sys_executable() -> str:
    return os.environ.get("PYTHON", "") or __import__("sys").executable


if __name__ == "__main__":
    unittest.main()
