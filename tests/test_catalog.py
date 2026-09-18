from __future__ import annotations

import unittest

from support import PKG

from catalog import (
    by_id,
    expand_selection,
    helper_workflow_ids,
    is_helpers_only,
    load_workflows,
    pick_role_winners,
    recommended_ids,
)
from domain import Device, Hardware
from weights import load_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfs = load_workflows(PKG / "workflows.json")

    def test_core_exists(self) -> None:
        ids = {w.id for w in self.wfs}
        self.assertIn("ubuntuai-core", ids)
        self.assertTrue(by_id(self.wfs)["ubuntuai-core"].always)

    def test_ids_unique(self) -> None:
        ids = [w.id for w in self.wfs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_expand_pulls_requires(self) -> None:
        got = expand_selection(("ubuntuai-coding",), self.wfs)
        self.assertEqual(got[0], "ubuntuai-core")
        self.assertIn("ubuntuai-chat", got)
        self.assertIn("ubuntuai-coding", got)

    def test_unknown_id_raises(self) -> None:
        with self.assertRaises(KeyError):
            expand_selection(("ubuntuai-nope",), self.wfs)

    def test_serve_offered_when_rocm_can_be_installed(self) -> None:
        hw = Hardware(
            cpu_name="x",
            ram_bytes=1,
            devices=(
                Device("igpu", "amd", "radeon", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        serve = by_id(self.wfs)["ubuntuai-serve"]
        self.assertTrue(serve.offered(hw))
        self.assertTrue(serve.satisfied(hw))
        self.assertFalse(serve.ready(hw))
        self.assertIn("rocminfo", serve.packages_for(hw))
        self.assertNotIn("nvidia-cuda-toolkit", serve.packages_for(hw))
        hybrid = by_id(self.wfs)["ubuntuai-hybrid"]
        npu = Hardware(
            cpu_name="x",
            ram_bytes=1,
            devices=(
                Device("npu", "amd", "xdna", "/dev/accel/accel0", "xdna"),
                Device("igpu", "amd", "radeon", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        self.assertTrue(hybrid.offered(npu))
        self.assertTrue(npu.hybrid_ok())

    def test_serve_hidden_on_cpu_only(self) -> None:
        hw = Hardware(
            cpu_name="cpu",
            ram_bytes=8 * 1024**3,
            devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
        )
        serve = by_id(self.wfs)["ubuntuai-serve"]
        self.assertFalse(serve.offered(hw))
        self.assertFalse(serve.satisfied(hw))

    def test_serve_offered_on_nvidia_without_smi(self) -> None:
        hw = Hardware(
            cpu_name="x",
            ram_bytes=16 * 1024**3,
            devices=(
                Device("dgpu", "nvidia", "rtx", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        serve = by_id(self.wfs)["ubuntuai-serve"]
        self.assertTrue(serve.offered(hw))
        self.assertIn("cuda", hw.installable_backends())
        self.assertIn("nvidia-cuda-toolkit", serve.packages_for(hw))
        self.assertNotIn("rocminfo", serve.packages_for(hw))

    def test_speech_engines(self) -> None:
        ids = {w.id for w in self.wfs}
        self.assertIn("ubuntuai-tts-openmoss", ids)
        self.assertIn("ubuntuai-tts-rhvoice", ids)
        self.assertIn("ubuntuai-tts-espeak", ids)
        self.assertIn("ubuntuai-stt-whisper", ids)
        self.assertNotIn("ubuntuai-speech", ids)

    def test_strix_defaults_runnable_speech(self) -> None:
        hw = Hardware(
            cpu_name="AMD RYZEN AI MAX+ 395",
            ram_bytes=122 * 1024**3,
            devices=(
                Device("npu", "amd", "xdna", "/dev/accel/accel0", "xdna"),
                Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        winners = pick_role_winners(self.wfs, hw)
        self.assertEqual(winners["stt"], "ubuntuai-stt-whisper")
        self.assertEqual(winners["tts"], "ubuntuai-tts-rhvoice")
        rec = recommended_ids(self.wfs, hw)
        self.assertIn("ubuntuai-tts-rhvoice", rec)
        self.assertIn("ubuntuai-stt-whisper", rec)
        self.assertNotIn("ubuntuai-tts-espeak", rec)
        self.assertIn("ubuntuai-chat", rec)
        chat = by_id(self.wfs)["ubuntuai-chat"]
        self.assertEqual(chat.runtime_bins, ())
        self.assertTrue(chat.offered(hw))
        alias = expand_selection(("ubuntuai-speech",), self.wfs, hw)
        self.assertIn("ubuntuai-tts-rhvoice", alias)
        self.assertIn("ubuntuai-stt-whisper", alias)
        self.assertNotIn("ubuntuai-tts-openmoss", alias)

    def test_low_ram_cpu_still_gets_speech(self) -> None:
        hw = Hardware(
            cpu_name="cpu",
            ram_bytes=2 * 1024**3,
            devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
        )
        winners = pick_role_winners(self.wfs, hw)
        self.assertEqual(winners["tts"], "ubuntuai-tts-rhvoice")
        self.assertEqual(winners["stt"], "ubuntuai-stt-whisper")
        openmoss = by_id(self.wfs)["ubuntuai-tts-openmoss"]
        self.assertFalse(openmoss.ram_ok(hw))
        self.assertEqual(openmoss.vendor, "openmoss")
        self.assertIn("moss-tts-local-q8", openmoss.required_weights)

    def test_helpers_only_presentment(self) -> None:
        index = by_id(self.wfs)
        expected = {
            "ubuntuai-image": "Image helpers",
            "ubuntuai-video": "Video helpers",
            "ubuntuai-rag": "Document helpers",
            "ubuntuai-coding": "Editor helpers",
        }
        for wid, title in expected.items():
            wf = index[wid]
            self.assertTrue(wf.helpers_only, wid)
            self.assertTrue(is_helpers_only(wf), wid)
            self.assertTrue(wf.default, wid)
            self.assertEqual(wf.title, title)
        self.assertFalse(index["ubuntuai-chat"].helpers_only)
        ids = helper_workflow_ids(self.wfs)
        self.assertEqual(ids, frozenset(expected))
        unmarked = tuple(w for w in self.wfs if not w.helpers_only)
        self.assertEqual(helper_workflow_ids(unmarked), frozenset())
        hw = Hardware(
            cpu_name="cpu",
            ram_bytes=16 * 1024**3,
            devices=(Device("cpu", "cpu", "cpu", None, "cpu"),),
        )
        rec = recommended_ids(self.wfs, hw)
        for wid in expected:
            self.assertIn(wid, rec)

    def test_helpers_only_title_mark_without_bool(self) -> None:
        from dataclasses import replace

        chat = by_id(self.wfs)["ubuntuai-chat"]
        marked = replace(chat, title="Chat (helpers)", helpers_only=False)
        summary = replace(chat, summary="Local chat (helpers)", helpers_only=False)
        self.assertTrue(is_helpers_only(marked))
        self.assertTrue(is_helpers_only(summary))
        self.assertEqual(helper_workflow_ids((marked,)), frozenset({chat.id}))

    def test_chat_requires_tiny_default_gguf(self) -> None:
        chat = by_id(self.wfs)["ubuntuai-chat"]
        self.assertIn("qwen3-0.6b-q8_0", chat.required_weights)
        tiny = next(w for w in load_catalog(PKG / "weights.json") if w.id == "qwen3-0.6b-q8_0")
        self.assertTrue(tiny.default)


if __name__ == "__main__":
    unittest.main()
