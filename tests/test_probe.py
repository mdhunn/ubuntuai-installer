from __future__ import annotations

import unittest

from support import CPUINFO, MEMINFO, NVIDIA_LSPCI, STRIX_LSPCI

from unittest.mock import patch

from probe import probe


class ProbeTests(unittest.TestCase):
    def test_strix_halo(self) -> None:
        with patch("probe.shutil.which", return_value=None):
            hw = probe(lspci_text=STRIX_LSPCI, cpuinfo=CPUINFO, meminfo=MEMINFO)
        self.assertIn("xdna", hw.backends())
        self.assertIn("vulkan", hw.backends())
        self.assertIn("rocm", hw.installable_backends())
        self.assertTrue(hw.hybrid_ok())
        self.assertEqual(hw.primary_backend(), "xdna")
        kinds = {d.kind for d in hw.devices}
        self.assertIn("npu", kinds)
        self.assertIn("igpu", kinds)
        self.assertGreater(hw.ram_bytes, 100 * 1024**3)

    def test_nvidia_without_smi_is_vulkan(self) -> None:
        with patch("probe.shutil.which", return_value=None):
            hw = probe(lspci_text=NVIDIA_LSPCI, cpuinfo=CPUINFO, meminfo=MEMINFO)
        gpu = next(d for d in hw.devices if d.vendor == "nvidia")
        self.assertEqual(gpu.kind, "dgpu")
        self.assertEqual(gpu.backend, "vulkan")
        self.assertIn("cuda", hw.installable_backends())

    def test_cpu_always_present(self) -> None:
        hw = probe(lspci_text="", cpuinfo=CPUINFO, meminfo=MEMINFO)
        self.assertEqual(hw.backends(), frozenset({"cpu"}))
        self.assertEqual(hw.installable_backends(), frozenset({"cpu"}))
        self.assertFalse(hw.hybrid_ok())


if __name__ == "__main__":
    unittest.main()
