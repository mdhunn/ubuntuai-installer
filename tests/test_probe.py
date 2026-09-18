from __future__ import annotations

import unittest

from support import (
    CPUINFO,
    GENERIC_AMD_LSPCI,
    MEMINFO,
    NVIDIA_LSPCI,
    RX7900_LSPCI,
    STRIX_LSPCI,
)

from unittest.mock import patch

from domain import Device, Hardware
from probe import probe


def _strix_gpu(hw: Hardware) -> Device:
    return next(d for d in hw.devices if d.kind == "igpu" and d.vendor == "amd")


class ProbeTests(unittest.TestCase):
    def test_strix_halo(self) -> None:
        hw = probe(
            lspci_text=STRIX_LSPCI,
            cpuinfo=CPUINFO,
            meminfo=MEMINFO,
            rocminfo=False,
        )
        self.assertIn("xdna", hw.backends())
        self.assertIn("vulkan", hw.backends())
        self.assertNotIn("rocm", hw.backends())
        self.assertIn("rocm", hw.installable_backends())
        self.assertTrue(hw.hybrid_ok())
        self.assertEqual(hw.primary_backend(), "xdna")
        self.assertEqual(_strix_gpu(hw).backend, "vulkan")
        kinds = {d.kind for d in hw.devices}
        self.assertIn("npu", kinds)
        self.assertIn("igpu", kinds)
        self.assertGreater(hw.ram_bytes, 100 * 1024**3)
        self.assertTrue(any("Vulkan" in n for n in hw.notes))

    def test_strix_halo_with_rocminfo_stays_vulkan(self) -> None:
        hw = probe(
            lspci_text=STRIX_LSPCI,
            cpuinfo=CPUINFO,
            meminfo=MEMINFO,
            rocminfo=True,
        )
        gpu = _strix_gpu(hw)
        self.assertEqual(gpu.backend, "vulkan")
        self.assertNotIn("rocm", hw.backends())
        self.assertIn("vulkan", hw.backends())
        self.assertIn("rocm", hw.installable_backends())
        self.assertNotEqual(hw.primary_backend(), "rocm")
        self.assertTrue(hw.strix_halo_class())
        joined = " ".join(hw.notes)
        self.assertIn("Vulkan", joined)
        self.assertIn("gfx1150", joined)
        self.assertIn("libgomp", joined)
        self.assertNotIn("rocminfo is not", joined)

    def test_strix_gfx1150_injector_stays_vulkan(self) -> None:
        hw = probe(
            lspci_text=GENERIC_AMD_LSPCI,
            cpuinfo="model name\t: AMD Ryzen\n",
            meminfo=MEMINFO,
            rocminfo=True,
            gfx="gfx1150",
        )
        gpu = next(d for d in hw.devices if d.vendor == "amd" and d.kind == "igpu")
        self.assertEqual(gpu.backend, "vulkan")
        self.assertNotEqual(hw.primary_backend(), "rocm")
        self.assertTrue(hw.strix_halo_class())

    def test_amd_dgpu_with_rocminfo_stays_rocm(self) -> None:
        hw = probe(
            lspci_text=RX7900_LSPCI,
            cpuinfo="model name\t: AMD Ryzen 9\n",
            meminfo=MEMINFO,
            rocminfo=True,
        )
        gpu = next(d for d in hw.devices if d.vendor == "amd")
        self.assertEqual(gpu.kind, "dgpu")
        self.assertEqual(gpu.backend, "rocm")
        self.assertEqual(hw.primary_backend(), "rocm")
        self.assertFalse(hw.strix_halo_class())
        self.assertFalse(any("gfx1150" in n for n in hw.notes))

    def test_strix_primary_ranks_vulkan_ahead_of_rocm(self) -> None:
        hw = Hardware(
            cpu_name="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
            ram_bytes=122 * 1024**3,
            devices=(
                Device("igpu", "amd", "8060S", "/dev/dri/renderD128", "vulkan"),
                Device("igpu", "amd", "8060S-rocm-label", None, "rocm"),
                Device("cpu", "cpu", "cpu", None, "cpu"),
            ),
        )
        self.assertTrue(hw.strix_halo_class())
        self.assertEqual(hw.primary_backend(), "vulkan")

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
