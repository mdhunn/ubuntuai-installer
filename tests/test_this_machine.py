from __future__ import annotations

import unittest

import support  # noqa: F401

from probe import probe


class ThisMachineTests(unittest.TestCase):
    def test_strix_halo_when_present(self) -> None:
        hw = probe()
        if "xdna" not in hw.backends():
            self.skipTest("not an XDNA machine")
        self.assertIn("vulkan", hw.backends())
        self.assertTrue(hw.hybrid_ok())
        nodes = {d.node for d in hw.devices if d.node}
        self.assertTrue(nodes & {"/dev/accel/accel0", "/dev/dri/renderD128"})
