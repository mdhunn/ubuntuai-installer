from __future__ import annotations

import unittest

import support  # noqa: F401

from probe import probe
from runtime import explain_setup
from users import guess_user, target_for
from validate import collect, format_health


class ThisMachineTests(unittest.TestCase):
    def test_strix_halo_when_present(self) -> None:
        hw = probe()
        if "xdna" not in hw.backends():
            self.skipTest("not an XDNA machine")
        self.assertTrue({"vulkan", "rocm"} & hw.backends())
        self.assertTrue(hw.hybrid_ok())
        nodes = {d.node for d in hw.devices if d.node}
        self.assertTrue(nodes & {"/dev/accel/accel0", "/dev/dri/renderD128"})

    def test_config_explain_matches_probe(self) -> None:
        hw = probe()
        user = guess_user()
        text = explain_setup(user, target_for(user), hw)
        self.assertIn(hw.cpu_name.split()[0], text)
        self.assertIn("Listen:", text)
        health = format_health(collect(target_for(user), hw))
        self.assertTrue(health)
