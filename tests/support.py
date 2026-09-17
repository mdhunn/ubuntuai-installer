from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "usr" / "share" / "ubuntuai-installer"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

STRIX_LSPCI = """
00:00.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Root Complex [1022:1507]
67:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics] [1002:1586] (rev c1)
68:00.1 Signal processing controller [1180]: Advanced Micro Devices, Inc. [AMD] Strix/Krackan/Strix Halo Neural Processing Unit [1022:17f0] (rev 11)
""".strip()

NVIDIA_LSPCI = """
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA104 [GeForce RTX 3070] [10de:2484] (rev a1)
""".strip()

CPUINFO = "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n"
MEMINFO = "MemTotal:       128000000 kB\n"
