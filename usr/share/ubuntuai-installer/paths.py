"""Runtime and source-tree paths."""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
WORKFLOWS_FILE = PKG_DIR / "workflows.json"
WEIGHTS_FILE = PKG_DIR / "weights.json"
VENDORS_FILE = PKG_DIR / "vendors.json"
LIMITS_TEMPLATE = PKG_DIR / "limits" / "30-ubuntuai.conf"

ETC_DIR = Path("/etc/ubuntuai")
ENV_FILE = ETC_DIR / "ubuntuai.env"
PROFILE_FILE = Path("/etc/profile.d/ubuntuai.sh")
LIMITS_FILE = Path("/etc/security/limits.d/30-ubuntuai.conf")
USER_CONFIG_NAME = Path("ubuntuai") / "config.json"

HELPER_CANDIDATES = (
    Path("/usr/local/sbin/ubuntuai-installer-helper"),
    Path("/usr/sbin/ubuntuai-installer-helper"),
    Path(__file__).resolve().parents[3] / "usr" / "sbin" / "ubuntuai-installer-helper",
)


def helper_path() -> Path:
    for p in HELPER_CANDIDATES:
        if p.is_file():
            return p
    return HELPER_CANDIDATES[0]


def user_config_path(home: Path) -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / USER_CONFIG_NAME
    return home / ".config" / USER_CONFIG_NAME
