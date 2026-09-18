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

# Lemonade snap ProtectHome cannot read /home or follow ~/Models symlinks.
SNAP_LEMONADE_COMMON = Path("/var/snap/lemonade-server/common")
SNAP_LEMONADE_MODELS = SNAP_LEMONADE_COMMON / "ubuntuai-models"

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


def lemonade_extra_models_dir(kind: str) -> Path:
    """Snap-visible extra_models_dir. Empty when Lemonade is not a snap."""
    if kind == "snap":
        return SNAP_LEMONADE_MODELS
    return Path()


def is_home_path(path: Path) -> bool:
    """True when path is under /home. The snap cannot use that as extra_models_dir."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    parts = resolved.parts
    return len(parts) >= 2 and parts[0] == "/" and parts[1] == "home"
