from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from support import PKG  # noqa: F401

from users import guess_user


class GuessUserTests(unittest.TestCase):
    def test_explicit_wins(self) -> None:
        self.assertEqual(guess_user("alice"), "alice")

    def test_ubuntuai_user_env(self) -> None:
        env = {
            "UBUNTUAI_USER": "bob",
            "SUDO_USER": "carol",
            "PKEXEC_UID": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(guess_user(), "bob")

    def test_sudo_user_when_root(self) -> None:
        env = {
            "UBUNTUAI_USER": "",
            "SUDO_USER": "dana",
            "PKEXEC_UID": "",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "users.os.getuid", return_value=0
        ), patch("users._seated_user", return_value=None):
            self.assertEqual(guess_user(), "dana")

    def test_current_uid_when_not_root(self) -> None:
        env = {
            "UBUNTUAI_USER": "",
            "SUDO_USER": "",
            "PKEXEC_UID": "",
        }

        class PW:
            pw_name = "erin"

        with patch.dict(os.environ, env, clear=False), patch(
            "users.os.getuid", return_value=1001
        ), patch("users.pwd.getpwuid", return_value=PW()):
            self.assertEqual(guess_user(), "erin")

    def test_root_without_env_raises(self) -> None:
        env = {
            "UBUNTUAI_USER": "",
            "SUDO_USER": "",
            "PKEXEC_UID": "",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "users.os.getuid", return_value=0
        ), patch("users._seated_user", return_value=None):
            with self.assertRaises(RuntimeError):
                guess_user()
