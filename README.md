# Ubuntu AI Installer

Adds a local-AI workstation layer to Ubuntu or an official flavor. It does not replace the desktop.

A checked box must become a working install. Archive packages are installed with apt. OpenMOSS is fetched from a pinned GitHub release into `~/.local`. Required model files are linked from folders you already have, or downloaded into `~/Models`.

The analog is [Ubuntu Studio Installer](https://ubuntustudio.org/ubuntu-studio-installer/). Core plumbing first. Workflow checkboxes second. A config tool instead of Audio Configuration.

## What it does on this machine

This host is an AMD Ryzen AI MAX+ 395 (Strix Halo) on Ubuntu 26.04.

- iGPU. Radeon 8060S. Vulkan via Mesa.
- NPU. XDNA2 at `/dev/accel/accel0`.
- Archive packages. `llama.cpp-tools` and `libggml0-backend-vulkan`.

The installer puts the seated desktop user in `render` and `video`, sets `memlock` for `@render`, creates `~/Models`, and can install the chat stack from Ubuntu. It will not download 20 GB of weights. It will not silently rewrite NPU firmware. The login name comes from the environment (`SUDO_USER`, `PKEXEC_UID`, `UBUNTUAI_USER`, or the current uid).

## Install from this tree

```
sudo make install PREFIX=/usr/local
ubuntuai-validate
ubuntuai-installer
```

CLI without a GUI:

```
ubuntuai-installer --list
ubuntuai-installer --dry-run ubuntuai-core ubuntuai-chat ubuntuai-hybrid
sudo ubuntuai-installer --install ubuntuai-core ubuntuai-chat ubuntuai-hybrid
ubuntuai-installer --scan-weights
ubuntuai-installer --add-scan-folder /path/to/weights
ubuntuai-installer --scan-folder /tmp/one-off --scan-weights
ubuntuai-installer --organize-weights copy --remove-source
ubuntuai-installer --organize-weights move
ubuntuai-installer --repair
ubuntuai-installer --repair --repair-advisor openai --openai-uri https://api.openai.com/v1 --openai-key "$OPENAI_API_KEY"
ubuntuai-installer --repair --repair-advisor local
ubuntuai-installer --repair-comment "keep the whisper file"
ubuntuai-installer --repair-approve --dry-run
ubuntuai-installer --list-downloads
ubuntuai-validate
```

`--user` is only needed when the process is root and `SUDO_USER` is missing. The Repair tab and `--repair` print a plain-language plan. Approve is a separate action.

## Commands

| Command | Role |
|---|---|
| `ubuntuai-installer` | Checklist GUI, or `--install` on the CLI |
| `ubuntuai-config` | Installed apps, listen address, chat model, speech, GPU, health |
| `ubuntuai-validate` | Devices, groups, firmware, binaries |

## Layout of a checkout

See `AGENTS.md`. Doctrine lives in `DOCTRINE.md`. Style lives in `STYLE.md`.

## Tests

```
make test
```

## License

GPL-3.0-or-later. Same family as Ubuntu Studio Installer.
