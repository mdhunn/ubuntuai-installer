# AGENTS.md

This file is for coding agents. Humans start at `README.md`.

Read `DOCTRINE.md` then `STYLE.md` before the first edit. Those two files own product law and style. Do not copy their lists into this file.

## What this is

Ubuntu AI Installer adds a local-AI workstation layer to an official Ubuntu flavor. It does not convert the desktop. The analog is `ubuntustudio-installer`.

The audience is any Ubuntu user. They are not assumed to have models, GitHub clones, or CUDA. A checked workflow must end in a runnable binary and the files that binary needs.

Core plumbing still matters. Drivers, groups, memlock, a shared model store, and `ubuntuai-validate`. Apps are checkboxes on top. Checkboxes that only detect missing tools are a bug.

## Commands

```
make test
make install PREFIX=/usr/local
make uninstall PREFIX=/usr/local
ubuntuai-installer --list
ubuntuai-installer --install ubuntuai-core ubuntuai-chat --dry-run
ubuntuai-installer --install ubuntuai-core ubuntuai-chat
ubuntuai-installer --scan-weights
ubuntuai-installer --scan-folder /path/to/weights
ubuntuai-installer --add-scan-folder /path/to/weights
ubuntuai-installer --organize-weights
ubuntuai-installer --organize-weights copy --remove-source
ubuntuai-installer --repair
ubuntuai-installer --repair --repair-advisor openai --openai-uri https://api.openai.com/v1 --openai-key "$OPENAI_API_KEY"
ubuntuai-installer --repair-approve
ubuntuai-installer --list-downloads
ubuntuai-validate
ubuntuai-config --explain
ubuntuai-config --show
```

GUI needs a session bus and a display. CLI must remain complete without either.

## Layout

| Path | Owns |
|---|---|
| `usr/share/ubuntuai-installer/workflows.json` | Workflow catalog. Source of truth for names, apt, groups, backends. |
| `usr/share/ubuntuai-installer/weights.json` | Weight catalog. Required ids are fetched on apply if missing. |
| `usr/share/ubuntuai-installer/vendors.json` | Pinned vendor runtimes. URL, archive flavor, binaries. |
| `usr/share/ubuntuai-installer/vendor.py` | Download, extract, wrap vendor binaries into `~/.local`. |
| `usr/share/ubuntuai-installer/weights.py` | Scan, classify, symlink/copy/move, download, ensure required weights. |
| `usr/share/ubuntuai-installer/repair.py` | Diagnose, plan, human-approved repair of config, vendors, and checksums. |
| `usr/share/ubuntuai-installer/runtime.py` | Which installer workflows are present and how to reach them. |
| `usr/share/ubuntuai-installer/probe.py` | Hardware report. |
| `usr/share/ubuntuai-installer/apply.py` | Plan and privileged apply. Idempotent. |
| `usr/share/ubuntuai-installer/validate.py` | Real checks against devices, groups, firmware, binaries. |
| `usr/share/ubuntuai-installer/ui/` | Thin GTK4 and Qt6 adapters. No business logic. |
| `usr/sbin/ubuntuai-installer-helper` | pkexec target. The only root writer. |
| `tests/` | unittest. No pytest. |

Install copies `usr/` into `PREFIX`. Keep runtime paths working from a source tree too.

## Invariants

1. Catalog changes go in `workflows.json` first. Python reads the catalog. Do not hard-code workflow ids in UI code.
2. Probe never installs. Validate never installs. Apply is the only writer.
3. Apply is idempotent. A second run converges to the same files and groups.
4. Privileged writes go through the helper. Package names must match `^[a-zA-Z0-9.+-]+$`. `model_root` must stay under the target user's home.
5. Bind defaults to `127.0.0.1`. `0.0.0.0` needs an explicit config choice.
6. No model weights in git. Organize offers symlink, copy, or move. Copy/move may delete the original only after a checksum match. Do not assume SHA-256. Use the algorithm the download page or API publishes. No unpinned `curl | sh`. No `pip install torch` on system Python.
7. Firmware is a matched pair. Never half-update NPU firmware from the installer.
8. Grey out a workflow only when the required silicon cannot exist on this machine. Missing archive packages are Apply's job. Serve and Train stay available when an AMD or NVIDIA GPU is present. Apply installs ROCm or CUDA from Ubuntu. An NPU-only or CPU-only box still hides them. Do not pip-install PyTorch.
9. Speech engines have `role` `tts` or `stt`. Defaults come from `pick_role_winners`. `ubuntuai-speech` is a CLI alias for those winners.
10. A checked workflow with `vendor` must install that runtime from `vendors.json`. A checked workflow with `required_weights` must link or download every id. Apply is not done if those steps are skipped.
11. Vendor installs go under `~/.local` so a desktop user does not need a second password after groups are set. Put `~/.local/bin` on PATH from `/etc/profile.d/ubuntuai.sh`.
12. Weights are classified by suffix and by directory shape. Hugging Face and Diffusers trees are one store entry. Do not flatten them. `classify()` is for files. `detect_bundle()` is for directories.
13. Repair is human-in-the-loop. Diagnose and plan never apply themselves. Approve is a separate action. Advisor JSON may only use allowlisted step kinds in `repair.py`.
14. Never hard-code a desktop username. `guess_user()` reads `--user`, then `UBUNTUAI_USER`, `PKEXEC_UID`, `SUDO_USER`, then the current uid. Docs omit `--user`. Tests use the process user or a name that does not need `/etc/passwd`.
15. Checksums come from the download page. Parse `oid sha256:`, GitHub `digest`, HTML labels, and sidecar `.md5` / `.sha256` / `.sha1` / `.sha512`. Verify with `hashlib.new(algo)`. Copy/move of local files still compares both sides so a mismatch cannot delete the original.
16. Repair plans are shown as plain English plus a numbered list. `repair-plan.json` is an implementation file. It is not the user interface.
17. The OpenAI repair path takes a URI and an API key. Both can come from the GUI, `--openai-uri` / `--openai-key`, config, or `OPENAI_BASE_URL` / `OPENAI_API_KEY`.
18. Apply failures show a plain-English reason first. Technical logs (helper command, exit code, apt text) are behind a details expander in the GUI and after a "Technical details" heading on the CLI. Capture helper stdout and stderr. Do not stop at `failed with 1`.
19. `ubuntuai-config` only tunes apps the installer actually put on the machine (recorded Apply plus live binaries). Overview is sentences. Settings are chat model, speech, and GPU. Advanced is paths, URI, key, and raw checks. `--explain` is the CLI for new users. `--show` is JSON.

## How to change a workflow

1. Edit `workflows.json`.
2. Add or adjust a unittest that loads the catalog.
3. Run `make test`.
4. Dry-run on this machine before a live apply.

## How to change hardware detection

1. Edit `probe.py` injectors so tests can pass fake `lspci` and sysfs.
2. Cover the new device in `tests/test_probe.py`.
3. Run `ubuntuai-validate` on real hardware before calling it done.

## Definition of done

- `make test` passes.
- `ubuntuai-validate` was run on the real machine for plumbing changes.
- GUI changes were opened, not only imported, when a display is available.
- Docs that name a command still match the CLI.

## Out of bounds

Do not add a desktop flavor. Do not add Pinokio. Do not add unpinned install scripts. Do not leave OpenMOSS or whisper as detect-only. Workflow ids stay in `workflows.json`. Weight blobs stay in `weights.json`. Vendor archives stay in `vendors.json`.
