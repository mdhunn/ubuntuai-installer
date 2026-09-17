# Style guide

For agents and humans. Prose in this repo follows the same rules as the code comments.

## Prose

Short declarative sentences. One thought per sentence.

The long dash is banned. A colon as a mid-sentence connector is banned. A colon before a list is fine.

Do not write "this simply", "just", "easily", "robust", or "production-ready". Name the file or the command.

User-facing strings in the GUI follow this too.

Never write a login name in docs, comments, or tests. The seated user comes from the environment. CLI examples omit `--user`.

Repair text shown to a human is sentences. JSON keys and step kinds stay in `repair.py`. The GUI and CLI print `format_plan()`, not the saved JSON.

A failed Apply prints English first. Exit codes, apt logs, and helper command lines belong in technical details, not in the heading.

Checksum copy in the UI is "checksum". Name an algorithm only when a download page published that algorithm.

Comments exist only for a non-obvious why. No phase labels. No narration of the next line.

## Python

- Python 3.12+. This host runs 3.14. Stay in the stdlib plus PyGObject or PyQt6.
- `from __future__ import annotations`.
- Frozen dataclasses for `Device`, `Hardware`, `Workflow`, `Action`, `Check`.
- Functions return data. They do not print unless they are the CLI edge.
- No mutable default arguments.
- No global install of PyTorch. HTTP uses stdlib `urllib`. Vendor archives extract with `tarfile` and `filter="data"`.
- `subprocess` always with a list of args. Never `shell=True`.
- Paths are `pathlib.Path`.

Name things after the domain. `Hardware.backends()` not `get_accel_set()`.

## JSON catalogs

Three JSON files. Do not invent a fourth format.

- `workflows.json`. `id` is the metapackage name (`ubuntuai-chat`, not `Chat`). `apt` lists Ubuntu package names always needed. `apt_for_backend` maps a backend id to extra packages. `vendor` is a key in `vendors.json`. `required_weights` lists ids in `weights.json`.
- `weights.json`. Filenames, URLs, sizes. Optional `checksum` as `algo:hex`. Empty means look it up from the download page. Never the blobs.
- `vendors.json`. Pinned archive URLs and binary names.

`hide_unless_backend` is a list of backend ids. Offer the checkbox when probe already has one of them, or when `Hardware.installable_backends()` can add one via apt (AMD GPU → ROCm, NVIDIA GPU → CUDA). Empty means always offered. Grey out only when the silicon is missing. `apt_for_backend` lists extra packages per backend. Apply merges those with `apt`.

## UI

GTK4 plus libadwaita is the GNOME path. Qt6 is the Plasma path. `main.py` picks. Both call `apply.py` and `probe.py`. Neither toolkit owns policy.

No custom theme. No bundled icon font. One SVG app icon.

## Tests

`unittest` only. Tests inject fake `lspci` text and temp dirs. They do not run apt. They do not require a display. They do not hard-code a login. `pwd.getpwuid(os.getuid())` is the process user.

A hardware test may skip when XDNA is absent. It must not fake a pass.

## Git

Small commits. One invariant or one workflow per commit when possible. Do not commit weights, `__pycache__`, or `/etc` copies.
