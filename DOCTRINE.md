# Doctrine

Product law for Ubuntu AI Installer. Change this file when the product changes. Do not shadow it in AGENTS.md.

## Job

This is an installer for any person who already has Ubuntu or an official flavor. A checked box must become a working program. Apt packages get installed. Vendor runtimes that are not in Ubuntu get fetched from a pinned release and placed on PATH. Required model files get linked from disk if present, otherwise downloaded.

Copy Ubuntu Studio Installer's job, not its app list.

Studio adds creative metapackages plus the audio graph people get wrong. This installer adds local-AI workflow groups plus the inference graph people get wrong.

It is additive. Kubuntu stays Kubuntu. GNOME stays GNOME. The user does not need a pre-existing AI tree, GitHub literacy, or a CUDA workstation. Probe the machine. Install what that machine can run.

## Four layers

1. **Core.** Drivers, groups, memlock, udev-visible device nodes, shared model store, `ubuntuai-validate`.
2. **Workflows.** Chat, coding, image, video, speech, RAG, serve, train, hybrid. Checkboxes.
3. **Config.** Backend, context versus RAM, quant, bind address, power. The JACK buffer-size analog.
4. **Catalog.** Names and sizes of models live in `weights.json`. The blobs are opt-in downloads or files already on disk.

If core is wrong, every app is a paperweight.

## Hardware first

Probe, then pick one primary backend. Mixing CUDA wheels onto a Vulkan box is a bug.

Chat stays on the backend that already runs. Serve and Train may install a second GPU stack from Ubuntu when the card can use it.

| Silicon | Backend |
|---|---|
| NVIDIA dGPU | CUDA once `nvidia-smi` exists. Serve and Train install `nvidia-cuda-toolkit` if it does not. |
| AMD GPU with `/dev/kfd` | ROCm once `rocminfo` exists. Serve and Train install `rocminfo`, HIP, and the llama.cpp HIP backend if it does not. Vulkan remains the Chat default until then. |
| AMD iGPU without ROCm packages | Vulkan for Chat. Serve and Train stay checkable. Apply installs the archive ROCm packages. |
| AMD XDNA2 NPU | `amdxdna` plus matched firmware plus XRT plus FastFlowLM. Cannot be apt-installed. Hybrid stays grey without the NPU. |
| Intel GPU or NPU | Level Zero or Vulkan. No CUDA or ROCm to install. Serve and Train stay grey. |
| CPU | llama.cpp CPU. First class. Serve and Train stay grey. |

Hybrid is first class on Strix Halo-class APUs. NPU (FastFlowLM) and iGPU (llama.cpp Vulkan) do not discover each other. The installer must wire both.

This machine is the existence proof. Ryzen AI MAX+ 395. Radeon 8060S. XDNA2 at `/dev/accel/accel0`. 122 GiB RAM. Ubuntu 26.04. Kernel 7.0.

## Studio mappings we keep

| Studio | This project |
|---|---|
| `ubuntustudio-audio-core` | `ubuntuai-core` |
| realtime `audio` group | `render` and `video`, plus `/dev/accel` |
| `memlock` unlimited | same, via `@render` |
| Audio Configuration | `ubuntuai-config` |
| fonts metapackage | shared model store |
| logout to apply | relogin after groups, reboot after DKMS |
| GTK4 and Qt6 UIs | same split, thin adapters |
| pkexec helper | `ubuntuai-installer-helper` |

## Shared model store

Ollama, llama.cpp, ComfyUI, and FastFlowLM must not each grow a private blob tree.

Default writable root is `~/Models`. Existing trees such as `~/AI models` are extra search paths. They are not overwritten.

The store is format-aware. GGUF is one format, not the only one.

- Single files: `.gguf`, `.safetensors`, `.onnx`, `.ckpt`, `.pt`, `.pth`, `.ggml`, `.bin`, `.q4nx`
- Directories: Hugging Face trees (`config.json` plus safetensors or `pytorch_model.bin`) stay a directory under `hf/`. Diffusers trees (`model_index.json`) stay under `diffusers/`. EXL2 trees stay under `exl2/`.

Do not flatten a Hugging Face or Diffusers directory into loose files. llama.cpp loads GGUF. vLLM, ComfyUI, and converters load safetensors in place.

Subdirs the installer creates:

```
gguf safetensors mmproj loras vae clip whisper embeddings flm openmoss
hf onnx pytorch diffusers controlnet unet exl2
```

The Weights tab scans `~/AI models`, Downloads, Hugging Face cache, ComfyUI `models`, and `q4nx_files`, plus any folders the user adds. Added folders are stored in `~/.config/ubuntuai/config.json` as `scan_folders`. Scanning `/` is refused. New files are checkboxed. Organize offers symlink, copy, or move. Copy and move place a real file in the store. The original is removed only after the checksum of source and destination match. If the file is a catalog download, that checksum uses the algorithm published on its page. Move always asks for that removal. Copy asks with a checkbox. Symlink never deletes the original. Catalog downloads are unchecked by default and land in the matching subdir.

## Repair

Repair is a separate tab. It is not Apply.

1. Classical diagnose always runs. Validate, config, wrappers, and checksums of catalog models. The hash algorithm is the one published on the download page. Hugging Face, GitHub releases, HTML labels, and sidecar files are queried.
2. Optional advisor. OpenAI-compatible URI and API key from the GUI, CLI, config, or `OPENAI_BASE_URL` / `OPENAI_API_KEY`. A local OpenAI-compatible server is used if `/v1/models` answers. The advisor may call `lookup_checksum` and `web_search`. If the advisor fails, the classical plan is used.
3. Human in the loop. The plan is shown as plain English and a numbered list. The user may request changes, discard, or approve. JSON on disk is not the UI. No advisor step runs before Approve.
4. Allowed steps only. `note`, `redownload`, `reinstall_vendor`, `write_core`, `fix_bind`, `hash_file`, `install_workflow`. Shell from a model is dropped.

The desktop user is never a literal in the tree. `guess_user()` reads the environment. `--user` is an override for root sessions without `SUDO_USER`.

## Potato principles as they apply here

**Laziness Protocol.** A new workflow is a catalog entry, not a new Python package. Delete code that duplicates `workflows.json`.

**Foundational Thinking.** The domain types are `Device`, `Hardware`, `Workflow`, `Action`. Write those first. UI last.

**Experience First.** Defaults must work for a first-time Ubuntu user after Apply. Chat on localhost. TTS that speaks. STT that transcribes. Do not leave a checkbox that only prints "binary not on PATH". If Apply fails, say what happened in sentences. Keep the apt log behind details. Ship fewer polished workflows rather than ten broken frontends. Large vendor stacks such as OpenMOSS stay opt-in because of download size. When they are checked, they install fully.

**Model the Domain.** Backend choice, bind address, and workflow availability are data. Not scattered `if nvidia` blocks in the GUI.

**Boundary Discipline.** Parse `lspci`, sysfs, and apt at the edge. Inside the program, trust `Hardware` and `Workflow`.

**Make Operations Idempotent.** Apply converges. Groups, limits, env files, and apt marks can be re-run.

**Prove It Works.** `ubuntuai-validate` talks to `/dev/dri`, `/dev/kfd`, `/dev/accel`, firmware symlinks, and binaries. Importing a module is not proof.

**Encode Lessons in Structure.** The first NPU firmware mismatch lives in validate, not in a comment. The first `0.0.0.0` footgun lives in config defaults.

**Never Block on the Human.** Reversible dry-runs and catalog edits proceed. Firmware symlink swaps and LAN binds pause. A checked OpenMOSS or whisper box does not wait for a later manual download.

## Refusals

- A new Ubuntu flavor as the only path.
- Pinokio or unpinned `curl | sh` app stores. Pinned GitHub release tarballs listed in `vendors.json` are how non-archive runtimes get installed.
- Ollama as the only engine.
- vLLM or SGLang via `pip`. Serve installs archive ROCm or CUDA. It does not pip-install those stacks.
- Global `pip install torch`.
- Shipping weight files or abliterated defaults.
- Silent cloud fallback.
- Snaps for GPU or NPU runtimes.
- Half-updating NPU firmware so the driver and `npu.sbin` disagree.

## Apt versus vendor

Prefer Ubuntu archive packages. On 26.04 that includes `llama.cpp-tools`, `libggml0-backend-vulkan`, `libggml0-backend-hip`, `rocminfo`, `whisper.cpp`, `rhvoice`, and `espeak-ng`.

When the archive cannot ship a runtime, `vendors.json` names a pinned archive, install prefix, and binaries. OpenMOSS is `pwilkin/openmoss` Vulkan Linux x64 into `~/.local/lib/ubuntuai/openmoss`, with wrappers in `~/.local/bin`. CUDA archives are not the default. They are 650 MiB and Vulkan already runs on NVIDIA.

Required weights live in `weights.json` and are listed on the workflow as `required_weights`. Apply order is: reuse a file already in the scan roots, else download into `~/Models/<subdir>`. Whisper's required ggml is small and is fetched when STT is selected. OpenMOSS local GGUF plus sidecar is large and is fetched only when that TTS box is checked.

Speech defaults stay the engines that become usable with a small apt transaction: RHVoice TTS and whisper.cpp STT. OpenMOSS is the quality TTS option. Checking it installs the server and the GGUF pair. eSpeak NG is the always-on floor. `ubuntuai-speech` on the CLI expands to the two recommended engines.

FastFlowLM and NVIDIA CUDA remain vendor work. Do not add a PPA without the UI saying so.

## Security

Local means local. systemd units bind `127.0.0.1` until config says otherwise. The helper accepts an allowlist of verbs and package names. Model roots cannot escape the user's home.
