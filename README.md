# MOTIONSALT Upscaler — Xyether Edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/motionssalt/upscale/blob/main/MOTIONSALT_Upscaler.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A private Google Colab notebook that runs the **Xyether** anime upscale & CC engine on a free
Colab GPU. The engine is shipped as a self-contained compiled module (`engine/config_utils.so`)
and is loaded directly from this repo — nothing is fetched from Drive, and there is **no
password / OTP step**.

---

## What this repo is

- `MOTIONSALT_Upscaler.ipynb` — the two-cell Colab notebook (**Step 1: Load engine → Step 2: Pick clip & process**).
- `engine/config_utils.so` — the compiled Xyether engine (owner's own self-trained models baked in).
- `LICENSE` — MIT for the notebook code. The engine and its trained models are private, all rights reserved by the owner.

The previous AnimeJaNai / Real-ESRGAN pipeline (including the weekly weight-mirroring GitHub
Action and `scripts/fetch_latest_weights.py`) has been fully retired — this repo now runs
exclusively on the Xyether engine.

---

## How to use it

1. **Open in Colab** via the badge above.
2. **Enable a GPU:** `Runtime` → `Change runtime type` → `GPU` → Save.
3. **Step 1 — Load engine.** Run the first code cell. It installs runtime dependencies
   (`pymongo`, `dnspython`, `gdown`, `pycuda`, `tensorrt==11.0.0.114`, `opencv-python`,
   `pillow`, `pepedpid`, plus `ffmpeg`) and loads `engine/config_utils.so` directly from this
   repo using the exact `importlib.util.spec_from_file_location` + `exec_module` pattern the
   engine expects.
4. **Step 2 — Pick clip & process.** Configure the form:
   - **Target File / Folder Path** — paste an existing `/content/...` path, **or** leave it
     empty to use the inline file picker that appears when the cell runs (no need to open
     Colab's file-browser panel).
   - **Model Selection** — one of the ten Xyether trained models:
     - `2x Xyether Anime Sharp`
     - `2x Xyether Anime Soft`
     - `1x Xyether Compression Remover`
     - `1x Xyether Dark CC v1`
     - `2x Xyether Dark cc V2`
     - `2x xyether Dark Blue CC`
     - `2x xyether White CC`
     - `1x Xyether Tiktok CC (Soft)`
     - `1x xyether tiktok cc (strong)`
     - `1x Faster Xyether Compression Remover`
   - **Speed_Boost / Force_1080p** — booleans, passed through to the engine unchanged.
   - **codec / crf_value** — `h264_nvenc` or `hevc_nvenc`, CRF 0–23.
   - **auto_download** — direct browser download when the engine finishes.
   - **create_share_link** — after the engine finishes, upload the output to `tmpfiles.org` and
     print a **~1-hour shareable direct-download link** so you can grab the file from a
     different browser or device.

The cell calls the engine with the exact original signature:

```python
config_utils.run_processing(
    video_path=video_path,
    model_choice=model_choice,
    Speed_Boost=Speed_Boost,
    Force_1080p=Force_1080p,
    codec=codec,
    crf_value=crf_value,
    auto_download=auto_download,
)
```

Nothing about the engine or its call has been modified — the only structural change vs. the
original notebook is that the `authenticate_telegram(Auth_Token)` gate around it is gone.

---

## Repo layout

```
.
├── MOTIONSALT_Upscaler.ipynb    # The Colab notebook (2 form cells + intro/footer md)
├── engine/
│   └── config_utils.so          # Compiled Xyether engine (self-contained)
├── LICENSE                      # MIT (notebook code only)
└── README.md
```

---

## License

The notebook code is MIT-licensed (see [LICENSE](LICENSE)). The `engine/config_utils.so` binary
and the trained models baked into it are the owner's private work — all rights reserved, not
redistributable.
