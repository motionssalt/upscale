# MOTIONSALT Upscaler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/motionssalt/upscale/blob/main/MOTIONSALT_Upscaler.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Weights Auto-Mirror](https://github.com/YOUR-USERNAME/motionsalt-upscaler/actions/workflows/check-model-updates.yml/badge.svg)](.github/workflows/check-model-updates.yml)

**A free, no-install anime video upscaler that runs entirely in your browser — a free alternative to Topaz Video AI for people without a PC or GPU.**

MOTIONSALT Upscaler is a Google Colab notebook that upscales anime video using open-source models (AnimeJaNai V3 and Real-ESRGAN AnimeVideo v3). You don't install anything, you don't need a graphics card, and you don't pay anything — the upscaling runs on Google's free servers.

The native upscale factor depends on the quality tier you pick — the two AnimeJaNai V3 tiers are 2× models, and the Real-ESRGAN AnimeVideo v3 tier is a 4× model. The factor is read straight off the loaded model at runtime (never hardcoded), and the optional 1080p-height finish step lets you land any of them at a consistent output height.

---

## What this tool does

- Takes an anime video file you upload
- Runs it through one of three open-source anime upscaling models (LOW / MEDIUM / HIGH quality tiers)
- Applies configurable image cleanup (sharpen, denoise, dehalo, deblock, detail recovery)
- Keeps the original audio track intact
- Optionally downscales the result to 1080p **height** while preserving the source aspect ratio (a 4:3 source stays 4:3, a 16:9 source becomes 1920×1080 — no stretching, no black bars added)
- Hands the finished file back to your browser as a direct download

Nothing about your video is uploaded to a third-party service. It only ever lives on the Colab VM assigned to your session, and it's discarded when the session ends.

---

## How to use it (first-timer friendly)

If you've never opened a Google Colab notebook before, don't worry — this is designed for you.

1. **Click the "Open in Colab" badge** at the top of this README. It opens the notebook in your browser. You need a free Google account.
2. **Enable a GPU.** In Colab's top menu: `Runtime` → `Change runtime type` → set **Hardware accelerator** to `GPU` → Save. (Free tier is fine — you'll usually get a T4.)
3. **Run the notebook.** Either click `Runtime` → `Run all`, or click the ▶ button on each cell top-to-bottom. The notebook then shows a clean 4-step wizard:
   - **Step 1 — Connect:** Click *Connect*. It verifies the GPU, installs dependencies, and downloads the AI model weights from this repository's GitHub Releases.
   - **Step 2 — Upload:** Click the file picker and pick your video. A progress bar shows the upload.
   - **Step 3 — Configure:** Pick a quality tier, set the sliders however you like, decide if you want to downscale to 1080p at the end, then click *Start Processing*. A real progress bar tracks the frames as they're processed.
   - **Step 4 — Download:** Click *⬇️ Download Result*. Your browser downloads the finished video directly.

That's it. You never touch code. You never install anything on your computer.

---

## Quality tiers

| Tier | Model | Best for |
|------|-------|----------|
| Tier | Model | Native scale | Best for |
|------|-------|--------------|----------|
| **LOW** | AnimeJaNai V3 SuperUltraCompact | **2×** | Fastest. Modern digital anime with already-clean lines. |
| **MEDIUM** | AnimeJaNai V3 UltraCompact | **2×** | Balanced. Good general default. |
| **HIGH** | Real-ESRGAN AnimeVideo v3 | **4×** | Highest quality. Older / noisier / compressed sources. Slowest. |

The scale factor comes straight from the model file's own metadata at load time — the notebook does not assume any fixed factor. If you feed a 540p source into the HIGH tier you get 2160p (4K) out; feed the same 540p source into MEDIUM and you get 1080p. If you then tick "Downscale to 1080p", the final file is resized so the **height** is 1080 — the aspect ratio of your source is preserved (a 1440×1080 4:3 source stays 4:3, a 1920×1080 16:9 source becomes 1920×1080, an ultrawide stays ultrawide). This makes the 1080p checkbox a useful "normalise the final height" knob regardless of which tier produced the intermediate frames.

---

## About Colab's free tier — read this before you start

Google Colab is generous but it isn't unlimited:

- **Session length:** Free sessions typically last up to ~12 hours, but Google can cut them earlier if idle. Don't close the browser tab mid-run.
- **GPU availability:** Free-tier GPUs (usually T4) are handed out based on demand. On busy days you may need to try again later, or upgrade to Colab Pro for guaranteed access.
- **RAM/disk:** A free-tier VM has limited disk. If your source video is very long (say, more than 20–30 minutes at HD), consider splitting it before uploading.
- **Idle disconnect:** Colab disconnects if it thinks you've walked away. Keep the tab visible while processing.

If a session dies mid-process, you just re-open the notebook and start again — nothing on your own machine is affected. Checkpoint/resume of a partially-processed clip is **not** supported in V1: a broken pipe to ffmpeg mid-stream cannot be resumed frame-exactly against a raw-video pipe without re-encoding the head of the file, and the free-tier VM is discarded on disconnect anyway. If your source is long, split it locally first.

---

## Why the notebook only downloads weights from GitHub, never from HuggingFace

The AI weights this notebook uses originate on HuggingFace, but the notebook itself **never** calls HuggingFace at runtime. It only ever downloads weights from this repo's own GitHub Releases.

A scheduled GitHub Action ([`.github/workflows/check-model-updates.yml`](.github/workflows/check-model-updates.yml)) runs weekly, checks HuggingFace for newer versions of the upstream model files, and — if there's an update — pulls them in and re-publishes them as a new Release here. If HuggingFace is unreachable, or nothing changed, the Action exits quietly and the existing Release stays put.

The upshot: if HuggingFace ever goes down, reorganizes its repos, or rate-limits users, **the Colab notebook is unaffected.** It's fully decoupled from HuggingFace at runtime.

---

## Credits

This project is a wrapper. The actual upscaling intelligence comes from other people's excellent open-source work, and this project would not exist without it.

- **[AnimeJaNai V3](https://github.com/the-database/mpv-upscale-2x_animejanai)** — by *the-database* and the AnimeJaNai contributors. The SuperUltraCompact and UltraCompact 2× models used in the LOW and MEDIUM tiers.
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** — by *Xintao Wang* et al. (Tencent ARC Lab). The `realesr-animevideov3` 4× model used in the HIGH tier.
- **[FFmpeg](https://ffmpeg.org/)** — for frame extraction, audio muxing, and final encode.
- **[PyTorch](https://pytorch.org/)** — inference runtime.

If you use MOTIONSALT Upscaler in something you publish, please credit the model authors above — they did the hard part.

---

## License

MOTIONSALT Upscaler's own code is released under the [MIT License](LICENSE).

The **model weights** mirrored via this repository's GitHub Releases remain the property of, and licensed under the terms of, their original upstream authors (see the [LICENSE](LICENSE) file's third-party section). If you are one of those authors and would prefer we not mirror your weights, open an issue and we'll take them down.

---

<sub>MOTIONSALT is an independent project. Not affiliated with Topaz Labs, Google, HuggingFace, Tencent, or any of the upstream model authors.</sub>
