# Xyether Colab Notebook — Unlock & Audit Forensic Notes

> **Status: stopped at the engine's remote auth gate, by design.** These notes are the
> complete record of a one-time audit of the *second* Colab notebook
> (`1_PuF9vvUuqDbfR_frjnBpwpvCiCxUUrU`, the "Xyether" password/OTP build). The notebook
> was **not** run end-to-end, and the locked content was **not** pushed here — for
> concrete, documented reasons below. This file is the only record of what was found,
> what was decided, and why.

Date of audit: 2026-08-19 · Auditor: automated assistant (Genspark) · Target notebook: https://colab.research.google.com/drive/1_PuF9vvUuqDbfR_frjnBpwpvCiCxUUrU

---

## 1. What the notebook actually is (after de-obfuscation)

The notebook has two cells:

### Cell 1 — "System Setup & Authentication" (the lock)
What you see in Colab is a single ROT13-obfuscated blob. Decoded, it is ordinary
Python that does the following, in order:

1. Clears a Google-Drive cache folder (`/content/drive/MyDrive/Xyether_Cache`).
2. If a local `build_xyether_upscaler_cc.py` exists, it just runs that and stops.
3. Otherwise it `pip install`s: `pymongo`, `dnspython`, `gdown`, `pycuda`,
   `tensorrt==11.0.0.114`, `opencv-python`, `pillow`, **`pepedpid`** (see §4).
4. Downloads a compiled binary, `config_utils.so`, from a **public Google Drive file**
   (`id=1GRXulTmvZckA81wnwyGCWlaoPi3pbWTU`), loads it into the runtime, then **deletes
   the file** (`os.remove`) so the user can't inspect it afterwards.
5. Calls `config_utils.authenticate_telegram(Auth_Token)` — this is the OTP/password gate.

### Cell 2 — "Universal Xyether Upscale & CC Engine" (the payload UI)
A plain Colab form: target path, model choice (10 "Xyether" 1x/2x anime & color-correction
models), speed/force-1080p toggles, nvenc codec + CRF slider, auto-download. It calls
`config_utils.run_processing(...)` — i.e. **all real logic lives inside the compiled
`.so`, not in the notebook.**

---

## 2. The critical correction to the original assumption

The task brief assumed: *"enter the OTP → one-time setup → everything needed to run it
exists locally afterwards, no ongoing server-side dependency."*

**That is not how this build works.** String analysis of `config_utils.so` (451,920 bytes,
sha256 `10e91515…c4817`, stripped ELF, Cython-compiled) shows:

- `Requesting model stream from server (...)`
- `Model stream received into RAM memory (...)`
- `Verifying Auth Token with API Server...` / `Unauthorized session token.` /
  `Invalid Auth_Token.` / `verify_token_strict` / `_HMAC_SECRET` / `_XOR_KEY`
- `run_upscale_pipeline` (the entry point Cell 2 calls)

In plain terms: **the actual model weights are never stored locally and are not in the
notebook or the `.so`.** On every run, the `.so` phones an API server, re-validates the
token, and *streams the model into RAM at runtime*. The OTP doesn't "unlock" a self-contained
tool — it buys a short window of access to a server the notebook author controls. The
notebook is a thin client; the product lives on someone else's server.

**Consequence:** there is nothing meaningful to "extract so it runs independently." The only
local artifact is the 442 KB compiled shim, which is useless (and non-redistributable) on its own.

---

## 3. Why nothing from this notebook was pushed

| Candidate | Decision | Reason |
|---|---|---|
| Notebook `.ipynb` as-is | **Not pushed** | Ships an obfuscated credential gate + hardcoded third-party Drive/binary links; redistributing it redistributes *their* auth system, not your code. |
| De-obfuscated Cell 1 source | **Not pushed** | Same content, just readable — still their auth bootstrap. Kept locally at `decoded_setup_cell.py` for your reference only. |
| `config_utils.so` binary | **Not pushed** | Compiled, stripped, license unknown, contains their auth/server logic (`_HMAC_SECRET`, `_XOR_KEY`, token verification). Not yours to redistribute, and pushing a stripped binary to a public repo is exactly the "anything not mine to redistribute" the brief says to scrub. |
| The 10 "Xyether" model weights | **Cannot be pushed** | They are **not in the notebook at all** — they're streamed from the API server at runtime. There is no file to extract. |
| Forensic notes (this file) | **Pushed** | The only artifact that is both yours (the audit) and safe to publish. |

If the goal is "a working anime/CC upscaler in a repo I control," the honest path is to
rebuild the pipeline on top of **openly licensed models** (the existing MOTIONSALT project
in this repo already does exactly that with AnimeJaNai V3 / Real-ESRGAN), not to clone a
gated binary that depends on a server you don't own.

---

## 4. Security findings you should know about

1. **`pepedpid` — a probable supply-chain risk.** This is a real but obscure PyPI package
   (v0.1.2, abi3 compiled wheels, no author, no homepage, no description, no source repo).
   It is *not* an upscaling dependency. A nameless compiled wheel installed alongside an
   auth flow is a classic persistence/side-load pattern. **Recommendation:** treat any
   runtime where you previously ran this notebook as potentially exposed; if you ever typed
   real credentials there, rotate them. (Package inspected statically only — never executed.)
2. **The `.so` self-deletes after loading** (`os.remove(dest_so)`), which is anti-inspection
   behavior, not something a legitimate setup script needs.
3. **The engine phones home twice**: once to verify the token against a MongoDB-backed API
   (hence `pymongo`/`dnspython`), and again to stream the model. The "3-minute OTP" gates a
   *server-side session*, not a local unlock.
4. **The repo's own history is clean.** A full secret-pattern scan of `git log -p` across all
   branches (PATs, `ghp_`, `mongodb://`, Telegram bot tokens, GCP `AIza` keys, HF tokens,
   private keys) returned **no matches**. The many `bot: upload file via secrets pusher`
   commits turned out to be the weights-mirror workflow, not leaked secrets.
5. The temporary PAT used for this push was scoped to this repo only, was used solely for the
   `git push`, and is recorded nowhere in the pushed files.

---

## 5. What was done, step by step

1. Pulled the notebook source directly from Google Drive (public "anyone with link" export) —
   no OTP needed to *read* the file; the OTP only gates the *runtime engine*.
2. De-obfuscated Cell 1 (ROT13) and mapped the full setup/auth flow.
3. Downloaded and statically analyzed `config_utils.so` (strings-only; never executed).
4. Audited the existing `motionssalt/upscale` repo (README, weights-mirror script, GitHub
   Action, CHANGES.md) and its full git history for secrets — clean.
5. Wrote these notes; committed and pushed **only this file** to `main`.

## 6. What was deliberately *not* done
- Did not execute `config_utils.so` or `authenticate_telegram` (would have required a live OTP
  and would have run unaudited compiled code that phones home).
- Did not push the notebook, its decoded source, or the binary.
- Did not attempt to bypass the server-side model stream (there is nothing local to bypass *to*).

*If you want, the next step is a clean-room re-implementation of the Xyether pipeline using the
open models already mirrored in this repo — say the word and that can be scaffolded here.*
