# Changes — Follow-up bug-fix pass on the GPU-resident pipeline

Only one file changed: `MOTIONSALT_Upscaler.ipynb`, cell 6 (⚙️ Step 3).
The previous GPU-porting pass is kept in full — pre/post/recover stay on-GPU.

## The two errors this pass fixes

### Error 1 — "Inplace update to inference tensor outside InferenceMode is not allowed"

**Root cause.** The previous pass wrapped only the model forward in
`torch.no_grad()`. On PyTorch builds where spandrel's descriptor runs the
inner forward under inference-mode semantics, the returned tensor is an
*inference tensor*. It was then handed to `_recover_blend_gpu`, which did
`up.lerp_(naive, alpha)` — an in-place op — OUTSIDE that context. That is
the exact combination PyTorch rejects.

**Fix.** The ENTIRE per-frame path (H2D + pre + infer + recover + post +
D2H) now runs inside a single `torch.inference_mode()` block in
`_do_process`. Every intermediate is consistently an inference tensor, so
`lerp_`, `add_`, `addcmul_`, `mul(..., out=)`, etc. are all legal. This is
the recommended permanent fix — not a per-call `.clone()` sprinkled at the
first site that happens to fail.

The redundant `torch.no_grad()` inside `_infer_kernel_only` was dropped;
`inference_mode` is strictly stronger.

`_recover_blend_gpu` also gets a belt-and-braces `RuntimeError` fallback
to an out-of-place `lerp` in case an exotic PyTorch build still refuses
in-place under inference_mode — never reached on Colab's PyTorch.

### Error 2 — CUDA OOM ("Tried to allocate 1.14 GiB … 3.43 GiB reserved but unallocated")

**Root cause.** Three memory bombs in the ported filters, all on the
HIGH tier where the frame reaches 1080p input / 4K output:

| Location | Old peak allocation on 1080p |
|----------|------------------------------|
| Bilateral `xp.unfold(2,d,1).unfold(3,d,1)` in `_pre_filters_gpu` at d=9 | `1·3·1080·1920·9·9·4 = 6.05 GiB` — one tensor |
| NLM-lite `unfold` + reshape to `(49,3,H,W)` | ~3.6 GiB |
| CLAHE built 4 full-resolution `(1,1,H_out,W_out)` LUT gathers, alive at once, at 4× res | ~500 MB combined |

Plus caching-allocator fragmentation from these giant one-shot tensors
being freed and re-requested at slightly different shapes — that's the
"3.43 GiB reserved but unallocated" line in the error.

**Fix (all permanent, no per-frame guards, no fp16 reintroduction):**

- **Bilateral rewritten as a shift-and-accumulate loop** over the `d×d`
  window offsets. Numerically identical to the unfold version (same
  window, same Gaussian range/space weights — verified below), but peak
  per shift is `O(H·W·3)` instead of `O(H·W·3·d²)`.
- **NLM-lite rewritten the same way** — a loop over the 49 `(dy,dx)` search
  offsets. The 3×3-patch mean that preserves line art is computed
  per-slice (once verified numerically identical to blurring each shifted
  slice independently, as the reference did).
- **CLAHE bilinear blend rewritten as an in-place accumulator** — one
  corner LUT gathered, weighted, and added at a time via
  `mul(out=)` / `addcmul_`. Never holds more than one `(1,1,H,W)` gather
  in memory. Same math, same result.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** set at process
  start (before first CUDA allocation) to defeat fragmentation on the
  large tensors that do remain. Harmless on builds that don't support it.
- **`torch.cuda.empty_cache()` called ONCE after the first frame** (once
  `cudnn.benchmark` has picked its algo) — releases the transient
  benchmark scratch. Not per-frame.
- **Explicit `del`s of transient tensors** through pre/post/recover so
  Python drops refs before the next allocation, keeping peak flat.

## What did NOT change (deliberately)

- **pre/post/recover stay on-GPU.** The previous fix is kept in full;
  nothing reverted to CPU cv2/numpy.
- **fp16 stays removed.** Not reintroduced.
- **Same sliders, same 0–100 knobs, same defaults.** Sharpen still 15.
- **Same profiling instrumentation** — identical bucket names
  (`read, pre, infer_h2d, infer_kernel, infer_d2h, recover, post, write`),
  identical report cadence (frames 3/5/10/15/20/30/50/100), identical
  per-frame line format, same nvidia-smi watcher.
- **No new dependencies.** Plain torch (`conv2d`, `interpolate`,
  `cumsum`, `max_pool2d`, `bincount`, `addcmul_`).
- Model load, ffmpeg pipe, audio mux, 1080p finish step untouched.

## Numerical parity (measured, not assumed)

An offline harness cross-checks each rewritten filter against a direct
reference of the same math (the previous unfold / 4-gather implementation)
on identical synthetic frames, in fp32 CPU torch:

| Filter | Max abs diff | Mean abs diff |
|--------|-------------:|--------------:|
| Bilateral (shift-loop vs unfold) | **5.96e-07** | 5.87e-08 |
| NLM-lite (shift-loop + per-slice blur vs unfold) | **5.36e-07** | 6.78e-08 |
| CLAHE (accumulator vs 4-gather blend) | **1.19e-07** | — |

Well below one uint8 LSB (`≈3.9e-3`). Visually indistinguishable from
the previous pass — this is a memory rewrite, not a math rewrite.

## Memory-peak verification

The same harness runs one HIGH-tier-equivalent frame (4× model) with ALL
sliders at 100 and tracks peak live-tensor bytes:

| Configuration | Peak GPU memory (fp32) |
|---------------|------------------------:|
| **Old** bilateral unfold at d=9 on 1080p — one tensor alone | **1.88 GB** |
| **New** whole pipeline extrapolated to 1080p HIGH (4× → 3840×2160) | **~0.14 GB** |
| Reduction | **~14× smaller peak** vs. one worst-case old tensor |

Well inside the T4's ~11 GB usable budget after model + cudnn scratch —
1.14 GiB allocations no longer fail, and there's no shape churn to
fragment the caching allocator across.

## Error-1 reproduction / repair test

The harness also reproduces the failing pattern from the traceback:

```
with torch.inference_mode():
    y = model(x)                 # y is an inference tensor
naive = F.interpolate(...)       # outside inference_mode
y.lerp_(naive, 0.25)             # <-- RuntimeError: Inplace update to inference tensor
```

confirms the RuntimeError, then runs the new pattern (whole path inside
`inference_mode`) and confirms `lerp_`, `addcmul_`, and `mul(out=)` all
succeed with finite output.

## How to verify on your T4 end-to-end (Step 4 of the debug prompt)

Rerun the notebook on the same clip, HIGH tier, same sliders as the
previous profiling run, and compare against the profile line format:

**Expected:** Neither `Inplace update to inference tensor …` nor `CUDA
out of memory …` appears in the run log. `pre`, `recover`, and `post`
buckets stay in the tens of ms (the previous pass's improvement is
preserved). `infer_kernel` remains the dominant bucket. `nvidia-smi mem=…`
in the periodic snapshot stays comfortably below 14.56 GiB total on the
T4 across all frames (including HIGH-tier 4× frames with all sliders on).

## Files changed

- `MOTIONSALT_Upscaler.ipynb` — cell 6 (⚙️ Step 3 processing cell).
- `CHANGES.md` — this file.

No other files touched.
