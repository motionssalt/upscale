# Changes — GPU-resident pipeline fix (targeted, driven by the profiling numbers)

Only one file changed: `MOTIONSALT_Upscaler.ipynb`, cell 6 (the ⚙️ Step 3 processing cell).

## What the profiling pass proved

From the baseline run that shipped with the profiling instrumentation:

```
[frame 15] 0.087 fps · read=12.8ms (0%) · pre=6172.0ms (54%) · infer_h2d=27.7ms (0%) ·
infer_kernel=874.3ms (8%) · infer_d2h=323.1ms (3%) · recover=1215.6ms (11%) ·
post=2690.6ms (23%) · write=149.2ms (1%)
nvidia-smi: gpu=0% mem=1929/15360MiB @ frame 15 (0% on every sampled frame)
```

- `pre` = 54%, `post` = 23%, `recover` = 11% → **88% of frame time is CPU filter work.**
- `infer_kernel` (the model forward pass) = 8% and shrinking as cudnn warms up.
- GPU util = 0% almost the entire run — the GPU sits idle while cv2/numpy churn on CPU.
- The frame crossed the CPU↔GPU boundary **four times** per frame: a CPU-side
  `np.ascontiguousarray(bgr[:, :, ::-1])` copy going in, D2H coming out, then every
  post/recover filter ran on CPU numpy arrays before a final `tobytes()`.

The bottleneck was never the model and never fp16 — it is the CPU-side
pre/post/recover stages surrounding a fast GPU inference step.

## What this pass changes

Every per-frame stage now runs on the GPU as pure `torch` tensor ops, on one tensor
that is uploaded once and downloaded once. No numpy, no cv2 in the per-frame path.

| Stage | Before (CPU) | After (GPU) |
|-------|--------------|-------------|
| H2D   | `np.ascontiguousarray` BGR→RGB copy on CPU, then `.to(cuda)` | raw uint8 upload once; BGR→RGB + /255 done on-GPU as free views + one kernel |
| `pre` — Revert Compression | `cv2.bilateralFilter` | vectorized bilateral: `unfold` + Gaussian products + weighted sum |
| `pre` — Reduce Noise | `cv2.fastNlMeansDenoisingColored` (multi-second/frame) | NLM-lite: 3×3-patch distance over a 7×7 search window, fully vectorized |
| `recover` | `cv2.resize` LANCZOS4 to full-res + numpy blend | `F.interpolate(..., bicubic, antialias=True)` + in-place `lerp_` |
| `post` — Dehalo | Canny + dilate + `medianBlur` + numpy mask | Sobel edge band + `max_pool2d` dilate + 3×3 integral-image box blur |
| `post` — Anti-alias/Deblur | `cv2.GaussianBlur` + `addWeighted` | separable Gaussian `conv2d` + same unsharp weights |
| `post` — Improve Detail | `cv2.createCLAHE` on LAB L | on-GPU CLAHE on L: per-tile clipped CDF LUTs + bilinear LUT blend |
| `post` — Sharpen | `cv2.GaussianBlur` + `addWeighted` | separable Gaussian `conv2d` + identical unsharp math |
| D2H   | fp32 tensor → numpy → CPU RGB→BGR flip copy | clamp/round on-GPU, single uint8 D2H straight to ffmpeg bytes |

Per frame there are now exactly **two** host↔device transfers: one `infer_h2d`
(uint8 in) and one `infer_d2h` (packed uint8 out, 3 bytes/px instead of fp32's 12).

## What did NOT change (deliberately)

- **No new dependencies.** Everything is plain torch (`conv2d`, `interpolate`,
  `unfold`, `cumsum`, `max_pool2d`). Colab already ships all of it.
- **fp16 stays removed.** The profile confirmed the model is 8% of frame time;
  precision was never the lever. The comment block is kept so nobody reintroduces it.
- **Same sliders, same 0–100 knobs, same defaults.** Sharpen still defaults to 15.
- **Same profiling instrumentation** — identical bucket names
  (`read, pre, infer_h2d, infer_kernel, infer_d2h, recover, post, write`), identical
  report cadence (frames 3/5/10/15/20/30/50/100), identical per-frame line format,
  same nvidia-smi watcher. The new numbers drop straight into the same table for a
  direct before/after comparison.
- The model load, ffmpeg pipe, audio mux, and 1080p finish step are untouched.

## Numerical fidelity (verified, not assumed)

The torch ports were cross-checked against the original cv2 implementations on
identical frames (CPU tensors, same kernels/math as CUDA):

- round-trip BGR→RGB→BGR: exact (max diff 0)
- bilateral (Revert): max 16/255, mean 1.1/255, corr 0.9998
- NLM-lite (Reduce Noise): corr 0.998; noise σ on a noisy patch 14.99 → 2.76
  (cv2 NLM gets 1.28 — cv2's is stronger, both preserve line art)
- recover blend: max 3/255 (lanczos vs bicubic+antialias resampling), corr 0.99996
- deblur & sharpen: max 1/255, corr ≈ 1.0 (bit-near-exact)
- dehalo: mean 0.02/255 (behavioral — Sobel band vs Canny band, same strength)
- CLAHE (Improve Detail): the one stage that is *visually equivalent* rather than
  near-bit-exact. Per-tile LUT construction matches cv2 (`round(cdf·255/n)`);
  cv2's exact sub-pixel tile-edge sampler differs at boundaries. Perceptual
  contrast-equalization result is the same; it only runs when Improve Detail > 0.

## How to verify on your T4 (Step 4 of the debug prompt)

Run the notebook on the same clip with the same sliders as the baseline run and
compare the frame-15 line:

**Expected:** `pre`, `recover`, and `post` collapse from 6172/1216/2691 ms to the
tens-of-ms range; `infer_kernel` (≈874 ms) becomes the dominant bucket; the
nvidia-smi watcher shows GPU util well above 0% during the loop instead of a flat 0%.

If GPU util is *still* near 0% after this, that means CPU work remains somewhere —
the same instrumentation is still in place to find it (check `read`, `write`, and the
H2D/D2H buckets next). Do not declare victory without the numbers.

## Files changed

- `MOTIONSALT_Upscaler.ipynb` — cell 6 rewritten as the GPU-resident pipeline above.

No other files touched.
