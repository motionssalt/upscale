# Changes — profiling pass (no speculative fixes)

Only one file changed: `MOTIONSALT_Upscaler.ipynb`, cell 6 (the ⚙️ Step 3 processing cell).

## Why this pass ships instrumentation, not a fix

The previous pass applied generic optimizations without evidence they targeted the real bottleneck, and speed did not improve. Per the debug prompt, this pass:

1. Adds real per-stage timing (Step 1) — with the model bucket split into `infer_h2d` / `infer_kernel` / `infer_d2h` so CPU-side stalls are visible separately from real GPU kernel time. `infer_kernel` uses `torch.cuda.Event` with a `cuda.synchronize()`, which is the only honest way to measure GPU wall-time.
2. Confirms the basics (Step 2) — logs `torch.cuda.is_available()`, the actual `.device` and `.dtype` of the first model parameter (ground truth, not what we *think* `.cuda()` did), and runs a background `nvidia-smi` watcher that samples GPU util + memory every 3 s during the loop and logs a snapshot every 6 s. Prints min/max/avg GPU util across the run in the final summary.
3. Does NOT apply Step 3 yet. Speculative fixes are what got us here. Once the profile identifies the actual slow stage on your Tesla T4 + your test clip, the next pass applies a targeted fix.
4. Removes the fp16 toggle as requested. Everything runs fp32. Comment left in place so nobody quietly reintroduces it.

## What you'll see in the log

Reports arrive at frames 3, 5, 10, 15, 20, 30, 50, 100 (so a 0.2 fps run surfaces a breakdown after ~15 s, not ~50 s). Each line looks like:

```
📊 [frame 5]  0.21 fps · read=12.3ms (0%) · pre=0.1ms (0%) · infer_h2d=42.1ms (1%) · infer_kernel=180.4ms (4%) · infer_d2h=38.7ms (1%) · recover=0.0ms (0%) · post=0.0ms (0%) · write=4460.0ms (94%)
📊 nvidia-smi: gpu=3% mem=2100/15109MiB @ frame 5
```

Read the dominant bucket → that IS the bottleneck. No guessing.

## How to interpret the numbers

- `infer_kernel` dominates AND GPU util ~100% → model itself is the bottleneck. fp16 / torch.compile / TensorRT become real levers.
- `infer_kernel` small AND GPU util low → I/O or Python-side stall. Look at whichever of `read` / `write` / `infer_h2d` / `infer_d2h` is largest.
- `write` dominates → ffmpeg pipe / x264 encoder is the wall (very likely on HIGH tier at 4K with `preset medium crf 16` on Colab's shared host CPU).
- `read` dominates → OpenCV CPU decode of the source.
- `infer_h2d` + `infer_d2h` dominate → memcpy overhead. `torch.from_numpy(...).to("cuda", non_blocking=True)` is currently a no-op (unpinned host memory) — flagged in a code comment, not fixed yet.

## Step 4 (prove it worked) — deferred, honestly

I cannot report before/after fps because I have no T4 and no test clip. Paste the profiling output from your next run and the fix pass follows from that, with real numbers.

## Files changed

- `MOTIONSALT_Upscaler.ipynb` — cell 6 rewritten with the instrumentation described above; `cb_fp16` widget removed from the UI; `_load_model` no longer takes a `want_fp16` argument; `_infer_tensor` now updates the profile buckets directly.

No other files touched.
