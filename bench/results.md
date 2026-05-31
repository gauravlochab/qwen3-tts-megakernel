# Performance results

**Setup.** Single RTX 5090 (Blackwell, sm_120), CUDA 13.0, PyTorch 2.9.1+cu130, bf16, batch size 1.
Sampling: `do_sample=True, temperature=0.9, top_k=50, repetition_penalty=1.05` (and the same for the
sub-talker / code-predictor). Greedy decoding degenerates to silence (a known codec-LM failure mode),
so sampling is used. Warmup runs precede all timed runs. End-to-end times use `torch.cuda.synchronize`
barriers; per-stage uses CUDA events; the isolated trunk uses a tight `step()` loop.

## 1. Megakernel decode, isolated (Qwen3-0.6B)
`python -m qwen_megakernel.bench` → **1029 tok/s, 0.97 ms/step** (matches the repo's 1036). This is the
on-device `generate_nosync` path (no per-step host sync).

## 2. Megakernel as the talker trunk — our integration path (isolated, 300 steps)
**1.082 ms/step (924 steps/s)**, including the per-step `.item()` host sync and the (currently unused)
full-vocab lm_head. Only ~0.11 ms/step above the nosync baseline → the integration overhead is tiny.

## 3. Per-stage breakdown (reference PyTorch pipeline, one utterance ≈ 42 frames / 3.36 s audio)

| Stage | Time | Share |
|---|---:|---:|
| Talker trunk (28-layer Qwen3) | 814 ms | 24% |
| **Code-predictor (5-layer, 15 groups/frame)** | **2377 ms** | **71%** |
| Codec + misc | 157 ms | 5% |
| **Total** | **3348 ms** | RTF 0.99 |

(CUDA-event attributed; the trunk number includes the 110-token prefill done as one batched forward.)

## 4. End-to-end RTF (batch 1, single 5090, non-streaming)

| Pipeline | RTF | Notes |
|---|---:|---|
| Reference (PyTorch talker, sdpa) | ~0.99 | baseline |
| **Megakernel talker** | **~0.77** | 3 runs: 0.770 / 0.766 / 0.768 |

The megakernel replaces the talker trunk (≈814 ms → ≈165 ms for a ~152-step utterance, ~5× cheaper),
which moves end-to-end RTF from ~0.99 to ~0.77. The gain is **Amdahl-bounded**: the talker was never the
wall.

## 5. Honest bottleneck analysis & where the real win is

- **The talker is not the bottleneck.** At 12.5 Hz it needs ~12.5 tokens/s; the kernel does ~924/s. The
  megakernel makes an already-cheap stage cheaper.
- **The code-predictor is 71% of the budget** and the megakernel does not touch it (the task scopes the
  kernel to the talker, not the "codebook generator"). This is *why* end-to-end RTF only improves modestly.
- **The micro-opt that looks obvious is negligible:** skipping the wasted full-vocab (151,936) lm_head saves
  ~0.1 ms/step (measured) — not worth a kernel recompile.
- **The optimization that matters (bonus):** the code-predictor is the *same Qwen3 kernel family* (5 layers,
  hidden 1024, head_dim 128, 16/8 heads, θ=1e6). Driving it on the same megakernel, or batching/CUDA-graphing
  its 15 per-frame depth-steps, attacks the real 71%. This is the highest-leverage next step.

## 6. TTFC and the brief's targets — stated honestly

- **TTFC** is not yet meaningful: the current pipeline is **non-streaming** (full utterance → codec). Streaming
  via the 12 Hz codec's `chunked_decode(left_context=25)` is the next step; TTFC will then be dominated by
  prefill (~110 talker steps ≈ 120 ms) + first-frame code-predictor + codec.
- **RTF.** Brief target RTF < 0.15. Achieved ~0.77 (single 5090, batch 1, non-streaming, unoptimized
  code-predictor, no `torch.compile`/CUDA-graphs). This is consistent with the unoptimized reference
  (the fully-optimized configs add flash-attn, compile, and CUDA graphs). Reaching <0.15
  requires optimizing the code-predictor + codec + compile/graphs — **future work**, reported transparently
  rather than hand-waved.

## Reproduce

```bash
python -m qwen_megakernel.bench                      # §1 isolated megakernel
PYTHONPATH=/path/to/qwen_megakernel python bench/kernel_step_bench.py   # §2 trunk per-step
PYTHONPATH=/path/to/qwen_megakernel python bench/stage_benchmark.py     # §3-4 per-stage + RTF
```
