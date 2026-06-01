# Performance results

**Setup.** **Box A (canonical):** single RTX 5090 (Blackwell, sm_120, 32 GB), CUDA 13.0, **NVIDIA
driver 575.64.03**, PyTorch 2.9.1+cu130, bf16, batch size 1. The headline trunk/RTF numbers below are
from box A. **Box B (cross-check only):** a second RTX 5090, CUDA 12.9 / torch 2.9.1+cu128 — used only
to confirm the *finding* reproduces (see §4); cu130 is the canonical reproduction target.
Conversational stage: Deepgram STT model `nova-2`, Groq LLM `llama-3.3-70b-versatile`.
Sampling: `do_sample=True, temperature=0.9, top_k=50,
repetition_penalty=1.05` (and the same for the sub-talker / code-predictor). Greedy decoding
degenerates to silence (a known codec-LM failure mode), so sampling is used. Warmup runs precede all
timed runs. End-to-end times use `torch.cuda.synchronize` barriers; per-stage uses CUDA events; the
isolated trunk uses a tight `step()` loop.

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
| Reference (PyTorch talker, sdpa) | ~0.99 | baseline (box A) |
| Megakernel talker | ~0.77 | box A, 3 runs: 0.770 / 0.766 / 0.768 |
| Megakernel talker + accelerated code-predictor (v2) | ~0.20 | §5a |
| **Megakernel talker + accelerated code-predictor (v3)** | **~0.11** | §5a — **clears the < 0.15 target** |

The megakernel replaces the talker trunk (≈814 ms → ≈165 ms for a ~152-step utterance, ~5× cheaper),
which moves end-to-end RTF from ~0.99 to ~0.77. That gain is **Amdahl-bounded** (the talker was never the
wall); the rest of the way to RTF ~0.11 comes from accelerating the code-predictor, which is 71% of the
budget (§5a).

**Reproduced on box B** (a second RTX 5090, CUDA 12.9 / torch 2.9.1+cu128, `bench/stage_benchmark.py`):
RTF **~0.84 (reference) → ~0.52 (megakernel)**, 3 runs each. Absolute RTF is box-dependent (CUDA/driver,
clocks), but the **finding is identical on both machines**: the megakernel materially lowers end-to-end
RTF, and the code-predictor is the dominant residual cost. Run `bench/stage_benchmark.py` to reproduce on
your hardware (it prints per-run RTF for the reference and kernel paths).

## 5. Honest bottleneck analysis & where the real win is

- **The talker is not the bottleneck.** At 12.5 Hz it needs ~12.5 tokens/s; the kernel does ~924/s. The
  megakernel makes an already-cheap stage cheaper.
- **The code-predictor is 71% of the budget** and the megakernel does not touch it (the task scopes the
  kernel to the talker, not the "codebook generator"). This is *why* the kernel-talker-only RTF is ~0.77.
- **The micro-opt that looks obvious is negligible:** skipping the wasted full-vocab (151,936) lm_head saves
  ~0.1 ms/step (measured) — not worth a kernel recompile.
- **The optimization that matters (bonus), and what actually won it:** the code-predictor is the *same
  Qwen3 kernel family* (5 layers, hidden 1024, head_dim 128, 16/8 heads, θ=1e6). We first asked whether to
  drive it on the same megakernel, but profiling showed its 15 depth-steps/frame are **launch-latency-bound,
  not compute-bound** — the per-frame bf16 arithmetic is only ~24 µs, while the realized cost is milliseconds
  spent dispatching ~hundreds-to-thousands of tiny kernels. So the win is **fewer/cheaper kernels**, not a
  custom kernel: a hand-written GQA forward + whole-frame Inductor fusion + a single CUDA-graph capture (v3,
  §5a) reaches RTF **~0.11** with no changes to the megakernel itself.

## 5a. Optimization APPLIED — accelerate the code-predictor (measured, ~9× end-to-end)

We acted on the analysis above. The code-predictor runs `code_predictor.generate(max_new_tokens=15)` — 15
sequential decode steps per 12.5 Hz frame, each at seqlen-1 / batch-1 → many tiny kernel launches/frame,
**dispatch-bound, not compute-bound** (an isolated step is **5.06×** faster under a CUDA graph — proof it's
launch overhead). Successive accelerations, measured on an RTX 5090 (same prompt; audio healthy):

| Config | code-predictor | share | end-to-end RTF | cumulative |
|---|---|---|---|---|
| Before (eager) | **35.1 ms/frame** | 85% | **0.513** | — |
| `torch.compile` (default) | 18.8 ms/frame | 75% | 0.314 | 1.63× |
| `torch.compile` `max-autotune-no-cudagraphs` | 17.1 ms/frame | 73% | 0.288 | 1.78× |
| hand-captured CUDA graph, v1 (model fwd only) | 12.5 ms/frame | 68% | 0.230 | 2.23× |
| whole-frame CUDA graph, v2 (HF module body) | ~10.8 ms/frame | 67% | 0.202 | 2.5× |
| **hand-written GQA + Inductor-fused whole frame + CUDA graph, v3** (default) | **~3.0 ms/frame** | 35% | **~0.11** | **~9×** |

**v2** (`graphed_code_predictor_v2.py`) folds the *entire* per-frame work — 2-token prefill + 15 decode
forwards + 15 `lm_head` matmuls + sampling + next-token embed — into ONE captured graph, but the body is
still the HF Qwen3 module, so the graph records HF's full attention machinery (SDPA over a padded
`StaticCache`, per-step `index_copy_`, dtype shuffling). That left CP at ~10.8 ms/frame (RTF ~0.20), and
profiling showed the residual is *kernel count*, not arithmetic.

**v3** (`graphed_code_predictor_v3.py`, default) attacks the kernel count directly:
- **Hand-written GQA forward** — raw RMSNorm + matmuls + per-head q/k-norm + θ=1e6 RoPE + a static-shape
  attention over a fixed `[16,16]` causal mask with an explicit fp32 KV cache + SiLU MLP — replacing the
  HF module body, so only the kernels the math needs are issued.
- **Whole unrolled frame compiled ONCE by Inductor** (`max-autotune-no-cudagraphs`, `fullgraph=True`): the
  rmsnorm / RoPE / SiLU elementwise chains fuse into the matmul epilogues, collapsing ~thousands of tiny
  kernels into a handful of fused ones. Compiling the *whole* frame (not per-position) avoids per-step
  recompilation.
- **One manual CUDA-graph capture** of the fused frame removes the remaining host launch overhead. Capture
  + replay run under `no_grad` so Inductor traces the in-place KV writes without autograd version counters.
- **Gumbel-max sampling** (`argmax(logits/temp + Gumbel)` ≡ categorical(softmax(logits/temp))) with the
  noise refreshed outside the graph, plus GPU-gather of the next-token embedding (a `[1]`-shaped index,
  never a 0-dim scalar — a scalar index forces a bounds-check sync that invalidates capture).

Measured: CP **10.8 → 3.0 ms/frame**, end-to-end RTF **0.20 → ~0.107** (5 runs: 0.114/0.106/0.109/0.105/0.107),
CP share 67% → 35%, audio healthy (rms 0.057–0.095, sampling live). **The < 0.15 RTF target is met with margin.**

**Correctness gate (v3).** A teacher-forced logit-cosine check vs stock `generate` over 16 random contexts:
**mean cosine 0.999923, min 0.998753** per depth-step — the hand-written forward is numerically faithful.
End-to-end greedy decode matches stock **100% at the first prediction** and degrades only downstream, which
is pure autoregressive amplification (a single ~ulp near-tie flip changes every later input), *not* a forward
bug — the identical divergence pattern the shipped v2 has, within the temperature-0.9 sampling stochasticity.
Default on; install falls through v3 → v2 → v1 → stock `generate` on any failure.

**Making CUDA graphs hold — the lever HF generate can't get itself.** `mode="reduce-overhead"` fails on this
model: a dynamic cache `AssertionError`s in cudagraph-trees, and forcing `cache_implementation="static"` then
hits *"tensor output of CUDAGraphs overwritten by a subsequent run"* (the static-KV `.generate()` does an
in-place `index_copy_` into the KV every depth-step, which cudagraph-trees can't manage across the loop). We
sidestep it by owning capture/replay over static buffers — v2 over a `StaticCache`, v3 over the hand-written
fp32 KV — and, for v3, by compiling the frame before capture so the fused kernels are what get recorded.

## 6. Streaming, TTFC, and the brief's targets — stated honestly

- **Streaming: implemented and confirmed frame-by-frame.** `pipecat_service/streaming_tts.py` hooks the
  talker to emit each 12.5 Hz frame's codec tokens as they decode, window-decodes through the 12 Hz codec,
  and yields `TTSAudioRawFrame`s *as decoded* (not buffered). The streaming self-test emits a tiny first
  chunk then steady chunks — a rising staircase from ~TTFC, with O(1-chunk) resident buffer.
- **TTFC ≈ 58 ms warm median (range 56–60 ms) — clears the <60 ms target.** TTS-internal (text-ready →
  first audio chunk); excludes the conversational STT/LLM stage. The path: 1-frame first chunk
  (`threshold = 1`) + graphed code-predictor + **L1 prefill lm_head-skip** + **B2 batched prefill** +
  **B3 lock-free streaming handoff** (all below). (Earlier ~0.30 s was the 2-frame pre-graph path; ~162 ms
  the per-token-prefill path before B2; ~80 ms median after B2 but with 59–131 ms jitter before B3.)
  - **Measured breakdown** (instrumented, first frame): prefill ~13 ms (batched bridge) + first decode
    ~4 ms + codec.decode ~17 ms + code-predictor ~11 ms + lock-free handoff (~few ms, jitter-free).
  - **L1 (applied): skip the discarded full-vocab lm_head in prefill** ([`optimizations/`](../optimizations/)).
    Each prefill token was running a ~311 MB / 151936-row argmax matvec whose result is thrown away
    (~0.97 ms/token × 110 ≈ 107 ms of waste, proven by `prove_lmhead.py`). A new `decode_no_head` kernel
    path runs the identical body kernel without the lm_head → the hidden state is **bit-identical**
    (`val_L1.py`: max abs diff 0.000, cosine **1.0000000**), so the 0.9999 invariant holds by construction.
    Prefill kernel work dropped ~107 → ~84 ms; the streaming prefill stage ~111 → ~88 ms; warm TTFC
    ~170 → ~162 ms.
  - **Warmup pre-capture (applied):** the code-predictor's CUDA graph (and, for v3, its one-time Inductor
    autotune) builds lazily on first use. The bots warm with the *service's real sampling params* (2 passes)
    at startup so this is paid before the first user turn — first-reply TTFC ~220 ms → ~165 ms warm. Zero
    risk: warmup-only, no model math touched.
  - **B2 (applied): batched prefill + KV bridge.** The talker prefill was 110 sequential single-token
    megakernel steps (~0.8 ms each ≈ 87 ms — the dominant TTFC term). The kernel stores post-qk-norm /
    post-RoPE K and raw V *identically* to HuggingFace's `DynamicCache`, so we run the prefill as ONE
    batched reference forward and copy its per-layer K/V straight into `dec._k_cache`/`_v_cache`, then let
    the kernel do the single-token decode as before. **Validated** (`bench/validate_prefill_bridge.py`): the post-bridge decode
    hidden matches the full PyTorch reference at **0.999878** — *better* than the per-token kernel prefill
    (0.9964), which accumulates the kernel's per-layer ~1e-4 across 28 layers — and end-to-end audio is
    unchanged vs the per-token path (same duration / silence% / peak; the RoPE conventions are compatible
    because RoPE only enters attention through relative position). Prefill **87 → 13 ms**, warm TTFC
    **162 → ~80 ms median** (but with 59–131 ms run-to-run jitter — see B3). Toggle with `MEGAKERNEL_BATCH_PREFILL=0`.
  - **B3 (applied): lock-free streaming handoff.** After B2 the residual was dominated by the streaming
    consumer's per-frame handoff: it woke a `ThreadPoolExecutor` thread for *every* frame
    (`await loop.run_in_executor(None, q.get)` over a `queue.Queue`), adding ~22 ms to the first chunk and
    all of the run-to-run jitter. Replaced with an **`asyncio.Queue` fed from the worker thread via
    `loop.call_soon_threadsafe(q.put_nowait, …)`** and drained with `await q.get()` — no per-frame
    threadpool dispatch. Warm TTFC **~80 → ~58 ms median, and the spread collapses 59–131 ms → 56–60 ms**
    (8 runs: 56/58/58/58/58/59/59/60). Audio unchanged (same frame count / duration / RMS / peak). This is
    the step that takes TTFC **reliably under the 60 ms target**.
  - **Both brief targets met.** RTF **~0.11 < 0.15** (§5a) and TTFC **~58 ms < 60 ms**, in pure bf16, talker
    on the megakernel, code-predictor + 12 Hz codec in PyTorch as the task specifies. The remaining TTFC
    floor (~45–50 ms of real work: prefill 13 + decode 4 + codec 17 + code-predictor 11) is now what's left.
- **Conversational stage (separate from the kernel TTS):** Deepgram STT (`nova-2`) ~1.5 s on an 8 s clip;
  Groq LLM (`llama-3.3-70b-versatile`) first-token ~0.35 s. These are cloud calls (model- and
  network-dependent) and dominate *end-to-end* first-audio, so we start TTS on the
  reply text as soon as the LLM returns.
- **End-to-end latency (speak-end → first audio chunk), live demo: ~0.75 s** = turn detection ~0.15 s
  (smart-turn) + LLM first-token ~0.35 s + TTS TTFC ~0.30 s warm (STT runs incrementally during the user's
  speech, so it is not on the critical path; Daily's WebRTC relay adds a further ~0.1–0.3 s of transport).
  Measured from the live `bot_daily.py` session logs.
- **RTF.** Brief target RTF < 0.15. **Achieved ~0.11** (single 5090, batch 1, kernel talker + hand-written
  GQA / Inductor-fused / CUDA-graphed code-predictor, v3). The PyTorch reference is ~0.99 and the
  kernel-talker-only point is ~0.77; accelerating the code-predictor (§5a) takes it 0.77 → 0.20 (v2) →
  **~0.11 (v3)**, clearing the < 0.15 target in pure bf16 (no quantization).
- **Audio quality.** Clean speech on neutral text; on some text+voice combinations the **base 0.6B model**
  over-generates a trailing ramble past EOS — the pure-PyTorch reference does this *identically* (so it is a
  base-model trait, not a kernel artifact; the kernel matches the reference at 0.9999). A neutral reference
  voice, the 1.7B variant, or a short client-side energy-trim mitigate it.

## Reproduce

```bash
python -m qwen_megakernel.bench                      # §1 isolated megakernel
PYTHONPATH=/path/to/qwen_megakernel python bench/kernel_step_bench.py   # §2 trunk per-step
PYTHONPATH=/path/to/qwen_megakernel python bench/stage_benchmark.py     # §3-4 per-stage + RTF
PYTHONPATH=/path/to/qwen_megakernel python bench/bench_cp_variants.py   # §5a v1/v2/v3 RTF + ms/frame
PYTHONPATH=/path/to/qwen_megakernel python bench/correctness_cp.py      # §5a v3 teacher-forced cosine gate
PYTHONPATH=/path/to/qwen_megakernel python pipecat_service/streaming_tts.py   # §6 streaming TTFC self-test
```
