# qwen3-tts-megakernel

Run AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a single-launch CUDA decode kernel for Qwen3-0.6B, ~1000 tok/s on an RTX 5090) as the **decode backend for the Qwen3-TTS *talker***, and stream the synthesized speech into a [Pipecat](https://docs.pipecat.ai) voice pipeline (STT → LLM → TTS → audio).

> **Status — end-to-end working on an RTX 5090.** The megakernel drives the Qwen3-TTS talker (hidden states match the reference at **0.9999** cosine), audio streams **frame-by-frame** into a Pipecat pipeline (Deepgram STT → Groq LLM → megakernel TTS → audio over WebRTC), and a live browser↔GPU voice demo runs end-to-end. Benchmarks measured; honest gap analysis included.

**Headline numbers** (RTX 5090, bf16, batch 1 — full methodology in [`bench/results.md`](bench/results.md)):

- **RTF ~0.11** — clears the **< 0.15** target  (0.99 PyTorch ref → 0.77 kernel talker → 0.11 after accelerating the code-predictor)
- **TTFC ~58 ms** warm median — clears the **< 60 ms** target  (162 → 58 ms)
- **Decode 1029 tok/s** isolated · 924 tok/s as the talker trunk (~5× cheaper/step than PyTorch)
- **0.9999** talker hidden-state cosine vs the reference

**▶ Demo recording:** [`recording/demo_voice_agent.mov`](recording/demo_voice_agent.mov) (4.9 MB) — live browser ↔ RTX 5090 voice loop, you talking end-to-end.
**Docs:** [`DEMO.md`](DEMO.md) (how to run / see the demo) · [`SETUP.md`](SETUP.md) (reproducible fresh-box setup) · [`bench/results.md`](bench/results.md) (numbers + methodology).

## Deliverables checklist

| Brief deliverable | Status | Where |
|---|---|---|
| Working repo + build instructions | ✅ | [Build / run](#build--run) + [`SETUP.md`](SETUP.md) |
| README: architecture decisions | ✅ | [What it does](#what-it-does) |
| README: **kernel modifications** | ✅ θ→1e6, `g_normalized` sampling seam, embed injection, no resize | [Kernel modifications](#kernel-modifications-what-we-changed-in-the-megakernel) |
| README: how to run the Pipecat demo | ✅ | [`DEMO.md`](DEMO.md) + [Build / run](#build--run) #3 |
| Perf: decode tok/s | ✅ 1029 isolated / 924 as trunk | [Performance](#performance) |
| Perf: TTFC | ✅ **~58 ms warm median (target <60 ms met)** *(162 → 58 ms; range 56–60 ms; batched prefill + lock-free handoff)* | [Performance](#performance) |
| Perf: RTF | ✅ **~0.11 (target <0.15 met)** | [Performance](#performance) |
| Perf: end-to-end latency | ✅ ~0.55 s speak→first-audio (was ~0.75 s before the TTFC optimizations) | [Performance](#performance) |
| Streaming (frame-by-frame, not buffered) | ✅ | [Streaming + Pipecat](#streaming--pipecat) |
| Demo recording (you talking, end-to-end) | ✅ | [`recording/demo_voice_agent.mov`](recording/demo_voice_agent.mov) |

## What it does

Modern neural TTS is an autoregressive transformer (a "talker") that predicts discrete audio *codec tokens*, plus a codec decoder that turns those tokens into a waveform. Qwen3-TTS's talker is a Qwen3 transformer — the *same architecture* the megakernel already accelerates — so we swap the talker's decode trunk onto the megakernel and keep the rest of the pipeline in PyTorch.

```
mic → STT → chat-LLM → [ Qwen3-TTS TTS service ] → audio → speaker
                          │
   text → inputs_embeds → ┌─ TALKER trunk (28-layer Qwen3)  ◄── runs on the MEGAKERNEL
                          │     final hidden read via the kernel's `g_normalized` seam
                          ├─ codec_head (1024→3072) + sampling          (PyTorch)
                          ├─ code-predictor (5-layer, codebooks 1–15)   (PyTorch, accelerated — §5a)
                          └─ 12 Hz causal-ConvNet codec                   (PyTorch) → 24 kHz PCM
```

The megakernel replaces **only** the talker trunk. The code-predictor (the "codebook generator") and the codec decoder stay in PyTorch, as the task specifies — the code-predictor is then accelerated in PyTorch (hand-written GQA + Inductor + CUDA graph), not moved onto the kernel.

## Performance

Measured on RTX 5090 (Blackwell, sm_120), CUDA 13.0, driver 575.64.03, torch 2.9.1+cu130, bf16, batch 1. Full methodology + a second-box cross-check in [`bench/results.md`](bench/results.md).

| Metric | Value | Target | Notes |
|---|---|---|---|
| Megakernel decode, isolated | 1029 tok/s, 0.97 ms/step | report | reproduced baseline |
| Kernel as talker trunk (our path) | **1.08 ms/step (924/s)** | — | ~5× cheaper than the PyTorch trunk |
| Per-stage: trunk / code-predictor / codec | 24% / **71%** / 5% | — | code-predictor dominated (before accel) |
| Streaming TTFC | **~58 ms median** (warm; range 56–60 ms — prefill 13 ms / decode 4 ms / codec 17 ms / code-pred 11 ms / lock-free handoff) | <60 ms | ✅ **met** (162 → 58) |
| End-to-end RTF | **0.99 (PyTorch ref) → 0.77 (kernel) → ~0.11 after accelerating the code-predictor** (~9×, [§5a](bench/results.md)) | <0.15 | ✅ **met** |
| End-to-end latency (speak → first audio) | **~0.55 s** | report | turn-detect ~0.15 + LLM ~0.35 + TTS TTFC ~0.058 (+ relay) |
| Conversational stage | STT (`nova-2`) ~1.5 s · LLM (`llama-3.3-70b-versatile`) ~0.35 s | — | cloud, separate from kernel TTS |

**Honest bottleneck analysis** *(full methodology + per-stage numbers in [`bench/results.md`](bench/results.md) §3–§6).* The megakernel makes the talker trunk ~5× cheaper, but the **code-predictor is ~71%** of the remaining budget — and it's **launch-latency-bound, not compute-bound** (~24 µs of arithmetic/frame, but milliseconds spent dispatching many tiny kernels). Two measured wins, both pure bf16 with **no megakernel changes**:

- **RTF 0.77 → ~0.11** — accelerate the code-predictor with a hand-written GQA forward + whole-frame Inductor fusion + one CUDA-graph capture (~10.8 → ~3.0 ms/frame). Gated at **0.9999** teacher-forced logit cosine vs stock (first-step greedy match 100%; later divergence is fp near-tie amplification, not a forward bug).
- **TTFC 162 → ~58 ms** — batch the 110-token talker prefill and bridge its K/V straight into the kernel's cache (the kernel stores K/V identically to HF; post-bridge decode matches the reference at **0.9999**, prefill 87 → 13 ms), then a lock-free `asyncio.Queue` streaming handoff that removed ~22 ms *and* the run-to-run jitter (spread 59–131 → 56–60 ms).

Both primary targets met, with the talker on the megakernel and the code-predictor/codec in PyTorch as the task specifies.

## Kernel modifications (what we changed in the megakernel)

Talker trunk == Qwen3-0.6B shapes, so **no resize / no recompile** — the changes are functional, all host-side or via existing seams:

| Aspect | Challenge | What we did |
|---|---|---|
| **Shape** | is the talker the kernel's target, or the codebook generator? | The 28L/1024/16Q-8KV/hd128/inter3072 talker trunk **== Qwen3-0.6B**; weight names map 1:1 to the kernel's per-layer packing — **no resize**. |
| **RoPE** | talker uses mRoPE `[24,20,20]` | For text→speech the 3 mRoPE axes are equal → **collapses to plain 1D RoPE**; the only change is rebuilding the host cos/sin tables at **θ=1e6** (one line; also fixes the kernel's latent θ=10000 bug). |
| **Sampling** | greedy argmax → robotic/silent audio | Read the post-RMSNorm hidden from the host-visible **`g_normalized` seam** (before the kernel's argmax), run `codec_head` + temp/top-k/top-p sampling **in PyTorch — no kernel surgery**. |
| **Embedding injection** | talker runs on `inputs_embeds`, not token ids | Write each step's embedding into `embed_weight` row 0 and call `step(token_id=0)` — **no kernel change**. |

## Validation

`talker/validate_talker_trunk.py` — kernel vs reference talker hidden states: **cosine 0.99991 (min 0.99979)** over 110 prefill positions, and ~0.9999 on every decode step. `talker/megakernel_talker.py` runs full kernel-driven synthesis. The accelerated code-predictor has its own gate: `bench/correctness_cp.py` reports a teacher-forced logit cosine of **0.9999** vs stock `generate`.

## Streaming + Pipecat

- **True frame-by-frame streaming** (`pipecat_service/streaming_tts.py`): a forward-hook captures each 12.5 Hz frame's 16 codec tokens as the talker decodes; a worker thread runs generation while the consumer window-decodes via the codec and yields `TTSAudioRawFrame`s — chunks are pushed *as they're decoded*, not buffered. **TTFC ≈ 58 ms** warm median (1-frame first chunk + batched prefill + a lock-free `asyncio.Queue` handoff; range 56–60 ms).
- **Pipecat pipeline** (`pipecat_service/bot_daily.py`): `DailyTransport → Deepgram STT → Groq LLM → MegakernelStreamingTTS → audio`, with Silero VAD + smart-turn. `bot_ws.py` is a no-cloud variant (browser ↔ GPU over a `ssh -L`-forwarded WebSocket).
- **Live demo:** browser ↔ RTX 5090 round trip — speak → transcribe → LLM reply → megakernel-talker TTS → streamed audio playback (see [`DEMO.md`](DEMO.md) + the [recording](recording/demo_voice_agent.mov)).

## How I used the coding agent

Built end-to-end with **Claude Code** (the brief encourages heavy agent use). Where it did the most work: a research swarm to map the megakernel internals + Qwen3-TTS talker/code-predictor/codec decomposition; the **mRoPE→1D θ collapse** proof (the highest-risk unknown, retired before spending on compute); discovering the **`g_normalized` host seam** for argmax-free sampling; the **code-predictor acceleration** (hand-written GQA + whole-frame Inductor fusion + CUDA graph, with a teacher-forced cosine gate); writing the validation harness, the streaming service, both Pipecat transports, and the per-stage benchmark; and multi-agent audits of the submission (secret scan, deliverables, doc-clarity). Net active GPU time: well under a day.

## Build / run

**Prereq:** complete the `/workspace` bootstrap in [`SETUP.md`](SETUP.md) once (clone the two upstream repos at pinned commits, flatten the service files into `/workspace`, fetch `ref.wav`, write keys to `/opt/cfg/.env`). Then everything runs **from `/workspace`**:

```bash
# RTX 5090 box (CUDA 13.0 -devel image, driver >=570; also runs on CUDA 12.9 / torch 2.9.1+cu128).
# Weights download on first run via HF_HOME + HF_TOKEN (model: Qwen/Qwen3-TTS-12Hz-0.6B-Base).
cd /workspace
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/qwen_megakernel
PY=/opt/venv/bin/python

# 1. Validation + kernel-driven synthesis
$PY talker/validate_talker_trunk.py   # 0.9999 hidden-state match vs reference
$PY talker/megakernel_talker.py       # kernel-driven audio

# 2. Benchmarks
$PY -m qwen_megakernel.bench          # isolated megakernel tok/s (from the upstream kernel repo)
$PY bench/kernel_step_bench.py        # kernel-as-trunk per-step
$PY bench/stage_benchmark.py          # per-stage breakdown + end-to-end RTF (§3-4)
$PY bench/bench_cp_variants.py        # code-predictor v1/v2/v3 RTF + ms/frame (§5a)
$PY bench/correctness_cp.py           # v3 teacher-forced cosine gate (§5a)

# 3. Live Pipecat voice demo (keys in /opt/cfg/.env)
$PY bot_daily.py                      # prints a Daily ROOM_URL to open in a browser
$PY bot_ws.py                         # or no-cloud: open http://localhost:8000 via ssh -L 8000:localhost:8000

# 4. Server-side end-to-end (no browser) -> demo_conversation.wav
$PY demo_e2e.py
```

## Repo layout

```
SETUP.md          fresh-box runbook (RTX 5090) + reproducible env
scripts/          setup_box.sh (one-shot env), reference_run.py, inspect_weights.py, demo_e2e.py (server-side STT→LLM→TTS)
requirements_frozen.txt   exact pinned versions
talker/           validate_talker_trunk.py (0.9999 match), megakernel_talker.py (kernel-driven synthesis)
bench/            results.md, kernel_step_bench.py (trunk per-step), stage_benchmark.py (§3-4), bench_cp_variants.py + correctness_cp.py (§5a code-predictor accel + gate)
pipecat_service/  megakernel_tts_service.py, graphed_code_predictor_v3.py (hand-written GQA + Inductor-fused whole-frame CUDA graph, RTF ~0.11) + _v2.py / graphed_code_predictor.py (v2/v1 fallbacks), streaming_tts.py, bot_daily.py (Daily demo), bot_ws.py + index.html (optional local WS demo)
recording/        demo_voice_agent.mov (end-to-end voice-agent demo)
```

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — the decode megakernel.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — open Qwen3-TTS models + reference code (Apache-2.0).
- [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) — real-time voice pipeline framework.
