# qwen3-tts-megakernel

Run AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a single-launch CUDA decode kernel for Qwen3-0.6B, ~1000 tok/s on an RTX 5090) as the **decode backend for the Qwen3-TTS *talker***, and stream the synthesized speech into a [Pipecat](https://docs.pipecat.ai) voice pipeline (STT → LLM → TTS → audio).

> **Status.** End-to-end working on an RTX 5090: the megakernel drives the Qwen3-TTS talker (hidden states match the reference at **0.9999** cosine), audio is streamed **frame-by-frame** into a Pipecat pipeline (Deepgram STT → Groq LLM → our megakernel TTS → audio over WebRTC), and a live browser↔GPU voice demo runs end-to-end. Benchmarks measured; honest analysis below.

## What it does

Modern neural TTS is an autoregressive transformer (a "talker") that predicts discrete audio *codec tokens*, plus a codec decoder that turns those tokens into a waveform. Qwen3-TTS's talker is a Qwen3 transformer — the *same architecture* the megakernel already accelerates — so we swap the talker's decode trunk onto the megakernel and keep the rest of the pipeline in PyTorch.

```
mic → STT → chat-LLM → [ Qwen3-TTS TTS service ] → audio → speaker
                          │
   text → inputs_embeds → ┌─ TALKER trunk (28-layer Qwen3)  ◄── runs on the MEGAKERNEL
                          │     final hidden read via the kernel's `g_normalized` seam
                          ├─ codec_head (1024→3072) + sampling          (PyTorch)
                          ├─ code-predictor (5-layer, codebooks 1–15)   (PyTorch)
                          └─ 12 Hz causal-ConvNet codec                   (PyTorch) → 24 kHz PCM
```

The megakernel replaces **only** the talker trunk. The code-predictor (the "codebook generator") and the codec decoder stay in PyTorch, as the task specifies.

## How the kernel serves as the talker trunk (no recompiles)

- **Talker trunk shape == Qwen3-0.6B** (28L / hidden 1024 / 16 Q / 8 KV / head_dim 128 / inter 3072, q/k-norm); weight tensor names map 1:1 to the kernel's per-layer packing — **no resizing**.
- **mRoPE collapses to plain 1D RoPE** for text→speech (3 position axes equal; interleaved-rope is a no-op), so the only RoPE change is rebuilding the host cos/sin tables with **θ=1e6**.
- **Sampling without kernel surgery:** read the post-RMSNorm hidden from the host-visible `g_normalized` buffer (before the kernel's argmax), run `codec_head` + sampling in PyTorch. (Greedy degenerates to silence — use the model's sampling.)
- **Embedding injection:** write each step's `inputs_embeds` into `embed_weight` row 0 and call `step(token_id=0)`.

## Validation

`talker/validate_talker_trunk.py` — kernel vs reference talker hidden states: **cosine 0.99991 (min 0.99979)** over 110 prefill positions, and ~0.9999 on every decode step. `talker/megakernel_talker.py` runs full kernel-driven synthesis.

## Streaming + Pipecat

- **True frame-by-frame streaming** (`pipecat_service/streaming_tts.py`): a forward-hook captures each 12.5 Hz frame's 16 codec tokens as the talker decodes; a worker thread runs generation while the consumer window-decodes via the codec and yields `TTSAudioRawFrame`s — chunks are pushed *as they're decoded*, not buffered. Measured **TTFC ≈ 0.30 s** (time to first audio chunk; streaming self-test).
- **Pipecat pipeline** (`pipecat_service/bot_daily.py`): `DailyTransport → Deepgram STT → Groq LLM → MegakernelStreamingTTS → audio`, with Silero VAD + smart-turn. `bot_ws.py` is a no-cloud variant (browser ↔ GPU over a `ssh -L`-forwarded WebSocket).
- **Live demo:** browser ↔ RTX 5090 round trip — speak → transcribe → LLM reply → megakernel-talker TTS → streamed audio playback.

## Performance (measured, RTX 5090, bf16, batch 1; trunk/RTF on CUDA 13 / torch 2.9.1+cu130)

Full methodology + analysis in [`bench/results.md`](bench/results.md). (The pipeline also runs on CUDA 12.9 / torch 2.9.1+cu128 — see [`SETUP.md`](SETUP.md).)

| Metric | Value | Notes |
|---|---|---|
| Megakernel decode, isolated | 1029 tok/s, 0.97 ms/step | reproduced baseline |
| Kernel as talker trunk (our path) | **1.08 ms/step (924/s)** | ~5× cheaper than the PyTorch trunk |
| Per-stage: trunk / code-predictor / codec | 24% / **71%** / 5% | code-predictor dominates |
| Streaming TTFC | **~0.30 s** | first audio chunk (streaming service) |
| End-to-end RTF | **0.99 (ref) → 0.77 (kernel)** | batch 1, non-streaming measurement |
| Conversational stage | STT ~1.5 s · LLM (Groq) ~0.35 s | cloud STT/LLM, separate from the kernel TTS |

**Honest bottleneck analysis.** The megakernel makes the talker trunk ~5× cheaper, but end-to-end RTF is Amdahl-bounded: the **code-predictor (71%)** dominates and the kernel doesn't touch it. The obvious micro-opt (skipping the wasted full-vocab lm_head) is negligible (~0.1 ms/step). **The real lever is accelerating the code-predictor** — it's the same Qwen3 kernel family (5 layers, head_dim 128, θ=1e6) — plus `torch.compile`/CUDA-graphs. The brief's RTF<0.15 target is not reached (~0.77 here); reported transparently rather than hand-waved.

## Repo layout

```
SETUP.md          fresh-box runbook (RTX 5090) + reproducible env
scripts/          setup_box.sh (one-shot env), reference_run.py, inspect_weights.py
requirements_frozen.txt   exact pinned versions
talker/           validate_talker_trunk.py (0.9999 match), megakernel_talker.py (kernel-driven synthesis)
bench/            results.md, kernel_step_bench.py
pipecat_service/  megakernel_tts_service.py, streaming_tts.py, bot_daily.py (Daily demo), bot_ws.py + index.html (local WS demo)
```

## Build / run

Fresh-box setup (env + deps) is in **[`SETUP.md`](SETUP.md)** via `scripts/setup_box.sh` (clean venv, torch, all pinned deps, editable `qwen_tts`, verify). Then:

```bash
# RTX 5090 box (CUDA 12.8+ devel image). Weights download on first run via HF_HOME + HF_TOKEN.
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/qwen_megakernel

# 1. Validation + kernel-driven synthesis
PYTHONPATH=$PYTHONPATH python talker/validate_talker_trunk.py   # 0.9999 hidden-state match
PYTHONPATH=$PYTHONPATH python talker/megakernel_talker.py       # kernel-driven audio

# 2. Benchmarks
python -m qwen_megakernel.bench                                  # isolated megakernel tok/s
PYTHONPATH=$PYTHONPATH python bench/kernel_step_bench.py         # trunk per-step

# 3. Live Pipecat voice demo (needs DEEPGRAM_API_KEY, GROQ_API_KEY, DAILY_API_KEY in /opt/cfg/.env)
PYTHONPATH=$PYTHONPATH python pipecat_service/bot_daily.py       # prints a Daily ROOM_URL to open in a browser
# or, no cloud (browser <-> GPU over an ssh -L tunnel):
PYTHONPATH=$PYTHONPATH python pipecat_service/bot_ws.py          # open http://localhost:8000 via ssh -L 8000:localhost:8000
```

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — the decode megakernel.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — open Qwen3-TTS models + reference code (Apache-2.0).
- [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) — real-time voice pipeline framework.
