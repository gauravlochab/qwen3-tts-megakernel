# qwen3-tts-megakernel

Run AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a single-launch CUDA decode kernel for Qwen3-0.6B, ~1000 tok/s on an RTX 5090) as the **decode backend for the Qwen3-TTS *talker***, and stream the synthesized speech, frame-by-frame, into a [Pipecat](https://docs.pipecat.ai) voice pipeline (STT → LLM → TTS → audio).

> **Status: work in progress.** Environment + baseline + reference pipeline are up and verified on an RTX 5090; the talker→kernel integration, streaming codec, Pipecat service, and benchmarks are being built.

## What it does

Modern neural TTS is an autoregressive transformer (a "talker") that predicts discrete audio *codec tokens*, plus a codec decoder that turns those tokens into a waveform. Qwen3-TTS's talker is a Qwen3 transformer — the *same architecture* the megakernel already accelerates — so we swap the talker's decode trunk onto the megakernel and keep the rest of the pipeline in PyTorch.

```
mic → STT → chat-LLM → [ Qwen3-TTS TTS service ] → audio → speaker
                          │
   text → inputs_embeds → ┌─ TALKER trunk (28-layer Qwen3)  ◄── runs on the MEGAKERNEL
                          │     reads final hidden state via the kernel's `g_normalized` seam
                          ├─ codec_head (1024→3072) + sampling          (PyTorch)
                          ├─ code-predictor (5-layer, codebooks 1–15)   (PyTorch)
                          └─ 12 Hz causal-ConvNet codec, streamed        (PyTorch) → 24 kHz PCM
```

The megakernel replaces **only** the talker trunk. The code-predictor (the "codebook generator") and the codec decoder stay in PyTorch, as the task specifies.

## Key technical findings (verified on hardware)

- **Talker trunk shape == Qwen3-0.6B**, so the kernel needs **no resizing**: 28 layers, hidden 1024, 16 Q / 8 KV heads, head_dim 128, intermediate 3072, q/k-norm. Weight tensor names map 1:1 to the kernel's per-layer packing.
- **mRoPE collapses to plain 1D RoPE** for text→speech (the talker's 3 position axes are equal; the interleaved-rope step is a no-op), so the only RoPE change is rebuilding the host cos/sin tables with **θ=1e6** — confirmed the kernel consumes host RoPE tables (no in-kernel θ).
- **Sampling without kernel surgery:** the kernel writes the final post-RMSNorm hidden state to a host-visible tensor (`g_normalized`) *before* its in-kernel argmax. We read that hidden state and run `codec_head` + temperature/top-k/top-p sampling in PyTorch.
- **Custom embeddings without recompiling:** point the kernel's `embed_weight` at a 1-row buffer holding the per-step `inputs_embeds` and pass `token_id=0` — the kernel reads our embedding directly.

## Hardware / environment

- **RTX 5090** (Blackwell, `sm_120`). The kernel is bf16-only and Blackwell-tuned.
- CUDA 13.0 toolkit, NVIDIA driver ≥ 575, PyTorch `2.9.1+cu130`. (Builds clean on CUDA 13 with `-arch=sm_120a`; baseline reproduced at ~1029 tok/s.)
- `qwen-tts` package (`transformers==4.57.3`) for the reference pipeline, codec, and code-predictor.

## Repo layout

```
scripts/
  reference_run.py    # run the stock PyTorch Qwen3-TTS-0.6B pipeline (baseline + ground truth)
  inspect_weights.py  # dump talker/code-predictor weight tensor names + shapes
talker/               # megakernel-backed talker decode  (in progress)
codec/                # streaming 12 Hz codec decode      (in progress)
pipecat_service/      # custom streaming TTS service       (in progress)
bench/                # TTFC / RTF / tok-s measurement      (in progress)
```

## Build / run (current)

```bash
# on an RTX 5090 box, CUDA 12.8+ devel image
uv venv && source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/AlpinDale/qwen_megakernel && uv pip install -e ./qwen_megakernel  # + transformers, ninja
git clone https://github.com/QwenLM/Qwen3-TTS && uv pip install -e ./Qwen3-TTS
python scripts/reference_run.py     # baseline synthesis (writes out_ref.wav)
```

## Performance (to be filled with measured p50/p95)

| Metric | Value | Notes |
|---|---|---|
| Megakernel decode (Qwen3-0.6B) | ~1029 tok/s, 0.97 ms/tok | reproduced baseline |
| Reference full pipeline RTF | ~1.0 | stock PyTorch, sdpa, non-streaming (baseline to beat) |
| TTFC | _tbd_ | time to first audio chunk |
| Steady-state RTF | _tbd_ | gen time ÷ audio duration |

Performance methodology and an honest bottleneck analysis (the talker is cheap at 12.5 Hz; the code-predictor dominates steady-state) will accompany the final numbers.

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — the decode megakernel.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — the open Qwen3-TTS models and reference code (Apache-2.0).
- [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) — real-time voice pipeline framework.
