# qwen3-tts-megakernel

Run AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a single-launch CUDA decode kernel for Qwen3-0.6B, ~1000 tok/s on an RTX 5090) as the **decode backend for the Qwen3-TTS *talker***, and stream the synthesized speech into a [Pipecat](https://docs.pipecat.ai) voice pipeline (STT → LLM → TTS → audio).

> **Status.** Megakernel drives the Qwen3-TTS talker end-to-end and produces real speech, validated on an RTX 5090 (hidden states match the reference at 0.9999 cosine). Benchmarks done. Streaming codec + Pipecat pipeline + live demo are the remaining work.

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

## Performance (measured, RTX 5090 / CUDA 13 / torch 2.9.1+cu130, bf16, batch 1)

Full methodology + analysis in [`bench/results.md`](bench/results.md).

| Metric | Value | Notes |
|---|---|---|
| Megakernel decode, isolated | 1029 tok/s, 0.97 ms/step | reproduced baseline |
| Kernel as talker trunk (our path) | **1.08 ms/step (924/s)** | ~5× cheaper than the PyTorch trunk |
| Per-stage: trunk / code-predictor / codec | 24% / **71%** / 5% | code-predictor dominates |
| End-to-end RTF | **0.99 (ref) → 0.77 (kernel)** | batch 1, non-streaming |

**Honest bottleneck analysis.** The megakernel makes the talker trunk ~5× cheaper, but end-to-end RTF is Amdahl-bounded: the **code-predictor (71%)** dominates and the kernel doesn't touch it. The obvious micro-opt (skipping the wasted full-vocab lm_head) is negligible (~0.1 ms/step). **The real lever is accelerating the code-predictor** — it's the same Qwen3 kernel family (5 layers, head_dim 128, θ=1e6) — plus streaming + `torch.compile`/CUDA-graphs. The brief's RTF<0.15 target is not reached (~0.77 here); reported transparently rather than hand-waved.

## Repo layout

```
scripts/   reference_run.py, inspect_weights.py
talker/    validate_talker_trunk.py (0.9999 match), megakernel_talker.py (kernel-driven synthesis)
bench/     results.md, stage_benchmark.py, kernel_step_bench.py
codec/     streaming 12 Hz codec        (in progress)
pipecat_service/  streaming TTS service  (in progress)
```

## Build / run

```bash
# RTX 5090 box, CUDA 12.8+ devel image
uv venv && source .venv/bin/activate
uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/AlpinDale/qwen_megakernel   # + transformers, ninja, accelerate
git clone https://github.com/QwenLM/Qwen3-TTS && uv pip install -e ./Qwen3-TTS
python scripts/reference_run.py                                  # baseline synthesis
PYTHONPATH=./qwen_megakernel python talker/validate_talker_trunk.py   # 0.9999 hidden-state match
PYTHONPATH=./qwen_megakernel python talker/megakernel_talker.py       # kernel-driven audio
```

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — the decode megakernel.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — open Qwen3-TTS models + reference code (Apache-2.0).
- [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) — real-time voice pipeline framework.
