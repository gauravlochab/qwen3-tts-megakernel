# Demo

**▶ Recording:** [`recording/demo_voice_agent.mov`](recording/demo_voice_agent.mov) — the live
browser ↔ RTX 5090 voice loop (user speaks → Deepgram STT → Groq LLM → megakernel Qwen3-TTS →
streamed audio back), end-to-end.

## Live voice demo (Daily, NAT-proof)

Run `bot_daily.py` from `/workspace` (keys in `/opt/cfg/.env`: `DEEPGRAM_API_KEY`, `GROQ_API_KEY`,
`DAILY_API_KEY`). It creates a Daily room and prints `ROOM_URL`; open that URL in a browser, allow
the mic, and talk. Pipeline: Daily mic -> **Deepgram STT** (`nova-2`) -> **Groq LLM**
(`llama-3.3-70b-versatile`) -> **megakernel Qwen3-TTS (streaming)** -> Daily audio. Use **headphones**
(avoids the bot hearing itself). Daily is used because WebRTC media is UDP peer-to-peer and a headless
GPU behind NAT cannot do direct media without a relay (the SmallWebRTC-over-SSH-tunnel path fails ICE).

## Server-side demo (no browser)

`python scripts/demo_e2e.py` runs the full loop on a fixed input and writes `demo_conversation.wav`:
real speech (`ref.wav`) -> Deepgram STT -> Groq LLM -> megakernel TTS reply, stitched into one clip.
Example exchange:
- **USER:** "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it. And thanks to you."
- **BOT:** "It sounds like you're feeling really hurt and conflicted about something I've done."

## Honest limitation: base-model over-generation

Qwen3-TTS-**0.6B** rambles past EOS on some text + reference-voice combinations, producing audio much
longer than the text. The **pure-PyTorch reference over-generates identically** on the same texts, so
this is a **base-model trait, not a kernel artifact** -- the kernel matches the reference to ~0.9999
cosine. Mitigations in place / recommended: a `max_new_tokens` cap (~16 s ceiling), a **neutral
reference voice** (the bundled clip is emotional, which the voice-clone copies), and the **1.7B**
variant. Tightening sampling (repetition penalty / low temperature) was tried and **hurt** quality
(repetition collapse), so the model's intended sampling is retained.

## Streaming (brief requirement)

True frame-by-frame streaming is implemented in `pipecat_service/streaming_tts.py`: audio chunks are
emitted **as the talker decodes** (first chunk << total), satisfying "push chunks as decoded, do NOT
buffer". **TTFC ≈ 0.30 s warm** (streaming self-test, the steady-state demo condition); the very first
utterance after process start is higher (~2 s) due to one-time CUDA/codec cold-start. Both are far below
the ~full-utterance latency of buffer-then-send. (Brief target is <60 ms — see `bench/results.md` §6 for
the honest gap analysis.)
