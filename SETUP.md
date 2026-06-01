# Fresh-box setup (RTX 5090 / sm_120 / CUDA 13.0)

Reproducible runbook for standing up the megakernel Qwen3-TTS Pipecat voice agent on a new
vast.ai instance. (Born from a painful recovery after an instance died mid-project.)

## 0. Pick a good box
- **RTX 5090 (32 GB), Blackwell / sm_120, CUDA 13.0**, `-devel` image (needs `nvcc` for the
  megakernel JIT build).
- **Check network!** A degraded box makes the live demo unusable (cloud STT/LLM/Daily calls
  pay multi-second penalties). After SSHing in, sanity-check:
  ```
  curl -s -o /dev/null -w 'groq %{time_total}s\n' https://api.groq.com/openai/v1/models -H 'Authorization: Bearer <GROQ_KEY>'
  ```
  Expect < ~0.3 s. If it's seconds, destroy and rent another.

## 1. Get the code + weights onto the box (`/workspace`)
- `qwen_megakernel/` (AlpinDale's kernel), `Qwen3-TTS/` (cloned repo, editable), `ref.wav`,
  and the `pipecat_service/` files (`bot_daily.py`, `bot_ws.py`, `streaming_tts.py`,
  `megakernel_tts_service.py`, `index.html`), plus `requirements_frozen.txt` + `scripts/setup_box.sh`.
- Model weights download on first run via `HF_HOME=/workspace/hf` + `HF_TOKEN`.

## 2. Build the environment
```
cd /workspace && bash scripts/setup_box.sh
```
This makes a clean venv at **/opt/venv** (kept off `/workspace`, which can develop stale inodes),
installs torch 2.9.1+cu130 + all pinned deps + editable `qwen_tts`, and verifies imports.

## 3. Keys
Write **/opt/cfg/.env** (kept off the corruptible workspace path):
```
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
DAILY_API_KEY=...
HF_TOKEN=...
```

## 4. Run
```
export HF_HOME=/workspace/hf HF_TOKEN=... PYTHONPATH=/workspace/qwen_megakernel
/opt/venv/bin/python bot_daily.py      # cloud Daily room (open the printed ROOM_URL)
# or the local, no-cloud WebSocket UI:
/opt/venv/bin/python bot_ws.py         # then: ssh -p <port> root@<box> -L 8000:localhost:8000 ; open http://localhost:8000
```
First run JIT-compiles the megakernel (1-3 min) + warms the vocoder.

## Notes / gotchas
- **Don't copy a venv between machines** — venvs hardcode absolute paths and large wheels copy as
  0-byte/stale inodes. Always rebuild via `setup_box.sh`.
- **bot_ws.py** serves a single esbuild JS bundle (`pcbundle.js`) so the browser loads ONE copy of
  `@pipecat-ai/client-js` (two copies -> `Class constructor E cannot be invoked without 'new'`).
  Build it on the box: `cd bundle && npm i @pipecat-ai/client-js@1.10.0 @pipecat-ai/websocket-transport@1.6.5 esbuild && ./node_modules/.bin/esbuild entry.js --bundle --format=esm --target=es2022 --outfile=/workspace/pcbundle.js`.
- SSH with `-N` for the tunnel (avoids `xterm-*` terminal errors closing the connection).