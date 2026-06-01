# Fresh-box setup (RTX 5090 / sm_120 / CUDA 13.0)

Reproducible runbook for standing up the megakernel Qwen3-TTS Pipecat voice agent on a new
vast.ai instance. (Born from a painful recovery after an instance died mid-project.)

## 0. Pick a good box
- **RTX 5090 (32 GB), Blackwell / sm_120, CUDA 13.0**, `-devel` image (needs `nvcc` for the
  megakernel JIT build). **NVIDIA driver ≥ 570** (measured on `575.64.03`), **≥ 60 GB disk**
  (weights ~2 GB + venv/torch ~8 GB + JIT cache + headroom).
- **Check network!** A degraded box makes the live demo unusable (cloud STT/LLM/Daily calls
  pay multi-second penalties). After SSHing in, sanity-check:
  ```
  curl -s -o /dev/null -w 'groq %{time_total}s\n' https://api.groq.com/openai/v1/models -H 'Authorization: Bearer <GROQ_KEY>'
  ```
  Expect < ~0.3 s. If it's seconds, destroy and rent another.

## 1. Bootstrap `/workspace` (copy-paste, in order)

Every runnable script does `sys.path.insert(0, "/workspace")` and imports `megakernel_tts_service`
/ `streaming_tts` as **top-level** modules, and uses `PYTHONPATH=/workspace/qwen_megakernel`. So the
`/workspace` layout below must exist before anything runs. Run these from a clone of THIS repo:

```bash
cd /workspace

# (a) the two load-bearing external repos (pinned to the commits this was built against)
git clone https://github.com/AlpinDale/qwen_megakernel /workspace/qwen_megakernel
git -C /workspace/qwen_megakernel checkout 5030e15
git clone https://github.com/QwenLM/Qwen3-TTS /workspace/Qwen3-TTS
git -C /workspace/Qwen3-TTS checkout 022e286

# (b) flatten THIS repo's service + bench + talker + script files into /workspace so the imports resolve
cp /path/to/this-repo/pipecat_service/*.py /workspace/
cp /path/to/this-repo/pipecat_service/index.html /workspace/
mkdir -p /workspace/bench && cp /path/to/this-repo/bench/*.py /workspace/bench/
mkdir -p /workspace/talker && cp /path/to/this-repo/talker/*.py /workspace/talker/   # README step 1 runs talker/*.py from /workspace
cp /path/to/this-repo/scripts/demo_e2e.py /workspace/
cp /path/to/this-repo/requirements_frozen.txt /path/to/this-repo/scripts/setup_box.sh /workspace/

# (c) reference voice clip for voice-clone (the demos' ref_text matches THIS clip)
curl -L -o /workspace/ref.wav https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav
#   ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
```

- **Model weights** download on first run via `HF_HOME=/workspace/hf` + `HF_TOKEN` — model id
  `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (Apache-2.0, ungated; a HF token still avoids rate limits).

## 2. Build the environment
```
cd /workspace && bash scripts/setup_box.sh
```
This makes a clean venv at **/opt/venv** (kept off `/workspace`, which can develop stale inodes),
installs torch 2.9.1+cu130 + all pinned deps + editable `qwen_tts`, and verifies imports.

## 3. Keys
Write **/opt/cfg/.env** — this is the ONE path every script reads (`bot_daily.py`, `bot_ws.py`,
`demo_e2e.py`, `stage_benchmark.py` all `load_dotenv("/opt/cfg/.env")`). Kept off `/workspace`
(corruptible inodes) and out of the repo (never committed):
```
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
DAILY_API_KEY=...
HF_TOKEN=...
```

## 4. Run (from `/workspace`, where the flat imports resolve)
```
cd /workspace
export HF_HOME=/workspace/hf HF_TOKEN=... PYTHONPATH=/workspace/qwen_megakernel
/opt/venv/bin/python bot_daily.py      # cloud Daily room (open the printed ROOM_URL)
# or the local, no-cloud WebSocket UI:
/opt/venv/bin/python bot_ws.py         # then: ssh -p <port> root@<box> -L 8000:localhost:8000 ; open http://localhost:8000
# or server-side, no browser (writes demo_conversation.wav):
/opt/venv/bin/python demo_e2e.py
```
First run JIT-compiles the megakernel (1-3 min) + warms the vocoder.

## Notes / gotchas
- **Don't copy a venv between machines** — venvs hardcode absolute paths and large wheels copy as
  0-byte/stale inodes. Always rebuild via `setup_box.sh`.
- **bot_ws.py** serves a single esbuild JS bundle (`pcbundle.js`) so the browser loads ONE copy of
  `@pipecat-ai/client-js` (two copies -> `Class constructor E cannot be invoked without 'new'`).
  Build it on the box: `cd bundle && npm i @pipecat-ai/client-js@1.10.0 @pipecat-ai/websocket-transport@1.6.5 esbuild && ./node_modules/.bin/esbuild entry.js --bundle --format=esm --target=es2022 --outfile=/workspace/pcbundle.js`.
- SSH with `-N` for the tunnel (avoids `xterm-*` terminal errors closing the connection).