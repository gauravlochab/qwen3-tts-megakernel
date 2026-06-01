#!/usr/bin/env bash
# One-shot environment setup for a fresh vast.ai RTX 5090 (sm_120 / Blackwell, CUDA 13.0).
# Rebuilds the exact env that runs the megakernel Qwen3-TTS Pipecat voice agent.
# Usage:  bash setup_box.sh    (run on the box, from /workspace; requirements_frozen.txt alongside)
set -e
cd /workspace
echo "[1/6] system deps (node for the local web UI bundle, ffmpeg/libsndfile for audio)..."
apt-get update -qq && apt-get install -y -qq python3-venv ffmpeg libsndfile1 nodejs npm >/dev/null 2>&1 || true

echo "[2/6] clean venv at /opt/venv..."
rm -rf /opt/venv; python3 -m venv /opt/venv
/opt/venv/bin/pip -q install --upgrade pip

echo "[3/6] torch 2.9.1 + torchaudio (cu130)..."
/opt/venv/bin/pip -q install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

echo "[4/6] the rest of the pinned requirements..."
# skip torch lines (installed above) + editable qwen_tts (installed below)
grep -viE "^torch==|^torchaudio==|^-e git" requirements_frozen.txt > /tmp/reqs_rest.txt
/opt/venv/bin/pip -q install -r /tmp/reqs_rest.txt

echo "[5/6] qwen_tts (editable)..."
cd /workspace/Qwen3-TTS && /opt/venv/bin/pip -q install -e . ; cd /workspace

echo "[6/6] verify..."
/opt/venv/bin/python - <<'PY'
import torch, torchaudio
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
from qwen_tts import Qwen3TTSModel
from pipecat.transports.daily.transport import DailyTransport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
print("SETUP_VERIFY_OK")
PY
echo "DONE. Put keys in /opt/cfg/.env (DEEPGRAM_API_KEY, GROQ_API_KEY, DAILY_API_KEY, HF_TOKEN)."
echo "Run:  export HF_HOME=/workspace/hf PYTHONPATH=/workspace/qwen_megakernel && /opt/venv/bin/python bot_daily.py"
