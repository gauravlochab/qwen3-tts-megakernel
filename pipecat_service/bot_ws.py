"""Kernel-driven Qwen3-TTS voice agent over a LOCAL WebSocket (no cloud, no Daily).

One uvicorn app serves the browser UI at GET / and a Pipecat WebSocket endpoint at /ws. Because a
WebSocket is plain TCP, it rides an `ssh -L 8000:localhost:8000` tunnel cleanly -- so a browser on a
laptop can talk to this bot on a headless GPU box behind NAT, with NO TURN/SFU relay (unlike WebRTC,
which needs UDP+STUN/TURN and cannot be ssh-forwarded).

Pipeline (same as the Daily bot): browser mic -> Deepgram STT -> Groq LLM ->
MegakernelQwen3TTSService (talker on the CUDA megakernel) -> browser playback.

Run:  python bot_ws.py        (uvicorn on :8000; model is built+warmed once at startup)
Then: ssh -p <port> root@<box> -L 8000:localhost:8000   and open http://localhost:8000
"""
import os, sys, asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv("/opt/cfg/.env")
sys.path.insert(0, "/workspace")

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, Response
import uvicorn

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_start.transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.frames.frames import TTSSpeakFrame
from megakernel_tts_service import build_kernel_tts, MegakernelQwen3TTSService, prewarm_kernel_tts

SYSTEM = ("You are a friendly voice assistant whose speech is synthesized by a CUDA megakernel "
          "running Qwen3-TTS. Reply in ONE very short spoken sentence — at most 12 words. Be direct.")
REF_TEXT_CLONE = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
HERE = os.path.dirname(os.path.abspath(__file__))

# Build the kernel TTS model ONCE for the whole server (heavy: weights + CUDA megakernel).
_TTS_MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _TTS_MODEL
    print("loading kernel-backed TTS model (once)...", flush=True)
    _TTS_MODEL = build_kernel_tts()
    # Pre-capture the code-predictor CUDA graph + warm the vocoder/sampler at startup (warmup-only;
    # 0.9999 unaffected) so the first user turn doesn't pay the one-time graph capture or any cold-start.
    prewarm_kernel_tts(_TTS_MODEL, "/workspace/ref.wav", REF_TEXT_CLONE)
    print("BOT READY — open http://localhost:8000 (via ssh -L 8000:localhost:8000) and talk.", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/pcbundle.js")
async def pcbundle():
    # Single esbuild bundle of client-js + websocket-transport -> guarantees ONE client-js copy
    # (the CDN/esm.sh path kept loading two, causing "Class constructor E cannot be invoked without 'new'").
    # Explicit JS content-type: browsers refuse to execute a module served as application/json.
    with open(os.path.join(HERE, "pcbundle.js"), "rb") as f:
        data = f.read()
    return Response(content=data, media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True,
            audio_in_sample_rate=16000, audio_out_sample_rate=24000,
            add_wav_header=False, serializer=ProtobufFrameSerializer()))

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    tts = MegakernelQwen3TTSService(_TTS_MODEL, ref_audio="/workspace/ref.wav", ref_text=REF_TEXT_CLONE)
    ctx = LLMContext([{"role": "system", "content": SYSTEM}])
    # Barge-in OFF: the produce-then-chunk TTS has a multi-second synth gap before audio; without this
    # any user sound during that gap cancels the in-flight reply (verified). Smart-turn stop at 1s.
    turns = UserTurnStrategies(
        start=[VADUserTurnStartStrategy(enable_interruptions=False),
               TranscriptionUserTurnStartStrategy(enable_interruptions=False)],
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=1.0)))])
    agg = LLMContextAggregatorPair(ctx, user_params=LLMUserAggregatorParams(user_turn_strategies=turns))
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(
        params=VADParams(confidence=0.8, start_secs=0.35, stop_secs=0.8, min_volume=0.6)))
    pipeline = Pipeline([transport.input(), vad, stt, agg.user(), llm, tts, transport.output(), agg.assistant()])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, audio_out_sample_rate=24000))

    @transport.event_handler("on_client_connected")
    async def _greet(t, client):
        await task.queue_frames([TTSSpeakFrame(
            "Hi! I'm a voice agent, and my speech is running on a single CUDA megakernel. Ask me anything.")])

    @transport.event_handler("on_client_disconnected")
    async def _bye(t, client):
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
