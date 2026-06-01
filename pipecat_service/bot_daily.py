"""Kernel-driven Qwen3-TTS voice agent on a Daily room (cloud-relayed WebRTC, NAT-proof).

Creates a public Daily room, joins as the bot, prints ROOM_URL for the human to open. Media
flows through Daily's cloud (both sides outbound) so it works from a headless GPU behind NAT.
Pipeline: Daily mic -> Deepgram STT -> Groq LLM -> megakernel streaming Qwen3-TTS -> Daily audio.

Needs DEEPGRAM_API_KEY, GROQ_API_KEY, DAILY_API_KEY in .env. Run: python bot_daily.py
"""
import os, sys, time, asyncio, requests
from dotenv import load_dotenv
load_dotenv("/opt/cfg/.env")
sys.path.insert(0, "/workspace")

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
from pipecat.turns.user_turn_strategies import UserTurnStrategies, default_user_turn_start_strategies
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.frames.frames import TTSSpeakFrame
from megakernel_tts_service import build_kernel_tts, prewarm_kernel_tts
from streaming_tts import MegakernelStreamingTTS

SYSTEM = ("You are a friendly voice assistant whose speech is synthesized by a CUDA megakernel "
          "running Qwen3-TTS. Reply in ONE short, natural spoken sentence.")
REF_TEXT_CLONE = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"


def create_room():
    key = os.getenv("DAILY_API_KEY")
    r = requests.post("https://api.daily.co/v1/rooms", headers={"Authorization": f"Bearer {key}"},
                      json={"privacy": "public", "properties": {"exp": int(time.time()) + 3600,
                            "enable_prejoin_ui": False, "start_video_off": True}})
    r.raise_for_status()
    return r.json()["url"]


async def main():
    room = create_room()
    print(f"\nROOM_URL: {room}\n", flush=True)
    print("loading kernel-backed TTS model (once)...", flush=True)
    tts_model = build_kernel_tts()
    # Pre-capture the code-predictor CUDA graph + warm the vocoder/sampler at startup with the service's
    # real sampling params, so the first user turn doesn't pay the one-time graph capture (~1.1s) or any
    # cold-start. Warmup-only — does not touch model math (0.9999 unaffected).
    prewarm_kernel_tts(tts_model, "/workspace/ref.wav", REF_TEXT_CLONE)

    transport = DailyTransport(room, None, "Megakernel TTS Bot",
                               DailyParams(audio_in_enabled=True, audio_out_enabled=True, audio_out_sample_rate=24000))
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    tts = MegakernelStreamingTTS(tts_model, ref_audio="/workspace/ref.wav", ref_text=REF_TEXT_CLONE)
    ctx = LLMContext([{"role": "system", "content": SYSTEM}])
    # snappier, less-twitchy turns: 1s smart-turn timeout + calmer VAD
    fast_turns = UserTurnStrategies(
        start=default_user_turn_start_strategies(),
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=1.0)))])
    agg = LLMContextAggregatorPair(ctx, user_params=LLMUserAggregatorParams(user_turn_strategies=fast_turns))
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(
        params=VADParams(confidence=0.8, start_secs=0.35, stop_secs=0.8, min_volume=0.6)))
    pipeline = Pipeline([transport.input(), vad, stt, agg.user(), llm, tts, transport.output(), agg.assistant()])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, audio_out_sample_rate=24000))

    @transport.event_handler("on_participant_joined")  # greets on every join (incl. rejoin)
    async def _greet(t, participant):
        await task.queue_frames([TTSSpeakFrame(
            "Hi! I'm a voice agent, and my speech is running on a single CUDA megakernel. Ask me anything.")])

    print("BOT READY — open ROOM_URL in your browser, allow the mic, and talk.", flush=True)
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
