"""Pipecat voice agent: Deepgram STT -> Groq LLM -> megakernel Qwen3-TTS -> WebRTC.

Run (serves SmallWebRTC client at http://localhost:7860):
    PYTHONPATH=/path/to/qwen_megakernel python bot.py
Offline wiring check (no transport/browser):
    PYTHONPATH=/path/to/qwen_megakernel python bot.py --selftest

Keys are read from a .env (DEEPGRAM_API_KEY, GROQ_API_KEY) and must NOT be committed.
"""
import os, sys, asyncio
from dotenv import load_dotenv
load_dotenv()

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.run import main
from pipecat.runner.utils import create_transport

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megakernel_tts_service import build_kernel_tts, MegakernelQwen3TTSService

SYSTEM = ("You are a friendly voice assistant whose speech is synthesized by a CUDA megakernel "
          "running Qwen3-TTS. Reply in ONE short, natural spoken sentence.")
REF_AUDIO = os.getenv("REF_AUDIO", "ref.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"

print("loading kernel-backed TTS model (once)...")
TTS_MODEL = build_kernel_tts()

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True, audio_out_sample_rate=24000),
}


def build_services():
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    tts = MegakernelQwen3TTSService(TTS_MODEL, ref_audio=REF_AUDIO, ref_text=REF_TEXT)
    context = LLMContext([{"role": "system", "content": SYSTEM}])
    agg = LLMContextAggregatorPair(context)
    return stt, llm, tts, agg


async def bot(runner_args):
    transport = await create_transport(runner_args, transport_params)
    stt, llm, tts, agg = build_services()
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    pipeline = Pipeline([transport.input(), vad, stt, agg.user(), llm, tts, transport.output(), agg.assistant()])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, audio_out_sample_rate=24000))
    await PipelineRunner(handle_sigint=getattr(runner_args, "handle_sigint", False)).run(task)


def _selftest():
    stt, llm, tts, agg = build_services()
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    pipeline = Pipeline([vad, stt, agg.user(), llm, tts, agg.assistant()])
    PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, audio_out_sample_rate=24000))
    print("CONSTRUCT OK: pipeline + task built (stt/llm/tts/agg/vad wired)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
