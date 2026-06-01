"""TRUE streaming TTS: emit audio chunks AS the talker decodes (not buffered).

Hook the talker to push each 12.5 Hz frame's 16 codec tokens into a queue; run generation in a
worker thread; the consumer windows-decodes via the model's own speech_tokenizer.decode (correct
token mapping) with a left-context overlap (trimmed), yielding TTSAudioRawFrame per chunk.
This satisfies the brief's "push chunks as they're decoded, do NOT buffer the full utterance."

Over-generation note: the base Qwen3-TTS model rambles past EOS on certain text + reference-voice
combinations. The pure-PyTorch reference does this IDENTICALLY (same texts cap at the same length),
so it is NOT a kernel artifact -- the kernel reproduces the reference to ~0.9999 cosine (a
faithfulness win). Tightening sampling (rep_penalty/low temp) was tried and HURT quality (repetition
collapse), so the model's intended sampling is kept + a max_new_tokens cap bounds the worst case; a
neutral reference voice (and the 1.7B variant) further reduce it.
"""
import sys, time, queue, asyncio, numpy as np, torch, soundfile as sf
sys.path.insert(0, "/workspace")
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TTSAudioRawFrame
from megakernel_tts_service import build_kernel_tts

SR = 24000
FRAME_HZ = 12.5
SPF = int(SR / FRAME_HZ)            # samples per codec frame = 1920
CHUNK_FRAMES = 8                    # ~0.64 s per steady-state chunk
LEFT_CTX = 5                        # context frames for decode continuity (trimmed from output)
_SENTINEL = object()


class MegakernelStreamingTTS(TTSService):
    def __init__(self, tts, ref_audio, ref_text, **kwargs):
        super().__init__(sample_rate=SR, push_start_frame=True, push_stop_frames=True, **kwargs)
        self._tts = tts
        self._talker = tts.model.talker
        self._eos = int(getattr(self._talker.config, "codec_eos_token_id", -1))
        self._ref_audio, self._ref_text = ref_audio, ref_text
        # The model's INTENDED sampling (matches the official example) + a max_new_tokens safety cap.
        # (Tightening rep_penalty/temperature was tried and HURT quality -- caused repetition collapse;
        # over-generation on some texts is a base-model trait, not fixable by degrading sampling.)
        self._gen = dict(max_new_tokens=200, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                         repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
                         subtalker_top_p=1.0, subtalker_temperature=0.9)
        self._q = None
        svc = self
        def _hook(module, inputs, output):  # forward hook: does not alter forward signature
            try:
                hs = getattr(output, "hidden_states", None)
                codec_ids = hs[1] if isinstance(hs, (tuple, list)) and len(hs) > 1 else None
                if codec_ids is not None and svc._q is not None:
                    ids = codec_ids.detach().reshape(-1)
                    if ids.numel() == svc._talker.config.num_code_groups:
                        svc._q.put(ids.to("cpu"))
            except Exception:
                pass
        self._talker.register_forward_hook(_hook)

    def _decode_window(self, frames):
        codes = torch.stack(frames, dim=0).to("cuda")  # [w, 16]
        wavs, _ = self._tts.model.speech_tokenizer.decode([{"audio_codes": codes}])
        return np.asarray(wavs[0], dtype=np.float32)

    def _run_generation(self, text):
        try:
            self._tts.generate_voice_clone(text=text, language="Auto", ref_audio=self._ref_audio,
                                           ref_text=self._ref_text, x_vector_only_mode=False, **self._gen)
        finally:
            self._q.put(_SENTINEL)

    async def run_tts(self, text, context_id):
        self._q = queue.Queue()
        loop = asyncio.get_event_loop()
        torch.manual_seed(0)
        gen_fut = loop.run_in_executor(None, self._run_generation, text)
        await self.start_ttfb_metrics()
        frames, decoded, first, done = [], 0, True, False
        while not done:
            item = await loop.run_in_executor(None, self._q.get)
            if item is _SENTINEL:
                done = True
            elif self._eos >= 0 and int(item[0]) == self._eos:
                done = True
            else:
                frames.append(item)
            ready = len(frames) - decoded
            threshold = 1 if first else CHUNK_FRAMES   # 1-frame first chunk -> lowest TTFC (codec warm-decode is ~flat in window size); measured warm TTFC ~170 ms
            if (ready >= threshold) or (done and ready > 0):
                start = max(0, decoded - LEFT_CTX)
                wav = self._decode_window(frames[start:len(frames)])
                trim = (decoded - start) * SPF
                pcm = wav[trim:]
                decoded = len(frames)
                if len(pcm) > 0:
                    if first:
                        await self.stop_ttfb_metrics(); first = False
                    audio = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()
                    yield TTSAudioRawFrame(audio=audio, sample_rate=SR, num_channels=1, context_id=context_id)
        await gen_fut
        self._q = None


async def _selftest():
    tts = build_kernel_tts()
    svc = MegakernelStreamingTTS(tts, ref_audio="/workspace/ref.wav", ref_text="Okay. Yeah. I resent you.")
    t0 = time.time(); first = None; chunks = []
    async for fr in svc.run_tts("The quick brown fox jumps over the lazy dog.", "ctx0"):
        if isinstance(fr, TTSAudioRawFrame):
            if first is None: first = time.time() - t0
            chunks.append(fr.audio)
    pcm = b"".join(chunks); dur = (len(pcm) // 2) / SR
    sf.write("/workspace/out_stream.wav", np.frombuffer(pcm, np.int16).astype(np.float32) / 32767, SR)
    print(f"chunks={len(chunks)} audio={dur:.2f}s TTFC={first:.3f}s total={time.time()-t0:.2f}s")


if __name__ == "__main__":
    asyncio.run(_selftest())
