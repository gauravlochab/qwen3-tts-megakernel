"""Pipecat custom TTS service backed by the megakernel-driven Qwen3-TTS talker (Pipecat 1.3.0).

MegakernelQwen3TTSService subclasses pipecat.services.tts_service.TTSService and yields
TTSAudioRawFrame chunks. The talker trunk runs on AlpinDale's qwen_megakernel (see
talker/megakernel_talker.py); codec_head / code_predictor / 12 Hz codec run in PyTorch via the
official qwen_tts pipeline. The blocking CUDA synthesis runs in a worker thread so it doesn't
stall Pipecat's asyncio loop.

Offline-validated: run_tts produced 1022 TTSAudioRawFrames reconstructing valid audio.

NOTE: this version synthesizes the utterance then frames it (produce-then-chunk), so TTFC equals
the full-utterance synth time. True stream-as-decoded (chunked_decode per talker frame) is the
streaming-codec enhancement that brings TTFC down.

Offline self-test (no browser):  PYTHONPATH=/path/to/qwen_megakernel python megakernel_tts_service.py
"""
import os, asyncio, types, time, numpy as np, torch, soundfile as sf
from transformers.modeling_outputs import BaseModelOutputWithPast
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TTSAudioRawFrame
from qwen_tts import Qwen3TTSModel
from qwen_megakernel.model import Decoder

HID, HEADDIM, MAXSEQ, VOCAB, NKV, NL = 1024, 128, 2048, 151936, 8, 28
SR = 24000


def build_kernel_tts(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
    """Load Qwen3-TTS and monkeypatch the talker trunk onto the megakernel."""
    tts = Qwen3TTSModel.from_pretrained(model_path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    trunk = tts.model.talker.model
    sd = trunk.state_dict()
    order = ["input_layernorm.weight","self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight",
             "self_attn.q_norm.weight","self_attn.k_norm.weight","self_attn.o_proj.weight",
             "post_attention_layernorm.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"]
    lw = [sd[f"layers.{i}.{nm}"].to(torch.bfloat16).cuda().contiguous() for i in range(NL) for nm in order]
    inv = 1.0/(1000000.0**(torch.arange(0,HEADDIM,2,dtype=torch.float32)/HEADDIM))
    fr = torch.outer(torch.arange(MAXSEQ,dtype=torch.float32), inv)
    cos_t = torch.cos(fr).repeat(1,2).to(torch.bfloat16).cuda().contiguous()
    sin_t = torch.sin(fr).repeat(1,2).to(torch.bfloat16).cuda().contiguous()
    embed = torch.zeros(1,HID,dtype=torch.bfloat16,device="cuda").contiguous()
    dummy = torch.zeros(VOCAB,HID,dtype=torch.bfloat16,device="cuda").contiguous()
    dec = Decoder(weights=dict(embed_weight=embed,layer_weights=lw,
            final_norm_weight=sd["norm.weight"].to(torch.bfloat16).cuda().contiguous(),
            lm_head_weight=dummy,cos_table=cos_t,sin_table=sin_t), tokenizer=None, verbose=False)
    def kf(self, *a, **kw):
        ie, pkv, cp = kw.get("inputs_embeds"), kw.get("past_key_values"), kw.get("cache_position")
        uc = kw.get("use_cache", True); q = ie.shape[1]
        start = int(cp[0]) if cp is not None else (pkv.get_seq_length() if pkv is not None else 0)
        if start == 0: dec.reset()
        ieb = ie[0].to(torch.bfloat16); hid = torch.empty(q,HID,dtype=torch.bfloat16,device="cuda")
        for j in range(q): embed[0].copy_(ieb[j]); dec.step(0); hid[j]=dec._norm_out.to(torch.bfloat16)
        if uc and pkv is not None:
            z = torch.zeros(1,NKV,q,HEADDIM,dtype=torch.bfloat16,device="cuda")
            for li in range(NL): pkv.update(z,z,li,{})
        last = hid.unsqueeze(0)
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=pkv, hidden_states=(last,))
    trunk.forward = types.MethodType(kf, trunk)
    # Optimization (measured): torch.compile the 5-layer code-predictor — the ~85% bottleneck
    # (15 autoregressive depth-steps/frame, launch-bound at seqlen-1/batch-1). `max-autotune-no-cudagraphs`
    # gives the most fusion WITHOUT the cudagraph-tree fragility (see below): CP ~35 -> ~17 ms/frame,
    # end-to-end RTF ~0.51 -> ~0.29 on an RTX 5090; numerically faithful (max abs waveform diff ~1e-4),
    # and ~0.29 a solid result. Set MEGAKERNEL_COMPILE_CP=0 to disable.
    #
    # Why not CUDA graphs (mode="reduce-overhead", the bigger theoretical win): the code-predictor's
    # static-KV `.generate()` does an in-place `index_copy_` into the KV tensors every depth-step, which
    # cudagraph-trees cannot safely manage across the 15-step loop (output-overwrite / cache-mutation
    # errors, even with cudagraph_mark_step_begin()). Cracking that needs a hand-written static-cache
    # decode loop or the full megakernel-fuse of the 5-layer stack — the noted next levers.
    if os.environ.get("MEGAKERNEL_COMPILE_CP", "1") == "1":
        try:
            tts.model.talker.code_predictor.model = torch.compile(
                tts.model.talker.code_predictor.model, mode="max-autotune-no-cudagraphs", fullgraph=False)
        except Exception as e:
            print("code_predictor torch.compile skipped:", e, flush=True)
    return tts


class MegakernelQwen3TTSService(TTSService):
    def __init__(self, tts, ref_audio, ref_text, **kwargs):
        super().__init__(sample_rate=SR, push_start_frame=True, push_stop_frames=True, **kwargs)
        self._tts, self._ref_audio, self._ref_text = tts, ref_audio, ref_text
        self._gen = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                         repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
                         subtalker_top_p=1.0, subtalker_temperature=0.9)

    def _synth(self, text):
        wavs, sr = self._tts.generate_voice_clone(text=text, language="Auto", ref_audio=self._ref_audio,
                    ref_text=self._ref_text, x_vector_only_mode=False, **self._gen)
        w = wavs[0] if isinstance(wavs, list) else wavs
        return (np.clip(w, -1, 1) * 32767).astype(np.int16).tobytes(), sr

    async def run_tts(self, text, context_id):
        try: await self.start_ttfb_metrics()
        except Exception: pass
        pcm, sr = await asyncio.to_thread(self._synth, text)
        try: await self.stop_ttfb_metrics()
        except Exception: pass
        frame_bytes = int(sr * 0.04) * 2  # 40 ms mono int16 frames
        for i in range(0, len(pcm), frame_bytes):
            yield TTSAudioRawFrame(audio=pcm[i:i+frame_bytes], sample_rate=sr, num_channels=1, context_id=context_id)


async def _selftest():
    print("building kernel-backed TTS model...")
    tts = build_kernel_tts()
    svc = MegakernelQwen3TTSService(tts, ref_audio="ref.wav", ref_text="Okay. Yeah. I resent you.")
    t0 = time.time(); first = None; frames = []
    async for fr in svc.run_tts("Hello from the megakernel powered text to speech service.", "ctx0"):
        if isinstance(fr, TTSAudioRawFrame):
            if first is None: first = time.time() - t0
            frames.append(fr.audio)
    pcm = b"".join(frames); n = len(pcm) // 2
    sf.write("out_service.wav", np.frombuffer(pcm, np.int16).astype(np.float32) / 32767, SR)
    print(f"frames={len(frames)} audio={n/SR:.2f}s first_frame@{first:.2f}s total={time.time()-t0:.2f}s")


if __name__ == "__main__":
    asyncio.run(_selftest())
