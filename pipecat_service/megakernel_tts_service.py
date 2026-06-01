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
        # L1: prefill skips the discarded full-vocab lm_head per token (step_prefill) — same body kernel,
        # so _norm_out + KV are byte-identical (0.9999 preserved), but ~107ms of wasted lm_head removed.
        # Falls back to step(0) if the no-head kernel op isn't present (older build).
        _pf = getattr(dec, "step_prefill", None) if q > 1 else None
        if _pf is not None:
            for j in range(q): embed[0].copy_(ieb[j]); _pf(); hid[j]=dec._norm_out.to(torch.bfloat16)
        else:
            for j in range(q): embed[0].copy_(ieb[j]); dec.step(0); hid[j]=dec._norm_out.to(torch.bfloat16)
        if uc and pkv is not None:
            z = torch.zeros(1,NKV,q,HEADDIM,dtype=torch.bfloat16,device="cuda")
            for li in range(NL): pkv.update(z,z,li,{})
        last = hid.unsqueeze(0)
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=pkv, hidden_states=(last,))
    trunk.forward = types.MethodType(kf, trunk)
    # Optimization (measured): the 5-layer code-predictor is the ~85% bottleneck (15 autoregressive
    # depth-steps/frame at seqlen-1/batch-1 -> ~700 tiny kernel launches/frame, DISPATCH-bound). Two
    # shippable accelerations, both validated on an RTX 5090 (end-to-end RTF, audio healthy):
    #
    #   (A) MEGAKERNEL_GRAPH_CP=1 (default, best): a HAND-CAPTURED CUDA graph of the WHOLE per-frame
    #       depth loop (graphed_code_predictor_v2.py) — folds the 2-token prefill + all 15 decode
    #       forwards + the 15 lm_head matmuls + top-k/softmax/multinomial sampling + next-token embed
    #       into ONE captured graph (only cache.reset() + input copy_ + graph.replay() per frame).
    #       Measured CP ~35 -> ~11 ms/frame, end-to-end RTF ~0.51 -> ~0.21. (v1, graphed_code_predictor.py,
    #       graphed only the model forward and left lm_head+sampling eager -> ~12.5 ms/frame, RTF 0.23.)
    #       This is the win HF generate can't get itself: its reduce-overhead path AssertionErrors on a
    #       dynamic cache and, with a static cache, the per-step in-place KV index_copy_ trips cudagraph-
    #       trees; we sidestep it by controlling capture/replay over a StaticCache + static I/O buffers.
    #   (B) MEGAKERNEL_COMPILE_CP=1 (fallback): torch.compile(max-autotune-no-cudagraphs) -> ~17 ms/frame,
    #       RTF ~0.29. Pure-fusion, no graph fragility; used when graph is disabled or unavailable.
    #
    # Numerically: the graph's StaticCache differs from the default DynamicCache by ~ulp (flips greedy
    # argmax only on near-ties; within the model's temperature-0.9 sampling stochasticity; multinomial is
    # graph-capturable and its philox RNG advances across replays). Falls back to v1 then stock generate
    # on any failure. Disable all with MEGAKERNEL_GRAPH_CP=0 MEGAKERNEL_COMPILE_CP=0.
    if os.environ.get("MEGAKERNEL_GRAPH_CP", "1") == "1":
        try:
            from graphed_code_predictor_v2 import install_graphed_code_predictor
            install_graphed_code_predictor(tts)
        except Exception as e:
            print("graphed code-predictor v2 skipped, trying v1:", e, flush=True)
            try:
                from graphed_code_predictor import install_graphed_code_predictor
                install_graphed_code_predictor(tts)
            except Exception as e2:
                print("graphed code-predictor skipped:", e2, flush=True)
    elif os.environ.get("MEGAKERNEL_COMPILE_CP", "1") == "1":
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
