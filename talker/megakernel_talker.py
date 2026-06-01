"""End-to-end KERNEL-DRIVEN audio synthesis.

Monkeypatches the Qwen3-TTS talker trunk (tts.model.talker.model) so its forward runs on
AlpinDale's qwen_megakernel, then lets the model's own generate_voice_clone drive everything
else (codec_head, sampling, the 5-layer code_predictor, the 12 Hz codec) unchanged.

How the kernel serves as the trunk:
  * talker trunk weights are loaded into the kernel's 11-tensors-per-layer packing (1:1 names);
  * RoPE cos/sin host-tables are rebuilt with theta=1e6 (the mRoPE->1D collapse, verified);
  * each position's inputs_embeds is injected by writing it into embed_weight row 0 and calling
    step(token_id=0) -- no kernel recompile;
  * the post-norm hidden state is read from the host-visible g_normalized buffer (we ignore the
    kernel's in-kernel argmax; codec_head + sampling run in PyTorch);
  * the HF DynamicCache is advanced with placeholder K/V so position bookkeeping stays correct
    (the kernel keeps its own KV internally). Batch size 1.

Validated: prefill AND decode hidden states match the reference talker at ~0.9999 cosine.
Note: greedy decoding degenerates to silence (a known codec-LM failure mode); the model's
intended sampling (temp 0.9 / top_k 50 / rep-penalty 1.05) produces proper speech.

Run:  PYTHONPATH=/path/to/qwen_megakernel python talker/megakernel_talker.py
"""
import types, time, torch, numpy as np, soundfile as sf
from transformers.modeling_outputs import BaseModelOutputWithPast
from qwen_tts import Qwen3TTSModel
from qwen_megakernel.model import Decoder

HID, HEADDIM, MAXSEQ, VOCAB, NKV, NL = 1024, 128, 2048, 151936, 8, 28
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_AUDIO = "ref.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
SYN = "The quick brown fox jumps over the lazy dog."


def build_kernel_talker(trunk):
    """Load the talker trunk weights into a megakernel Decoder (theta=1e6, injection buffer)."""
    sd = trunk.state_dict()
    order = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
             "self_attn.o_proj.weight", "post_attention_layernorm.weight",
             "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    lw = [sd[f"layers.{i}.{nm}"].to(torch.bfloat16).cuda().contiguous() for i in range(NL) for nm in order]
    inv = 1.0 / (1000000.0 ** (torch.arange(0, HEADDIM, 2, dtype=torch.float32) / HEADDIM))
    fr = torch.outer(torch.arange(MAXSEQ, dtype=torch.float32), inv)
    cos_t = torch.cos(fr).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_t = torch.sin(fr).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    embed = torch.zeros(1, HID, dtype=torch.bfloat16, device="cuda").contiguous()
    dummy = torch.zeros(VOCAB, HID, dtype=torch.bfloat16, device="cuda").contiguous()  # ignored head output
    dec = Decoder(weights=dict(embed_weight=embed, layer_weights=lw,
                  final_norm_weight=sd["norm.weight"].to(torch.bfloat16).cuda().contiguous(),
                  lm_head_weight=dummy, cos_table=cos_t, sin_table=sin_t), tokenizer=None, verbose=False)
    return dec, embed


def patch_trunk(trunk, dec, embed):
    def kernel_forward(self, *args, **kw):
        ie, pkv, cp = kw.get("inputs_embeds"), kw.get("past_key_values"), kw.get("cache_position")
        use_cache = kw.get("use_cache", True)
        assert ie is not None and ie.shape[0] == 1, "batch-1 kernel path"
        B, q, _ = ie.shape
        start = int(cp[0]) if cp is not None else (pkv.get_seq_length() if pkv is not None else 0)
        if start == 0:
            dec.reset()
        ie_b = ie[0].to(torch.bfloat16)
        hid = torch.empty(q, HID, dtype=torch.bfloat16, device="cuda")
        for j in range(q):
            embed[0].copy_(ie_b[j]); dec.step(0); hid[j] = dec._norm_out.to(torch.bfloat16)
        if use_cache and pkv is not None:  # advance HF cache length; kernel holds real KV internally
            z = torch.zeros(B, NKV, q, HEADDIM, dtype=torch.bfloat16, device="cuda")
            for li in range(NL):
                pkv.update(z, z, li, {})
        last = hid.unsqueeze(0)
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=pkv, hidden_states=(last,))
    trunk.forward = types.MethodType(kernel_forward, trunk)


def main():
    print("loading model...")
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    trunk = tts.model.talker.model
    dec, embed = build_kernel_talker(trunk)
    patch_trunk(trunk, dec, embed)

    torch.manual_seed(0)
    gen = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
               repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
               subtalker_top_p=1.0, subtalker_temperature=0.9)
    torch.cuda.synchronize(); t0 = time.time()
    wavs, sr = tts.generate_voice_clone(text=SYN, language="Auto", ref_audio=REF_AUDIO,
                                        ref_text=REF_TEXT, x_vector_only_mode=False, **gen)
    torch.cuda.synchronize(); t1 = time.time()
    w = wavs[0]; dur = len(w) / sr
    print(f"sr={sr} audio_dur={dur:.2f}s gen={t1-t0:.2f}s RTF={(t1-t0)/dur:.3f}")
    print(f"audio rms={np.sqrt(np.mean(w**2)):.4f} nan={np.isnan(w).any()}")
    sf.write("out_kernel.wav", w, sr)


if __name__ == "__main__":
    main()
