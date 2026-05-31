"""CORE VALIDATION: does the megakernel reproduce the Qwen3-TTS talker trunk's hidden states?

Strategy: hook tts.model.talker.model during a real generation to capture the exact
inputs_embeds it receives + the post-norm hidden states it produces (ground truth).
Then replay those inputs_embeds through the megakernel (talker weights, theta=1e6,
embedding injection via embed_weight row 0, read g_normalized) and compare position-by-position.

Result (RTX 5090): cosine sim mean 0.99991 / min 0.99979 over 110 positions => MATCH.

Run:  PYTHONPATH=/path/to/qwen_megakernel python talker/validate_talker_trunk.py
"""
import torch
from qwen_tts import Qwen3TTSModel
from qwen_megakernel.model import Decoder

HID, HEADDIM, MAXSEQ, VOCAB = 1024, 128, 2048, 151936
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_AUDIO = "ref.wav"  # any short reference voice clip (Base is zero-shot voice clone)

print("loading reference model...")
tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
trunk = tts.model.talker.model  # Qwen3TTSTalkerModel (layers, norm, rotary_emb, codec_embedding, text_embedding)

# --- 1. capture inputs_embeds + post-norm hidden states from the real talker trunk ---
cap = []
def hook(mod, args, kwargs, output):
    ie = kwargs.get("inputs_embeds")
    if ie is None:
        for a in args:
            if torch.is_tensor(a) and a.dim() == 3 and a.shape[-1] == HID:
                ie = a; break
    hs = output.last_hidden_state if hasattr(output, "last_hidden_state") else (output[0] if isinstance(output, (tuple, list)) else output)
    cap.append((None if ie is None else ie.detach().float().cpu(), hs.detach().float().cpu()))
h = trunk.register_forward_hook(hook, with_kwargs=True)

_ = tts.generate_voice_clone(text="The quick brown fox jumps over the lazy dog.", language="Auto",
        ref_audio=REF_AUDIO, ref_text="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!",
        x_vector_only_mode=False, max_new_tokens=64, do_sample=False, subtalker_dosample=False)
h.remove()

cands = [(ie, hs) for ie, hs in cap if ie is not None and ie.shape[1] > 1]
ie, ref_hs = max(cands, key=lambda x: x[0].shape[1])
T = ie.shape[1]
print(f"captured prefill: inputs_embeds {tuple(ie.shape)}  ref_hidden {tuple(ref_hs.shape)}  (T={T})")

# --- 2. build a megakernel Decoder with TALKER weights + theta=1e6 + injection embed buffer ---
sd = trunk.state_dict()
order = ["input_layernorm.weight","self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight",
         "self_attn.q_norm.weight","self_attn.k_norm.weight","self_attn.o_proj.weight",
         "post_attention_layernorm.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"]
layer_weights = []
for i in range(28):
    for nm in order:
        layer_weights.append(sd[f"layers.{i}.{nm}"].to(torch.bfloat16).cuda().contiguous())

inv = 1.0 / (1000000.0 ** (torch.arange(0, HEADDIM, 2, dtype=torch.float32) / HEADDIM))  # theta=1e6 (talker)
fr = torch.outer(torch.arange(MAXSEQ, dtype=torch.float32), inv)
cos_t = torch.cos(fr).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
sin_t = torch.sin(fr).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

embed_buf = torch.zeros(1, HID, dtype=torch.bfloat16, device="cuda").contiguous()    # injection row 0
dummy_lm = torch.zeros(VOCAB, HID, dtype=torch.bfloat16, device="cuda").contiguous()  # ignored head output, avoids OOB

weights = dict(embed_weight=embed_buf, layer_weights=layer_weights,
               final_norm_weight=sd["norm.weight"].to(torch.bfloat16).cuda().contiguous(),
               lm_head_weight=dummy_lm, cos_table=cos_t, sin_table=sin_t)
dec = Decoder(weights=weights, tokenizer=None, verbose=False)

# --- 3. replay inputs_embeds through the kernel, read g_normalized per position ---
ie_gpu = ie[0].to(torch.bfloat16).cuda()
dec.reset()
mk_hs = torch.empty(T, HID, dtype=torch.float32)
for t in range(T):
    embed_buf[0].copy_(ie_gpu[t])   # inject this position's embedding into row 0; kernel reads embed_weight[0]
    dec.step(0)                      # 28 layers + final norm -> _norm_out (g_normalized)
    mk_hs[t] = dec._norm_out.detach().float().cpu()

# --- 4. compare ---
ref = ref_hs[0]
cos = torch.nn.functional.cosine_similarity(ref, mk_hs, dim=-1)
rel = (ref - mk_hs).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
print(f"\n=== megakernel vs reference talker hidden states (T={T}) ===")
print(f"cosine sim:  mean {cos.mean():.5f}  min {cos.min():.5f}")
print(f"rel L2 err:  mean {rel.mean():.4f}  max {rel.max():.4f}")
print("VERDICT:", "MATCH" if cos.mean() > 0.99 else "DIVERGENT")
