"""Validate the B2 batched-prefill KV bridge: writing a single batched reference forward's per-layer
K/V into the megakernel's cache must reproduce the per-token-kernel prefill. The kernel stores
post-qk-norm/post-RoPE K and raw V identically to HuggingFace's DynamicCache, so the bridge should be
numerically faithful. We compare the DECODE hidden produced AFTER each prefill, all vs the full PyTorch
reference (HF prefill + HF decode). The bridge passes if it is no worse than the shipped per-token path.

    PYTHONPATH=/path/to/qwen_megakernel python bench/validate_prefill_bridge.py
"""
import os, sys, torch
import torch.nn.functional as F
from qwen_tts import Qwen3TTSModel
from qwen_megakernel.model import Decoder
from transformers import DynamicCache

HID, HEADDIM, MAXSEQ, VOCAB, NKV, NL = 1024, 128, 2048, 151936, 8, 28
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
trunk = tts.model.talker.model
trunk_fwd = trunk.forward
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

def cos(a, b): return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()

@torch.no_grad()
def run(ie, dec_emb):
    q = ie.shape[1]
    tmp = DynamicCache()
    trunk_fwd(inputs_embeds=ie.to(torch.bfloat16), use_cache=True, past_key_values=tmp,
              position_ids=torch.arange(q,device="cuda").unsqueeze(0),
              cache_position=torch.arange(q,device="cuda"))
    outd = trunk_fwd(inputs_embeds=dec_emb.view(1,1,HID).to(torch.bfloat16), use_cache=True, past_key_values=tmp,
                     position_ids=torch.tensor([[q]],device="cuda"), cache_position=torch.tensor([q],device="cuda"))
    ref = outd.last_hidden_state[0,0].clone()
    dec.reset(); ieb = ie[0].to(torch.bfloat16)
    for j in range(q): embed[0].copy_(ieb[j]); dec.step_prefill()
    embed[0].copy_(dec_emb); dec.step_no_sync(); allk = dec._norm_out.clone()
    tmp2 = DynamicCache()
    trunk_fwd(inputs_embeds=ie.to(torch.bfloat16), use_cache=True, past_key_values=tmp2,
              position_ids=torch.arange(q,device="cuda").unsqueeze(0),
              cache_position=torch.arange(q,device="cuda"))
    dec.reset()
    layers = tmp2.layers if hasattr(tmp2, "layers") else None
    for i in range(NL):
        kk = layers[i].keys if layers else tmp2.key_cache[i]
        vv = layers[i].values if layers else tmp2.value_cache[i]
        dec._k_cache[i,:,0:q,:] = kk[0].to(torch.bfloat16)
        dec._v_cache[i,:,0:q,:] = vv[0].to(torch.bfloat16)
    dec._position = q
    embed[0].copy_(dec_emb); dec.step_no_sync(); brg = dec._norm_out.clone()
    return cos(allk, ref), cos(brg, ref)

torch.manual_seed(0)
q = 110
for scale in (0.04, 0.02):
    ie = torch.randn(1,q,HID,device="cuda",dtype=torch.bfloat16)*scale
    de = torch.randn(HID,device="cuda",dtype=torch.bfloat16)*scale
    a_ref, b_ref = run(ie, de)
    print(f"[scale {scale}] per-token-kernel vs ref={a_ref:.6f}  BRIDGE vs ref={b_ref:.6f}  "
          f"-> bridge {'PASS (>= per-token path)' if b_ref >= a_ref - 0.0003 else 'FAIL'}", flush=True)
