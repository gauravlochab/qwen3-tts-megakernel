"""Isolated kernel trunk per-step cost (our per-step decode() path, in context).
Loads only the talker trunk weights from safetensors (no full model) and times step(0).
Result on RTX 5090: ~1.08 ms/step (924 steps/s).
"""
import glob, time, torch
from safetensors.torch import load_file
from qwen_megakernel.model import Decoder

HID, HEADDIM, MAXSEQ, VOCAB, NL = 1024, 128, 2048, 151936, 28
st = glob.glob("/workspace/hf/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/*/model.safetensors")[0]
print("loading talker trunk weights from", st)
full = load_file(st)
sd = {k[len("talker.model."):]: v for k, v in full.items() if k.startswith("talker.model.")}

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

vec = torch.randn(HID, dtype=torch.bfloat16, device="cuda")
dec.reset()
for _ in range(20):
    embed[0].copy_(vec); dec.step(0)
torch.cuda.synchronize(); t0 = time.perf_counter()
NSTEP = 300
for _ in range(NSTEP):
    embed[0].copy_(vec); dec.step(0)
torch.cuda.synchronize(); dt = time.perf_counter() - t0
print(f"kernel trunk per-step (incl .item sync + dummy lm_head): {dt/NSTEP*1000:.3f} ms/step ({NSTEP/dt:.0f} steps/s)")
print("(megakernel isolated baseline nosync = 0.97 ms/step / 1029 tok/s)")
