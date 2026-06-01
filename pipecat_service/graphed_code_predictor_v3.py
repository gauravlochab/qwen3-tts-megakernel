"""CUDA-graph decode loop for the Qwen3-TTS code-predictor — v3 (hand-written GQA, whole-frame graph).

WHY v3 EXISTS
-------------
v2 (graphed_code_predictor_v2.py) folds the whole 16-position frame into one CUDA graph, but the body
still calls the HuggingFace module `m(inputs_embeds=...)` for the 2-token prefill AND each of the 14
decode steps. Capturing the HF Qwen3 layer means the graph records HF's full attention machinery:
SDPA over a *padded* StaticCache (attention is computed against the whole max_cache_len window with an
additive mask), the per-step StaticCache.index_copy_ writes, eager rope helpers, and the dtype
shuffling around them. Profiling on an RTX 5090 attributed ~0.586 ms to each captured decode step and
called it a "compute floor" — but it is not compute, it is HF dispatch + padded-attention overhead. The
5-layer trunk at batch-1 / seqlen-1 over a 16-slot context is a handful of small matmuls; the arithmetic
is ~0.5 ms for the *entire* 16-step frame, not per step.

v3 replaces the HF module with a hand-written Qwen3 forward (`_layer`) that does exactly the kernels the
math needs and nothing else: RMSNorm, q/k/v projections, per-head q/k RMSNorm, RoPE, a static-shape
attention over a fixed [NG, NG] causal mask with an explicit fp32 KV cache, o_proj, and a SiLU MLP. The
whole 16-position frame (prefill positions 0..1 + 14 generated steps, all 15 lm_heads, sampling, and the
next-token embedding gather) is captured in ONE torch.cuda.CUDAGraph. Per frame at run time we do only:
write the 2-token context into a static buffer, refresh the Gumbel-noise buffer (host-side rand), one
graph.replay(), and a single .tolist() sync to read the codes out.

INTERFACE
---------
Drop-in for v2: replaces `cp.generate` and consumes the same 2-token `inputs_embeds` of shape [1, 2, H]
(position 0 = projected talker trunk hidden, position 1 = talker codec-embedding of code_0). It emits the
same `r.sequences` of shape [1, NG-1] = the 15 predicted codebooks (code_1..code_15). code_0 is supplied
by the caller in the prefill exactly as before, so the surrounding pipeline is untouched.

CORRECTNESS
-----------
The hand-written forward reproduces the HF Qwen3 layer numerically (RMSNorm folded in fp32 then cast,
per-head qk-norm, theta=1e6 RoPE with the rotate_half convention, head_dim**-0.5 scaling, GQA
repeat_interleave of KV). It is validated by (a) per-step logit cosine vs the HF module forward on
teacher-forced inputs, and (b) the same greedy-bit-match-vs-stock harness used for v1/v2. Sampling uses
the Gumbel-max identity argmax(logits/temp + Gumbel) ~ categorical(softmax(logits/temp)); top_k truncation
is applied before adding noise, so the sampled distribution matches v2's top_k softmax-multinomial. Fresh
Gumbel noise is generated OUTSIDE the graph each frame (the graph reads a static noise buffer), so every
frame samples independently. Greedy (do_sample=False) skips the noise and is a deterministic argmax.

Enable with MEGAKERNEL_GRAPH_CP=1 + MEGAKERNEL_CP_VARIANT=v3 (build_kernel_tts wires this). Any capture or
runtime error falls back to stock generate. Sampling params are captured at install; if they change the
graph is rebuilt.
"""
import os, math, torch
import torch.nn.functional as F


def install_graphed_code_predictor(tts):
    cp = tts.model.talker.code_predictor
    m, cfg = cp.model, cp.config
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype

    H = cfg.hidden_size
    NH = cfg.num_attention_heads
    NKV = cfg.num_key_value_heads
    HD = getattr(cfg, "head_dim", H // NH)
    NL = cfg.num_hidden_layers
    NG = cfg.num_code_groups
    EPS = cfg.rms_norm_eps
    THETA = float(cfg.rope_theta)
    GQA = NH // NKV
    NSTEP = NG - 1          # 15 generated codebooks
    SCALE = 1.0 / math.sqrt(HD)
    proj = cp.small_to_mtp_projection   # Identity on this checkpoint; applied for safety

    # ── gather the per-layer weights once (raw tensors, no module dispatch in the graph) ──────────
    L = []
    for i in range(NL):
        sa = m.layers[i].self_attn
        mlp = m.layers[i].mlp
        L.append(dict(
            ln1=m.layers[i].input_layernorm.weight,
            qw=sa.q_proj.weight, kw=sa.k_proj.weight, vw=sa.v_proj.weight,
            qn=sa.q_norm.weight, kn=sa.k_norm.weight, ow=sa.o_proj.weight,
            ln2=m.layers[i].post_attention_layernorm.weight,
            gw=mlp.gate_proj.weight, uw=mlp.up_proj.weight, dw=mlp.down_proj.weight,
        ))
    norm_w = m.norm.weight
    lm_heads = [cp.lm_head[k].weight for k in range(NSTEP)]          # [2048, H] each
    # code_predictor's own codec embeddings feed positions 2..NG-1 (indices 0..NSTEP-2 are used)
    cp_emb = [cp.model.codec_embedding[k].weight for k in range(len(cp.model.codec_embedding))]

    # ── RoPE tables (fp32), theta=1e6, rotate_half convention ─────────────────────────────────────
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32, device=dev) / HD))
    pos = torch.arange(NG, dtype=torch.float32, device=dev)
    fr = torch.outer(pos, inv)                                       # [NG, HD/2]
    cos_t = torch.cat([fr.cos(), fr.cos()], -1)                      # [NG, HD] fp32
    sin_t = torch.cat([fr.sin(), fr.sin()], -1)

    # static causal mask [NG, NG]: row p has 0 for cols 0..p, -inf for cols > p
    cmask = torch.triu(torch.full((NG, NG), float("-inf"), device=dev), diagonal=1)

    # ── static buffers (fixed GPU addresses for the captured graph) ───────────────────────────────
    ie_buf = torch.zeros(1, 2, H, device=dev, dtype=dt)                       # 2-token context in
    k_cache = torch.zeros(NL, NH, NG, HD, device=dev, dtype=torch.float32)    # GQA-expanded KV (fp32)
    v_cache = torch.zeros_like(k_cache)
    out_toks = torch.zeros(1, NSTEP, dtype=torch.long, device=dev)
    gumbel = torch.zeros(NSTEP, lm_heads[0].shape[0], device=dev, dtype=torch.float32)  # [15, 2048]

    state = {"graph": None, "do_sample": True, "temperature": 0.9, "top_k": 50}

    def _rms(x, w):
        # HF Qwen3RMSNorm: normalize in fp32, cast back to input dtype, then multiply by weight.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        return w * xf.to(x.dtype)

    def _rope(t, p):
        # t: [heads, HD] (fp32). p: python int position. rotate_half convention.
        c = cos_t[p].view(1, HD); s = sin_t[p].view(1, HD)
        half = HD // 2
        rot = torch.cat([-t[:, half:], t[:, :half]], -1)
        return t * c + rot * s

    def _layer_impl(x, p):
        # x: [1, H] residual stream. Writes KV at position p, returns updated [1, H].
        for li in range(NL):
            w = L[li]
            r = x
            h = _rms(x, w["ln1"])                                    # [1, H] (bf16)
            q = (h @ w["qw"].T).view(NH, HD)                         # [16, 128]
            k = (h @ w["kw"].T).view(NKV, HD)                        # [8, 128]
            v = (h @ w["vw"].T).view(NKV, HD)
            q = _rms(q, w["qn"]).float()                             # per-head qk-norm, then fp32
            k = _rms(k, w["kn"]).float()
            q = _rope(q, p)                                          # [16, 128] fp32
            k = _rope(k, p)
            k = k.repeat_interleave(GQA, 0)                          # [16, 128] expand KV to NH
            v = v.float().repeat_interleave(GQA, 0)
            k_cache[li, :, p, :] = k
            v_cache[li, :, p, :] = v
            # attention: q [16,1,128] x Kᵀ [16,128,NG] -> [16,1,NG]
            attn = torch.bmm(q.unsqueeze(1), k_cache[li].transpose(1, 2)) * SCALE
            attn = attn + cmask[p].view(1, 1, NG)
            attn = F.softmax(attn, -1)
            out = torch.bmm(attn, v_cache[li]).reshape(1, NH * HD).to(x.dtype)   # [1, 2048]
            x = r + out @ w["ow"].T                                  # o_proj -> [1, H]
            r = x
            h = _rms(x, w["ln2"])
            x = r + (F.silu(h @ w["gw"].T) * (h @ w["uw"].T)) @ w["dw"].T
        return x

    _layer = _layer_impl

    def _sample(h, k):
        # h: [1, H] last hidden. Returns a [1]-shaped index tensor (NEVER a 0-dim scalar — a
        # 0-dim tensor index triggers a bounds-check CPU sync that invalidates graph capture).
        hn = _rms(h, norm_w)
        logits = (hn @ lm_heads[k].T).float()                       # [1, V]
        if state["do_sample"]:
            lg = logits / state["temperature"]
            tk = state["top_k"]
            if tk:
                kth = torch.topk(lg, min(tk, lg.shape[-1]), dim=-1).values[:, -1:]   # [1, 1]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            idx = (lg + gumbel[k].unsqueeze(0)).argmax(dim=-1)      # [1]
        else:
            idx = logits.argmax(dim=-1)                             # [1]
        out_toks[0, k:k + 1] = idx
        return idx                                                  # [1]

    def _frame_fn():
        ie = proj(ie_buf)[0]                                        # [2, H]
        _layer(ie[0:1], 0)                                          # prefill position 0
        h = _layer(ie[1:2], 1)                                     # prefill position 1 -> hidden for code_1
        for k in range(NSTEP):
            idx = _sample(h, k)                                     # [1]
            if k < NSTEP - 1:
                e = proj(cp_emb[k][idx])                            # [1] index -> [1, H] gather (no sync)
                h = _layer(e, k + 2)

    def _refresh_gumbel():
        u = torch.rand_like(gumbel).clamp_(1e-20)
        gumbel.copy_(-torch.log(-torch.log(u)))

    # The frame's cost is launch/scheduling latency over ~1800 tiny kernels (the bf16 arithmetic is
    # ~24us), so the lever is FEWER kernels. Inductor fuses the rmsnorm/rope/silu elementwise chains
    # into the matmul epilogues. We compile the WHOLE unrolled frame ONCE (no varying python arg, so
    # no per-position recompilation), with max-autotune-no-cudagraphs (we own the capture), then wrap
    # the fused kernels in the manual CUDA graph below to also remove host launch overhead.
    fr = {"run": _frame_fn}
    if os.environ.get("MEGAKERNEL_CP_COMPILE", "1") == "1":
        try:
            import torch._dynamo as _dyn
            _dyn.config.cache_size_limit = max(getattr(_dyn.config, "cache_size_limit", 8), 64)
        except Exception:
            pass
        fr["run"] = torch.compile(_frame_fn, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)

    def _capture(sample_ie, fn):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                ie_buf.copy_(sample_ie)
                if state["do_sample"]: _refresh_gumbel()
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        ie_buf.copy_(sample_ie)
        if state["do_sample"]: _refresh_gumbel()
        with torch.cuda.graph(g):
            fn()
        state["graph"] = g

    def _build(sample_ie):
        try:
            _capture(sample_ie, fr["run"])
        except Exception as e:
            if fr["run"] is not _frame_fn:
                print("CP frame-compile capture failed -> uncompiled manual graph:", e, flush=True)
                fr["run"] = _frame_fn
                _capture(sample_ie, _frame_fn)
            else:
                raise

    _stock = cp.generate

    class _Res:
        pass

    @torch.no_grad()
    def graphed_generate(inputs_embeds=None, max_new_tokens=NSTEP, do_sample=True, top_p=1.0,
                         top_k=50, temperature=0.9, **kw):
        # no_grad wraps BOTH capture (_build) and replay: Inductor must trace the in-place KV-cache
        # mutation without autograd version counters (otherwise the compile backend raises on the
        # in-place SelectBackward), and it makes correctness independent of the caller's grad context.
        try:
            assert inputs_embeds is not None and inputs_embeds.shape[1] == 2, "expect 2-token context"
            assert max_new_tokens == NSTEP, f"capture is fixed at {NSTEP} steps"
            # the graphed sampler implements temperature + top_k only; a caller requesting nucleus
            # (top_p < 1.0) must NOT be silently downgraded -> raise so we fall back to stock generate.
            assert top_p == 1.0, "graphed path supports top_p=1.0 only; top_p<1.0 -> stock fallback"
            ie = inputs_embeds.to(dt)
            if (state["graph"] is None or do_sample != state["do_sample"]
                    or temperature != state["temperature"] or top_k != state["top_k"]):
                state["do_sample"], state["temperature"], state["top_k"] = do_sample, temperature, top_k
                _build(ie)
            ie_buf.copy_(ie)
            if state["do_sample"]:
                _refresh_gumbel()                                  # fresh noise OUTSIDE the graph
            state["graph"].replay()
            r = _Res(); r.sequences = out_toks.clone(); return r
        except Exception as e:
            print("graphed(v3) code-predictor fell back to stock generate:", e, flush=True)
            return _stock(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                          do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, **kw)

    @torch.no_grad()
    def _eager_tf_logits(inputs_embeds, teacher_tokens):
        # Validation only (NOT graphed): run the hand-written forward teacher-forced on the given
        # token sequence and return per-step logits [NSTEP, V]. Isolates forward correctness from
        # autoregressive divergence amplification. Uses its OWN local KV (leaves the graph cache alone).
        nonlocal k_cache, v_cache
        kc, vc = k_cache, v_cache
        k_cache = torch.zeros_like(kc); v_cache = torch.zeros_like(vc)
        try:
            ie = proj(inputs_embeds.to(dt))[0]                     # [2, H]
            _layer(ie[0:1], 0)
            h = _layer(ie[1:2], 1)
            out = []
            for k in range(NSTEP):
                hn = _rms(h, norm_w)
                out.append((hn @ lm_heads[k].T).float())          # [1, V]
                if k < NSTEP - 1:
                    tok = torch.as_tensor([int(teacher_tokens[k])], device=dev)
                    h = _layer(proj(cp_emb[k][tok]), k + 2)
            return torch.cat(out, 0)                               # [NSTEP, V]
        finally:
            k_cache, v_cache = kc, vc

    cp._v3_eager_tf_logits = _eager_tf_logits
    cp.generate = graphed_generate
    return tts
