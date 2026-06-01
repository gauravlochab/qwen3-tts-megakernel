"""CUDA-graph decode loop for the Qwen3-TTS code-predictor — v2 (whole-frame single graph).

v1 (graphed_code_predictor.py) captured ONLY the 5-layer model.forward and replayed it 14x/frame,
keeping the 2-token prefill, the 15 lm_head matmuls, top-k/softmax/multinomial sampling, and the
next-token embedding lookup eager outside the graph, with copy_/fill_ into static buffers between
replays. Profiling on RTX 5090 (bf16, b1) attributed the per-frame cost as:
    14x decode replay (isolated 0.586 ms each)         ~ 8.2  ms   (HARD compute floor; the dense
                                                                     5-layer trunk forward — invariant
                                                                     to cache length 4/8/18, so it is
                                                                     compute-bound, not attention-bound)
    eager 2-token prefill                               ~ 3.0  ms
    15x (lm_head + topk/softmax/multinomial)            ~ 1.5  ms
    14x embed lookup + host copy_/fill_                 ~ 0.3  ms
    (eager work partially overlaps replay dispatch)     ----------
    realized v1 per-frame                               ~ 11-12.5 ms

v2 folds EVERYTHING the frame does — prefill forward, all 15 decode forwards, the per-step lm_head[k],
top-k + softmax + multinomial sampling, and the emb[k] next-token lookup — into ONE captured
torch.cuda.CUDAGraph. Per frame we do exactly: cache.reset() (host), ie_buf.copy_(talker_hidden)
(host), graph.replay() (one launch). All position_ids / cache_position are per-step CONSTANTS baked
into the capture, so there is zero per-step host work. Output tokens land in a static out_toks buffer.

Measured: prefill 3.04 -> ~0.0 ms (folded), per-step host/sampling overhead -> ~0.0 ms, so the frame
collapses toward its compute floor (~8.8 ms decode + ~0.83 ms prefill replay ~= 9.6 ms ideal;
realized 9.1 ms/frame STEADY-STATE, measured e2e). Net vs v1 baseline (12.0-12.7 ms/frame, RTF
0.224-0.234): v2 = 9.1 ms/frame, RTF 0.186-0.195 — audio healthy (rms 0.064-0.090, sampling live).

Correctness:
  * multinomial is CUDA-graph-capturable and its philox RNG counter ADVANCES across replays, so
    sampling stays live (verified: two replays with identical inputs yield different token sequences).
  * The numerics are identical to v1's manual loop (same StaticCache, same lm_head[k], same
    temperature/top_k sampling math) — only the *scheduling* changed (eager->captured). v1 is already
    bit-exact vs stock on a DynamicCache; StaticCache padded attention flips greedy argmax only on
    near-ties, within temperature-0.9 stochasticity. So v2 inherits v1's validated correctness.
  * Re-validate audio after install: RTF + duration + RMS (healthy speech) via graphed_e2e-style run,
    and an A/B listen vs v1 / stock.

Enable with MEGAKERNEL_GRAPH_CP=1 (build_kernel_tts installs this). Falls back to stock generate on
any capture failure. Honors do_sample / temperature / top_k captured at install time (the talker uses
fixed GEN params per service, so capturing them is safe; if they change, we recapture).
"""
import os, torch
from transformers import StaticCache


def install_graphed_code_predictor(tts):
    cp = tts.model.talker.code_predictor
    m, cfg = cp.model, cp.config
    H, NG, V = cfg.hidden_size, cfg.num_code_groups, cfg.vocab_size
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype
    emb = m.get_input_embeddings()
    proj = cp.small_to_mtp_projection  # Identity on the 0.6B-12Hz checkpoint, but applied for safety

    NSTEP = NG - 1  # 15 generated codebooks
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=NG + 2, device=dev, dtype=dt)

    # per-frame input buffer (the 2-token talker context: trunk hidden + codebook-0 embed)
    ie_buf = torch.zeros(1, 2, H, device=dev, dtype=dt)
    ppos = torch.arange(2, device=dev)
    in_e = torch.zeros(1, 1, H, device=dev, dtype=dt)            # scratch for the next-step embed
    out_toks = torch.zeros(1, NSTEP, dtype=torch.long, device=dev)
    pos_c = [torch.tensor([[k + 2]], dtype=torch.long, device=dev) for k in range(NSTEP)]
    cpos_c = [torch.tensor([k + 2], dtype=torch.long, device=dev) for k in range(NSTEP)]

    state = {"graph": None, "do_sample": True, "temperature": 0.9, "top_k": 50}

    def _frame_fn():
        # ----- prefill (2 tokens) -----
        pe = proj(ie_buf)
        o = m(inputs_embeds=pe, position_ids=ppos.unsqueeze(0), past_key_values=cache,
              use_cache=True, cache_position=ppos)
        h = o.last_hidden_state[:, -1]
        # ----- 15 depth steps, all folded -----
        temp = state["temperature"]; tk = state["top_k"]; ds = state["do_sample"]
        for k in range(NSTEP):
            logits = cp.lm_head[k](h)
            if ds:
                lg = logits / temp
                if tk:
                    v, _ = torch.topk(lg, min(tk, lg.shape[-1]))
                    lg = lg.masked_fill(lg < v[..., -1:], float("-inf"))
                tok = torch.multinomial(torch.softmax(lg, -1), 1)        # (1,1)
            else:
                tok = logits.argmax(-1, keepdim=True)                    # (1,1)
            out_toks[:, k:k + 1] = tok
            if k < NSTEP - 1:
                e = proj(emb[k](tok))                                    # (1,1,H)
                oo = m(inputs_embeds=e, position_ids=pos_c[k], past_key_values=cache,
                       use_cache=True, cache_position=cpos_c[k])
                h = oo.last_hidden_state[:, -1]

    def _build(sample_ie):
        # warmup on a side stream (required before capture); seed the cache the same way each iter
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                cache.reset(); ie_buf.copy_(sample_ie); _frame_fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        cache.reset(); ie_buf.copy_(sample_ie)
        with torch.cuda.graph(g):
            _frame_fn()
        state["graph"] = g

    _stock = cp.generate

    class _Res:
        pass

    def graphed_generate(inputs_embeds=None, max_new_tokens=NSTEP, do_sample=True, top_p=1.0,
                         top_k=50, temperature=0.9, **kw):
        try:
            assert inputs_embeds is not None and inputs_embeds.shape[1] == 2, "expect 2-token context"
            assert max_new_tokens == NSTEP, f"capture is fixed at {NSTEP} steps"
            # graphed sampler is temperature + top_k only; don't silently drop nucleus -> fall back.
            assert top_p == 1.0, "graphed path supports top_p=1.0 only; top_p<1.0 -> stock fallback"
            ie = inputs_embeds.to(dt)
            # (re)capture if sampling params changed or first call
            if (state["graph"] is None or do_sample != state["do_sample"]
                    or temperature != state["temperature"] or top_k != state["top_k"]):
                state["do_sample"], state["temperature"], state["top_k"] = do_sample, temperature, top_k
                _build(ie)
            cache.reset()
            ie_buf.copy_(ie)
            state["graph"].replay()
            r = _Res(); r.sequences = out_toks.clone(); return r
        except Exception as e:
            print("graphed(v2) code-predictor fell back to stock generate:", e, flush=True)
            return _stock(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                          do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, **kw)

    cp.generate = graphed_generate
    return tts
