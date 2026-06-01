"""CUDA-graph decode loop for the Qwen3-TTS code-predictor — the ~85% inference bottleneck.

The code-predictor emits codebooks 1..15 per 12.5 Hz frame via 15 sequential HF `.generate()` steps
at seqlen-1/batch-1 → ~700 tiny kernel launches/frame, overwhelmingly DISPATCH-bound. HF generate's
own CUDA-graph path (`mode="reduce-overhead"`) fails here: with a dynamic cache it AssertionErrors in
cudagraph-trees, and with a static cache the per-frame loop hits "tensor output overwritten by a
subsequent run" (the static-KV `index_copy_` per depth-step can't be managed across the loop).

This module sidesteps that by hand-controlling capture/replay: a StaticCache + fixed input/position/
cache_position buffers, the 5-layer model.forward captured ONCE as a torch.cuda.CUDAGraph, then replayed
15× per frame with `.copy_()`/`.fill_()` into the static buffers; lm_head + sampling stay eager outside
the graph. Measured on an RTX 5090: code-predictor ~35 → ~12 ms/frame, end-to-end RTF ~0.51 → ~0.23
(2.2× cumulative), healthy speech.

Note: the StaticCache's padded attention differs numerically from the default DynamicCache by ~ulp,
which flips greedy argmax only on near-ties; under the model's temperature-0.9 sampling this is within
its own stochasticity (the manual loop is bit-exact vs stock when run on a DynamicCache). Enable with
MEGAKERNEL_GRAPH_CP=1; falls back to the stock generate on any capture failure.
"""
import torch
from transformers import StaticCache


def install_graphed_code_predictor(tts):
    cp = tts.model.talker.code_predictor
    m, cfg = cp.model, cp.config
    H, NG = cfg.hidden_size, cfg.num_code_groups
    dev, dt = next(m.parameters()).device, next(m.parameters()).dtype
    emb, proj = m.get_input_embeddings(), cp.small_to_mtp_projection

    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=NG + 2, device=dev, dtype=dt)
    in_e = torch.zeros(1, 1, H, device=dev, dtype=dt)
    pos = torch.zeros(1, 1, dtype=torch.long, device=dev)
    cpos = torch.zeros(1, dtype=torch.long, device=dev)
    state = {"graph": None, "out": None}

    def _fwd():
        return m(inputs_embeds=in_e, position_ids=pos, past_key_values=cache,
                 use_cache=True, cache_position=cpos)

    def _build():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): _fwd()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            state["out"] = _fwd()
        state["graph"] = g

    _stock = cp.generate

    class _Res:
        pass

    def graphed_generate(inputs_embeds=None, max_new_tokens=NG - 1, do_sample=True, top_p=1.0,
                         top_k=50, temperature=0.9, **kw):
        try:
            cache.reset()
            pe = proj(inputs_embeds); L = pe.shape[1]
            cp0 = torch.arange(L, device=dev)
            o = m(inputs_embeds=pe, position_ids=cp0.unsqueeze(0), past_key_values=cache,
                  use_cache=True, cache_position=cp0)
            hidden = o.last_hidden_state[:, -1]
            seqs = []; nxt = L
            if state["graph"] is None:
                _build()
            for k in range(max_new_tokens):
                logits = cp.lm_head[k](hidden)
                if do_sample:
                    lg = logits / max(temperature, 1e-5)
                    if top_k:
                        v, _ = torch.topk(lg, min(top_k, lg.shape[-1]))
                        lg = lg.masked_fill(lg < v[..., -1:], float("-inf"))
                    tok = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(-1)
                else:
                    tok = logits.argmax(-1)
                seqs.append(tok)
                if k == max_new_tokens - 1:
                    break
                in_e.copy_(proj(emb[k](tok.view(1, 1)))); pos.fill_(nxt); cpos.fill_(nxt)
                state["graph"].replay()
                hidden = state["out"].last_hidden_state[:, -1]
                nxt += 1
            r = _Res(); r.sequences = torch.stack(seqs, dim=1); return r
        except Exception as e:
            print("graphed code-predictor fell back to stock generate:", e, flush=True)
            return _stock(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                          do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, **kw)

    cp.generate = graphed_generate
    return tts
