# Optimizations applied to the upstream `qwen_megakernel`

The CUDA megakernel ([AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)) is an
external dependency cloned onto the box (not vendored here). Changes we make to *its* sources are kept
here as patches so they're reproducible; changes to *our* code (`pipecat_service/`) are committed directly.

## L1 — skip the discarded full-vocab `lm_head` during prefill (`L1_skip_prefill_lmhead.patch`)

**What & why.** During TTS prefill the megakernel decodes the ~110-token text prompt one token at a time.
Each `step()` runs the body kernel (writing the post-RMSNorm hidden state `g_normalized`, which is all the
TTS path consumes) **and then** `ldg_lm_head_fused` — a ~311 MB / 151936-row argmax matvec whose token is
**thrown away** during prefill. Measured cost: ~0.97 ms/token × 110 ≈ **107 ms of pure waste** (see
`bench/results.md` §6 and the `prove_lmhead.py` probe).

**The change.** Add `launch_ldg_decode_direct_nohead` (kernel.cu) + a `decode_no_head` torch op
(torch_bindings.cpp) + `Decoder.step_prefill()` (model.py): identical body-kernel launch, **no** lm_head.
`pipecat_service/megakernel_tts_service.py` uses `step_prefill()` in the prefill loop (falls back to
`step()` if the op isn't present, so it's safe on an unpatched build).

**Correctness — bit-identical.** Because only a discarded computation is skipped, `g_normalized` and the
KV cache are byte-for-byte the same. Validated (`val_L1.py`): per-token hidden state vs the full path =
**max abs diff 0.000, cosine 1.0000000**. The 0.9999 invariant is preserved by construction. Audio
healthy; decode-path RTF unchanged (~0.20).

**Measured impact (RTX 5090).** Isolated prefill kernel: ~107 ms → ~84 ms of kernel work removed; in the
streaming path, the prefill stage dropped ~111 ms → ~88 ms and warm TTFC ~170 ms → ~162 ms. L1 is the
bit-exact first step; its end-to-end TTFC win is smaller than the raw kernel saving because the remaining
prefill time was the per-token host loop, not the lm_head. Those larger levers were since closed — a
batched-prefill KV-bridge (per-token loop → one forward, 0.9999) and a lock-free `asyncio.Queue` streaming
handoff — taking warm TTFC to **~58 ms median** (see [`bench/results.md`](../bench/results.md) §6).

## How to apply (on the box, after cloning the upstream kernel)

```bash
cd /workspace/qwen_megakernel
patch -p0 < /path/to/optimizations/L1_skip_prefill_lmhead.patch   # or apply the three hunks by hand
rm -rf ~/.cache/torch_extensions/*                                # force a clean JIT rebuild
# next import of qwen_megakernel recompiles; verify:
python -c "import torch; from qwen_megakernel import model; print(hasattr(torch.ops.qwen_megakernel_C,'decode_no_head'))"
```
Set nothing else — `megakernel_tts_service.py` auto-detects `step_prefill` and uses it when present.
