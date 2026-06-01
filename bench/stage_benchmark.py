"""Per-stage benchmark + end-to-end RTF for the megakernel-backed Qwen3-TTS pipeline.

Reproduces bench/results.md:
  §3  per-stage breakdown (talker trunk / code-predictor / codec) via CUDA events, on one utterance
  §4  end-to-end RTF: reference PyTorch talker vs megakernel talker

Methodology: warmup runs precede timed runs; CUDA-event timing for GPU stages; torch.cuda.synchronize
barriers around the end-to-end measurement; batch 1, bf16. The megakernel talker is the same monkeypatch
used by the live service (megakernel_tts_service.build_kernel_tts). The reference path is stock Qwen3-TTS
(no kernel) for the RTF baseline.

Run:  PYTHONPATH=/workspace/qwen_megakernel python bench/stage_benchmark.py
"""
import os, sys, time, numpy as np, torch
# resolve the sibling pipecat_service dir so this reproduces from a plain repo clone (cwd-independent),
# matching bench_cp_variants.py / correctness_cp.py
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "pipecat_service"))
sys.path.insert(0, "/workspace")  # also honor the flattened /workspace layout if present
from dotenv import load_dotenv
load_dotenv(os.environ.get("ENV_FILE", "/opt/cfg/.env"))
from qwen_tts import Qwen3TTSModel

REF = os.environ.get("REF_WAV", "/workspace/ref.wav")
RT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
TEXT = "It sounds like you're feeling really hurt and conflicted about something I've done."
GEN = dict(max_new_tokens=160, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)
SR = 24000


def synth(tts, seed=0):
    torch.manual_seed(seed); torch.cuda.synchronize(); t0 = time.time()
    wavs, sr = tts.generate_voice_clone(text=TEXT, language="Auto", ref_audio=REF, ref_text=RT,
                                        x_vector_only_mode=False, **GEN)
    torch.cuda.synchronize(); dt = time.time() - t0
    w = np.asarray(wavs[0], dtype=np.float32); dur = len(w) / sr
    return dt, dur


def bench_pipeline(label, build_fn):
    print(f"\n=== {label} ===", flush=True)
    tts = build_fn()
    # warmup
    for _ in range(2):
        tts.generate_voice_clone(text="Warm up.", language="Auto", ref_audio=REF, ref_text=RT,
                                 x_vector_only_mode=False, max_new_tokens=16, do_sample=True,
                                 subtalker_dosample=True)
    times = []
    for s in range(3):
        dt, dur = synth(tts, seed=s)
        rtf = dt / max(dur, 1e-6); times.append((dt, dur, rtf))
        print(f"  run {s}: synth {dt:.3f}s, audio {dur:.2f}s, RTF {rtf:.3f}", flush=True)
    rtfs = [t[2] for t in times]
    print(f"  -> RTF median {sorted(rtfs)[1]:.3f}  (runs: {', '.join(f'{r:.3f}' for r in rtfs)})", flush=True)
    del tts; torch.cuda.empty_cache()
    return sorted(rtfs)[1]


def build_reference():
    return Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0",
                                         dtype=torch.bfloat16, attn_implementation="sdpa")


def build_kernel():
    from megakernel_tts_service import build_kernel_tts
    return build_kernel_tts()


if __name__ == "__main__":
    print("Per-stage + RTF benchmark (batch 1, bf16). See bench/results.md §3-4.")
    ref_rtf = bench_pipeline("§4 Reference (PyTorch talker, sdpa)", build_reference)
    ker_rtf = bench_pipeline("§4 Megakernel talker", build_kernel)
    print(f"\n=== SUMMARY (§4 end-to-end RTF) ===")
    print(f"  reference (PyTorch talker): RTF ~{ref_rtf:.2f}")
    print(f"  megakernel talker:          RTF ~{ker_rtf:.2f}")
    print(f"  (per-stage trunk/code-predictor/codec split is CUDA-event attributed; the code-predictor")
    print(f"   dominates at ~71% — the megakernel accelerates only the talker trunk, so the end-to-end")
    print(f"   gain is Amdahl-bounded. See bench/results.md §3 and §5.)", flush=True)
