"""Benchmark the code-predictor acceleration variants (v1 / v2 / v3) end-to-end on one RTX 5090.

For each variant it installs the corresponding graphed code-predictor, runs a warmup, then 5 timed
generations, reporting end-to-end RTF, the code-predictor share, and ms/frame — the numbers in
bench/results.md §5a. The talker always runs on the megakernel; only the code-predictor path changes.

    PYTHONPATH=/path/to/qwen_megakernel python bench/bench_cp_variants.py
"""
import os, sys, time, numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SVC = os.path.join(os.path.dirname(_HERE), "pipecat_service")
sys.path.insert(0, _SVC)                       # graphed_code_predictor* + megakernel_tts_service
try:
    from dotenv import load_dotenv
    load_dotenv(os.environ.get("ENV_FILE", "/opt/cfg/.env"))
except Exception:
    pass

# build the kernel-backed model WITHOUT auto-installing any CP accel (we install each variant by hand)
os.environ["MEGAKERNEL_GRAPH_CP"] = "0"; os.environ["MEGAKERNEL_COMPILE_CP"] = "0"
from megakernel_tts_service import build_kernel_tts
import importlib

REF = os.environ.get("REF_WAV", "ref.wav")
RT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
TEXT = "It sounds like you're feeling really hurt and conflicted about something I've done."
GEN = dict(max_new_tokens=160, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)

VARIANTS = [
    ("v1", "graphed_code_predictor"),
    ("v2", "graphed_code_predictor_v2"),
    ("v3", "graphed_code_predictor_v3"),
]

tts = build_kernel_tts()
cp = tts.model.talker.code_predictor
_stock = cp.generate
_cp = {"calls": 0, "time": 0.0}


def _wrap_timer():
    base = cp.generate
    def _timed(*a, **k):
        torch.cuda.synchronize(); t0 = time.time(); out = base(*a, **k)
        torch.cuda.synchronize(); _cp["time"] += time.time() - t0; _cp["calls"] += 1; return out
    cp.generate = _timed


def run(seed):
    _cp["calls"] = 0; _cp["time"] = 0.0
    torch.manual_seed(seed); torch.cuda.synchronize(); t0 = time.time()
    wavs, sr = tts.generate_voice_clone(text=TEXT, language="Auto", ref_audio=REF, ref_text=RT,
                                        x_vector_only_mode=False, **GEN)
    torch.cuda.synchronize(); dt = time.time() - t0
    w = np.asarray(wavs[0], dtype=np.float32); dur = len(w) / sr
    return dt, dur, _cp["calls"], _cp["time"], float(np.sqrt(np.mean(w ** 2)))


for tag, mod_name in VARIANTS:
    cp.generate = _stock
    try:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        mod.install_graphed_code_predictor(tts)
    except Exception as e:
        print(f"[{tag}] unavailable: {e}", flush=True); continue
    _wrap_timer()
    # warmup (also pays the v3 Inductor autotune once)
    tts.generate_voice_clone(text="Warm up.", language="Auto", ref_audio=REF, ref_text=RT,
                             x_vector_only_mode=False, max_new_tokens=16, do_sample=True, subtalker_dosample=True)
    print(f"\n=== {tag} ({mod_name}) — 5 runs ===", flush=True)
    rtfs = []
    for s in range(5):
        dt, dur, calls, cpt, rms = run(s); rtfs.append(dt / dur)
        print(f"  run {s}: total {dt:.3f}s | audio {dur:.2f}s | RTF {dt/dur:.3f} | "
              f"CP {cpt:.3f}s ({100*cpt/dt:.0f}%) over {calls} frames = {1000*cpt/max(calls,1):.2f} ms/frame | rms {rms:.3f}",
              flush=True)
    print(f"  median RTF = {sorted(rtfs)[len(rtfs)//2]:.3f}", flush=True)
