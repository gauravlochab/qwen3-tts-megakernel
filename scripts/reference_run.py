"""Run the stock PyTorch Qwen3-TTS-12Hz-0.6B pipeline.

This is our ground truth (for correctness comparison) and our baseline latency/RTF.
Uses sdpa attention so flash-attn is not required.

Env:
  HF_HOME, HF_TOKEN as usual.
Args:
  --ref-audio path to a reference voice clip (Base model is zero-shot voice clone)
"""
import argparse, time, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", default="ref.wav")
    ap.add_argument("--text", default="Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.")
    ap.add_argument("--out", default="out_ref.wav")
    ap.add_argument("--runs", type=int, default=2)  # first run warms the vocoder
    args = ap.parse_args()

    print("loading model (0.6B, sdpa, bf16)...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"model loaded in {time.time()-t0:.1f}s")

    gen = dict(max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
               repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
               subtalker_top_p=1.0, subtalker_temperature=0.9)

    for i in range(args.runs):
        torch.cuda.synchronize(); s = time.time()
        wavs, sr = tts.generate_voice_clone(
            text=args.text, language="Auto", ref_audio=args.ref_audio, ref_text=REF_TEXT,
            x_vector_only_mode=False, **gen,
        )
        torch.cuda.synchronize(); e = time.time()
        w = wavs[0]; dur = len(w) / sr
        tag = "warmup" if i == 0 else f"run{i}"
        print(f"[{tag}] sr={sr} audio_dur={dur:.2f}s gen={e-s:.2f}s RTF={(e-s)/dur:.3f}")
        if i == args.runs - 1:
            sf.write(args.out, w, sr)
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
