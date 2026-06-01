"""Server-side end-to-end demo (no browser): real speech in (ref.wav) -> Deepgram STT -> Groq LLM
-> megakernel Qwen3-TTS reply -> stitched demo_conversation.wav. Proves the full pipeline without a
WebRTC client. Official sampling + a max_new_tokens safety cap (base-model over-generation is a
documented trait, see DEMO.md)."""
import os, sys, requests, numpy as np, soundfile as sf, torch, librosa
sys.path.insert(0, "/workspace")
from dotenv import load_dotenv
load_dotenv("/opt/cfg/.env")
from megakernel_tts_service import build_kernel_tts

DG, GQ, REF = os.getenv("DEEPGRAM_API_KEY"), os.getenv("GROQ_API_KEY"), "/workspace/ref.wav"
RT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
GEN = dict(max_new_tokens=200, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)  # model's intended params + a cap
tts = build_kernel_tts()

# 1. user turn = real speech (ref.wav) -> Deepgram STT
r = requests.post("https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                  headers={"Authorization": f"Token {DG}", "Content-Type": "audio/wav"},
                  data=open(REF, "rb").read())
transcript = r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
print("USER:", transcript, flush=True)

# 2. Groq LLM reply
r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                  headers={"Authorization": f"Bearer {GQ}"},
                  json={"model": "llama-3.3-70b-versatile", "messages": [
                      {"role": "system", "content": "You are a warm, concise voice assistant powered by a CUDA megakernel running Qwen3-TTS. Reply in ONE short spoken sentence."},
                      {"role": "user", "content": transcript}]})
reply = r.json()["choices"][0]["message"]["content"]
print("BOT :", reply, flush=True)

# 3. kernel TTS reply
torch.manual_seed(3)
wavs, sr = tts.generate_voice_clone(text=reply, language="Auto", ref_audio=REF, ref_text=RT,
                                    x_vector_only_mode=False, **GEN)
rw = np.asarray(wavs[0], dtype=np.float32)
sf.write("/workspace/demo_reply.wav", rw, sr)

# 4. stitch: user turn + pause + bot reply
uw, usr = sf.read(REF); uw = np.asarray(uw, dtype=np.float32)
if uw.ndim > 1: uw = uw.mean(1)
if usr != sr: uw = librosa.resample(uw, orig_sr=usr, target_sr=sr)
conv = np.concatenate([uw, np.zeros(int(sr * 0.6), dtype=np.float32), rw])
sf.write("/workspace/demo_conversation.wav", conv, sr)
print(f"OK conversation {len(conv)/sr:.1f}s (user {len(uw)/sr:.1f}s + reply {len(rw)/sr:.1f}s) -> demo_conversation.wav", flush=True)
