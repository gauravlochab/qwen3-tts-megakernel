"""Dump Qwen3-TTS talker / code-predictor weight tensor names + shapes.

Confirms the talker trunk maps 1:1 onto the megakernel's 11-tensors-per-layer packing.
Pass the path to the model.safetensors (or let it glob the HF cache snapshot).
"""
import os, sys, glob, re, collections
from safetensors import safe_open


def find_safetensors():
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = glob.glob(os.path.expanduser(
        "**/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/*/model.safetensors"), recursive=True)
    hits += glob.glob("/workspace/**/Qwen3-TTS-12Hz-0.6B-Base/snapshots/*/model.safetensors", recursive=True)
    if not hits:
        sys.exit("model.safetensors not found; pass its path as arg1")
    return hits[0]


def main():
    st = find_safetensors()
    print("FILE", st)
    with safe_open(st, framework="pt") as f:
        keys = list(f.keys())
        shape = {k: tuple(f.get_slice(k).get_shape()) for k in keys}

    print("TOTAL_TENSORS", len(keys))
    print("TOP", dict(collections.Counter(".".join(k.split(".")[:2]) for k in keys)))

    def show(title, pred):
        print(f"\n=== {title} ===")
        for k in sorted(keys):
            if pred(k):
                print(f"  {k}  {shape[k]}")

    show("talker trunk layer 0", lambda k: re.search(r"\.layers\.0\.", k) and "code_predictor" not in k and "talker" in k)
    show("talker non-layer (embed/norm/head/proj)", lambda k: "talker" in k and ".layers." not in k and "code_predictor" not in k)
    show("code_predictor (layer 0 + heads)", lambda k: "code_predictor" in k and (".layers.0." in k or ".layers." not in k))


if __name__ == "__main__":
    main()
