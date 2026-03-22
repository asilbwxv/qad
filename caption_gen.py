import os
import sys
import json
import argparse
import glob
import warnings
from PIL import Image
import torch
from typing import List, Dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def normalize(t: str) -> str:
    return " ".join(t.strip().lower().split())

def load_blip(ckpt: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(ckpt)
    model = BlipForConditionalGeneration.from_pretrained(ckpt).to(DEVICE).eval()
    return proc, model

def generate_caption_candidates(img: Image.Image, proc, model, config) -> List[Dict]:
    """
    Generates a diverse candidate pool by sweeping temperatures and using 
    both beam search and sampling strategies.
    """
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)
    candidates = []
    
    with torch.inference_mode():
        # 1. Standard Beam Search (Safe baseline)
        if "beam" in config.algos:
            out = model.generate(**inputs, max_new_tokens=40, num_beams=config.beam_size, num_return_sequences=config.beam_size)
            texts = proc.batch_decode(out, skip_special_tokens=True)
            for t in texts:
                candidates.append({"text": normalize(t), "meta": {"algo": "beam"}})
                
        # 2. Diverse Sampling (Sweeping temperatures to force diversity)
        if "sample" in config.algos:
            temps = [0.7, 0.9, 1.1]
            top_ps = [0.9, 0.95]
            
            for t in temps:
                for p in top_ps:
                    try:
                        out = model.generate(
                            **inputs, max_new_tokens=40, do_sample=True, 
                            temperature=t, top_p=p, repetition_penalty=1.1,
                            num_return_sequences=config.samples_per_param
                        )
                        texts = proc.batch_decode(out, skip_special_tokens=True)
                        for text in texts:
                            candidates.append({"text": normalize(text), "meta": {"algo": "sample", "temp": t, "top_p": p}})
                    except Exception as e:
                        warnings.warn(f"BLIP sampling failed at temp {t}, top_p {p}: {e}")

    # Deduplicate while preserving origin metadata
    seen = set()
    unique_cands = []
    for c in candidates:
        if c["text"] and c["text"] not in seen:
            seen.add(c["text"])
            unique_cands.append(c)

    return unique_cands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Folder with images")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--ckpt", default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--algos", default="beam,sample", help="Comma-separated: beam,sample")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--samples_per_param", type=int, default=3, help="Samples generated per temp/top_p pairing")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process")
    args = parser.parse_args()
    
    img_files = glob.glob(os.path.join(args.dir, "**", "*.*"), recursive=True)
    img_files = [f for f in img_files if f.lower().endswith(IMG_EXTS)]
    
    proc, model = load_blip(args.ckpt)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as fout:
        for i, path in enumerate(img_files):
            if args.limit > 0 and i >= args.limit:
                break
            uid = os.path.splitext(os.path.basename(path))[0]
            try:
                img = Image.open(path).convert("RGB")
                cands = generate_caption_candidates(img, proc, model, args)
                
                rec = {
                    "image_id": uid,
                    "image_path": path,
                    "candidates": cands
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
            except Exception as e:
                warnings.warn(f"Failed processing {path}: {e}")

if __name__ == "__main__":
    main()
