import os
import sys
import json
import argparse
import glob
import warnings
import torch
import soundfile as sf
from typing import List, Dict, Tuple
from dataclasses import dataclass
import jiwer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

# Normalizer for deduplication
norm_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

@dataclass
class Candidate:
    text: str
    meta: Dict

def load_audio(path: str) -> Tuple[object, int]:
    audio, sr = sf.read(path)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    return audio, sr

def load_whisper(ckpt: str):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(ckpt)
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return proc, model

def generate_whisper_candidates(audio, sr, proc, model, config) -> List[Candidate]:
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(DEVICE, dtype=torch.float16)
    forced_decoder_ids = proc.get_decoder_prompt_ids(language=config.lang, task=config.task)
    
    candidates = []
    
    with torch.inference_mode():
        # 1. Standard / High-Quality Beam Search
        if "beam" in config.algos:
            out = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids,
                num_beams=config.beam_size, num_return_sequences=config.beam_size,
                return_dict_in_generate=True, output_scores=True
            )
            texts = proc.batch_decode(out.sequences, skip_special_tokens=True)
            for i, t in enumerate(texts):
                if t.strip():
                    logprob = out.sequences_scores[i].item() if hasattr(out, "sequences_scores") else 0.0
                    candidates.append(Candidate(text=t.strip(), meta={"algo": "beam", "logprob": logprob}))

        # 2. Controlled Low-Temperature Sampling (Sweeping T=[0.3, 0.6] prevents wild hallucinations)
        if "nucleus" in config.algos:
            for temp in [0.3, 0.6]:
                out = model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids,
                    do_sample=True, top_p=config.top_p, temperature=temp,
                    num_beams=1, num_return_sequences=max(2, config.n_samples // 2),
                    return_dict_in_generate=True, output_scores=True
                )
                texts = proc.batch_decode(out.sequences, skip_special_tokens=True)
                
                # Approximate sequence logprobs for sampling
                for i, t in enumerate(texts):
                    if t.strip():
                        # Use sequence score if available, else approximate with 0.0
                        score = out.sequences_scores[i].item() if hasattr(out, "sequences_scores") else -1.0
                        candidates.append(Candidate(text=t.strip(), meta={"algo": "nucleus", "temp": temp, "logprob": score}))

        # Diverse Beam Search (DBS)
        if "dbs" in config.algos:
            try:
                out = model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids,
                    num_beams=config.beam_size, num_beam_groups=config.beam_size,
                    diversity_penalty=1.0, num_return_sequences=config.beam_size,
                    return_dict_in_generate=True, output_scores=True,
                    custom_generate="transformers-community/group-beam-search",
                    trust_remote_code=True
                )
                texts = proc.batch_decode(out.sequences, skip_special_tokens=True)
                for i, t in enumerate(texts):
                    if t.strip():
                        logprob = out.sequences_scores[i].item() if hasattr(out, "sequences_scores") else 0.0
                        candidates.append(Candidate(text=t.strip(), meta={"algo": "dbs", "logprob": logprob}))
            except Exception as e:
                warnings.warn(f"DBS failed: {e}")

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Folder with audio files")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--model_type", choices=["whisper", "nemo"], default="whisper")
    parser.add_argument("--ckpt", default="openai/whisper-large-v3")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--algos", default="beam,dbs,nucleus", help="Comma-separated: beam,dbs,nucleus")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="Max files to process")
    args = parser.parse_args()

    audio_files = glob.glob(os.path.join(args.dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(AUDIO_EXTS)]
    
    proc, model = load_whisper(args.ckpt)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, path in enumerate(audio_files):
            if args.limit > 0 and i >= args.limit:
                break
            uid = os.path.splitext(os.path.basename(path))[0]
            try:
                audio, sr = load_audio(path)
                cands = generate_whisper_candidates(audio, sr, proc, model, args)
                
                # Deduplicate based on NORMALIZED text to avoid casing/punctuation duplicates
                seen_norm = set()
                unique_cands = []
                for c in cands:
                    norm_text = norm_transform(c.text)
                    if norm_text and norm_text not in seen_norm:
                        seen_norm.add(norm_text)
                        unique_cands.append({"text": c.text, "meta": c.meta})
                        
                rec = {"utt_id": uid, "audio_path": path, "candidates": unique_cands}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
            except Exception as e:
                warnings.warn(f"Failed processing {path}: {e}")

if __name__ == "__main__":
    main()