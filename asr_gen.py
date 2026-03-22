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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

@dataclass
class Candidate:
    text: str
    meta: Dict

def load_audio(path: str) -> Tuple[object, int]:
    audio, sr = sf.read(path)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    return audio, sr

# ==========================================
# WHISPER GENERATOR
# ==========================================
def load_whisper(ckpt: str):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    proc = WhisperProcessor.from_pretrained(ckpt)
    # Load in FP16 to save massive amounts of VRAM
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return proc, model

def generate_whisper_candidates(audio, sr, proc, model, config) -> List[Candidate]:
    """Generates a diverse set of candidates using Whisper."""
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(DEVICE, dtype=torch.float16)
    forced_decoder_ids = proc.get_decoder_prompt_ids(language=config.lang, task=config.task)
    
    candidates = []
    
    with torch.inference_mode():
        # Standard Beam Search
        if "beam" in config.algos:
            out = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids,
                num_beams=config.beam_size, num_return_sequences=config.beam_size,
                output_scores=True, return_dict_in_generate=True
            )
            texts = proc.batch_decode(out.sequences, skip_special_tokens=True)
            # Simplified logprob mapping for demonstration
            for i, t in enumerate(texts):
                if t.strip():
                    candidates.append(Candidate(text=t.strip(), meta={"algo": "beam", "logprob": out.sequences_scores[i].item()}))

        # Nucleus Sampling
        if "nucleus" in config.algos:
            out = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids,
                do_sample=True, top_p=config.top_p, temperature=config.temperature,
                num_beams=1, num_return_sequences=config.n_samples,
            )
            texts = proc.batch_decode(out, skip_special_tokens=True)
            for t in texts:
                if t.strip():
                    candidates.append(Candidate(text=t.strip(), meta={"algo": "nucleus"}))

    return candidates

# ==========================================
# NEMO GENERATOR (CONFORMER/CANARY)
# ==========================================
def load_nemo(ckpt: str):
    import nemo.collections.asr as nemo_asr
    # Load ASR model (e.g., stt_en_conformer_ctc_large)
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=ckpt).to(DEVICE).eval()
    return model

def generate_nemo_candidates(audio_path, model, config) -> List[Candidate]:
    """
    Scaffolding for NeMo generation. NeMo decoding strategies (like N-best beam search)
    require altering the decoding config before transcription.
    """
    from omegaconf import open_dict
    candidates = []
    
    # Temporarily adjust decoding strategy for N-best output
    with open_dict(model.cfg.decoding):
        model.cfg.decoding.strategy = "beam"
        model.cfg.decoding.beam.beam_size = config.beam_size
        model.cfg.decoding.beam.return_best_hypothesis = False # Returns N-best
        
    # NeMo transcribe returns lists of hypotheses
    hypotheses = model.transcribe(paths2audio_files=[audio_path], batch_size=1)[0]
    
    for idx, hyp in enumerate(hypotheses):
        # hyp typically contains text, score, alignents
        candidates.append(Candidate(
            text=hyp.text, 
            meta={"algo": "nemo_beam", "logprob": hyp.score}
        ))
        
    return candidates

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Folder with audio files")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--model_type", choices=["whisper", "nemo"], default="whisper")
    parser.add_argument("--ckpt", default="openai/whisper-large-v3")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--algos", default="beam,nucleus", help="Comma-separated: beam,nucleus")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0, help="Max files to process")
    args = parser.parse_args()

    audio_files = glob.glob(os.path.join(args.dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(AUDIO_EXTS)]
    
    # Load requested model
    if args.model_type == "whisper":
        proc, model = load_whisper(args.ckpt)
    else:
        model = load_nemo(args.ckpt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, path in enumerate(audio_files):
            if args.limit > 0 and i >= args.limit:
                break
            uid = os.path.splitext(os.path.basename(path))[0]
            try:
                if args.model_type == "whisper":
                    audio, sr = load_audio(path)
                    cands = generate_whisper_candidates(audio, sr, proc, model, args)
                else:
                    # NeMo usually handles file loading internally
                    cands = generate_nemo_candidates(path, model, args)
                
                # Deduplicate by text while preserving metadata
                seen = set()
                unique_cands = []
                for c in cands:
                    if c.text not in seen:
                        seen.add(c.text)
                        unique_cands.append({"text": c.text, "meta": c.meta})
                        
                rec = {"utt_id": uid, "audio_path": path, "candidates": unique_cands}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
            except Exception as e:
                warnings.warn(f"Failed processing {path}: {e}")

if __name__ == "__main__":
    main()
