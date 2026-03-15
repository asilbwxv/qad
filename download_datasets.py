import os
import json
import argparse
from pathlib import Path

def download_librispeech(output_dir, split="test"):
    """Downloads LibriSpeech via HuggingFace and extracts audio and references."""
    from datasets import load_dataset
    import soundfile as sf

    print(f"Downloading LibriSpeech ({split}.clean) ...")
    # LibriSpeech has 'clean' and 'other' configs. 
    ds = load_dataset("librispeech_asr", "clean", split=split)
    
    audio_dir = os.path.join(output_dir, "librispeech", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    refs_path = os.path.join(output_dir, "librispeech", "references.jsonl")
    
    with open(refs_path, "w", encoding="utf-8") as f:
        for item in ds:
            utt_id = item["id"]
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item["text"].lower()
            
            # Save audio
            audio_path = os.path.join(audio_dir, f"{utt_id}.wav")
            sf.write(audio_path, audio_array, sr)
            
            # Save reference
            ref_data = {"utt_id": utt_id, "audio_path": audio_path, "ref": text}
            f.write(json.dumps(ref_data) + "\n")
            
    print(f"✅ LibriSpeech downloaded to {audio_dir}")
    print(f"✅ References saved to {refs_path}")

def download_coco_karpathy(output_dir):
    """Downloads a subset of the MS COCO Karpathy split via HuggingFace."""
    from datasets import load_dataset
    
    print("Downloading MS COCO (Karpathy validation split) ...")
    # We use a popular HF repo that hosts the Karpathy split directly
    ds = load_dataset("HuggingFaceM4/COCO", split="validation")
    
    img_dir = os.path.join(output_dir, "coco", "images")
    os.makedirs(img_dir, exist_ok=True)
    
    refs_path = os.path.join(output_dir, "coco", "references.jsonl")
    
    # Just grab the first 500 for a quick evaluation set, or remove [:500] for all
    with open(refs_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds.select(range(500))):
            img_id = str(item.get("cocoid", i))
            image = item["image"]
            sentences = item["sentences"] # Multi-reference list
            
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            image.convert("RGB").save(img_path)
            
            ref_data = {"image_id": img_id, "image_path": img_path, "refs": sentences}
            f.write(json.dumps(ref_data) + "\n")

    print(f"✅ MS COCO downloaded to {img_dir}")
    print(f"✅ References saved to {refs_path}")

def main():
    parser = argparse.ArgumentParser(description="Download ASR and Captioning Datasets")
    parser.add_argument("--task", required=True, choices=["asr", "caption", "all"])
    parser.add_argument("--out", default="./data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.task in ["asr", "all"]:
        download_librispeech(args.out)
    if args.task in ["caption", "all"]:
        download_coco_karpathy(args.out)

if __name__ == "__main__":
    main()
