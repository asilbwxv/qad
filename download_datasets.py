import os
import json
import argparse

def download_librispeech(output_dir, split="test"):
    """Downloads the NOISY LibriSpeech split."""
    from datasets import load_dataset
    import soundfile as sf

    print(f"Downloading LibriSpeech ({split}.other) - Noisy Speech ...")
    # Swapped from "clean" to "other"
    ds = load_dataset("librispeech_asr", "other", split=split)
    
    audio_dir = os.path.join(output_dir, "librispeech_noisy", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    refs_path = os.path.join(output_dir, "librispeech_noisy", "references.jsonl")
    
    with open(refs_path, "w", encoding="utf-8") as f:
        for item in ds:
            utt_id = str(item["id"])
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item["text"].lower()
            
            audio_path = os.path.join(audio_dir, f"{utt_id}.wav")
            sf.write(audio_path, audio_array, sr)
            
            ref_data = {"utt_id": utt_id, "audio_path": audio_path, "ref": text}
            f.write(json.dumps(ref_data) + "\n")
            
    print(f"✅ Noisy LibriSpeech downloaded to {audio_dir}")

def download_fleurs_pt(output_dir, split="test"):
    """Downloads Google's FLEURS dataset for Portuguese."""
    from datasets import load_dataset
    import soundfile as sf

    print(f"Downloading FLEURS (pt_pt) - Portuguese Speech ...")
    # Pulling Portuguese (Portugal) dataset
    ds = load_dataset("google/fleurs", "pt_pt", split=split)
    
    audio_dir = os.path.join(output_dir, "fleurs_pt", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    refs_path = os.path.join(output_dir, "fleurs_pt", "references.jsonl")
    
    with open(refs_path, "w", encoding="utf-8") as f:
        for item in ds:
            utt_id = f"fleurs_pt_{item['id']}"
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            # FLEURS uses 'transcription' for the text
            text = item["transcription"].lower()
            
            audio_path = os.path.join(audio_dir, f"{utt_id}.wav")
            sf.write(audio_path, audio_array, sr)
            
            ref_data = {"utt_id": utt_id, "audio_path": audio_path, "ref": text}
            f.write(json.dumps(ref_data, ensure_ascii=False) + "\n")
            
    print(f"✅ FLEURS (PT) downloaded to {audio_dir}")

def download_coco_karpathy(output_dir):
    """Downloads a subset of the MS COCO Karpathy split."""
    from datasets import load_dataset
    
    print("Downloading MS COCO (Karpathy validation split) ...")
    ds = load_dataset("HuggingFaceM4/COCO", split="validation")
    
    img_dir = os.path.join(output_dir, "coco", "images")
    os.makedirs(img_dir, exist_ok=True)
    refs_path = os.path.join(output_dir, "coco", "references.jsonl")
    
    with open(refs_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds.select(range(500))):
            img_id = str(item.get("cocoid", i))
            image = item["image"]
            sentences = item["sentences"]
            
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            image.convert("RGB").save(img_path)
            
            ref_data = {"image_id": img_id, "image_path": img_path, "refs": sentences}
            f.write(json.dumps(ref_data) + "\n")

    print(f"✅ MS COCO downloaded to {img_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download ASR and Captioning Datasets")
    parser.add_argument("--task", required=True, choices=["asr", "caption", "all"])
    parser.add_argument("--out", default="./data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.task in ["asr", "all"]:
        download_librispeech(args.out)
        download_fleurs_pt(args.out)  # <--- Added Portuguese dataset
    if args.task in ["caption", "all"]:
        download_coco_karpathy(args.out)

if __name__ == "__main__":
    main()
