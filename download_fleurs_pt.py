import os
import json
import soundfile as sf
from datasets import load_dataset

def download_fleurs_pt(output_dir="./data"):
    print("Downloading FLEURS Portuguese (pt_br) ASR split...")
    ds = load_dataset("google/fleurs", "pt_br", split="test")
    
    audio_dir = os.path.join(output_dir, "fleurs_pt", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    refs_path = os.path.join(output_dir, "fleurs_pt", "references.jsonl")
    
    with open(refs_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            utt_id = f"pt_{item.get('id', i)}"
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item["transcription"].lower()
            
            audio_path = os.path.join(audio_dir, f"{utt_id}.wav")
            sf.write(audio_path, audio_array, sr)
            
            ref_data = {"utt_id": utt_id, "audio_path": audio_path, "ref": text}
            f.write(json.dumps(ref_data) + "\n")
            
    print(f"✅ FLEURS Portuguese dataset downloaded ({len(ds)} files) to {audio_dir}")

if __name__ == "__main__":
    download_fleurs_pt()
