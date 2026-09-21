import os
import json
import soundfile as sf
from datasets import load_dataset

def download_halas(output_dir="./data"):
    print("Downloading FULL HALAS Test Split (745 Examples)...")
    
    halas_ds_test = load_dataset("MatBar99/HALAS", split="test")
    
    halas_data = {}
    for item in halas_ds_test:
        raw_aid = str(item.get("audio_id", "")).strip().replace(".wav", "")
        clean_aid = raw_aid.replace("-", "_")
        halas_data[clean_aid] = item
        halas_data[raw_aid] = item
        
    audio_dir = os.path.join(output_dir, "halas", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    refs_path = os.path.join(output_dir, "halas", "references.jsonl")
    
    print(f"Loaded {len(halas_data)} HALAS test label mappings.")
    print("Streaming Earnings-22 audio to find exact matches...")
    
    found_count = 0
    target_count = len(halas_ds_test)
    
    with open(refs_path, "w", encoding="utf-8") as f:
        for split_name in ["test", "train", "validation"]:
            try:
                e22_ds = load_dataset("distil-whisper/earnings22", "chunked", split=split_name, streaming=True)
            except Exception:
                continue
                
            print(f"\nScanning Earnings-22 '{split_name}' split...")
            for i, item in enumerate(e22_ds):
                if found_count >= target_count:
                    break
                    
                file_id = str(item.get("file_id", "")).strip()
                seg_id = str(item.get("segment_id", "")).strip()
                
                possible_keys = [
                    f"{seg_id}_{file_id}",
                    f"{file_id}_{seg_id}",
                    f"{seg_id}-{file_id}",
                    f"{file_id}-{seg_id}"
                ]
                
                matched_key = None
                for pk in possible_keys:
                    if pk in halas_data:
                        matched_key = pk
                        break
                        
                if matched_key:
                    halas_item = halas_data[matched_key]
                    audio_array = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    
                    ref_text = halas_item.get("corrected_reference_text", "")
                    if not ref_text:
                        ref_text = halas_item.get("e22_reference_text", "")
                        
                    human_label = halas_item.get("whisper_large_v2_label", "Unknown")
                    
                    safe_name = f"{file_id}_{seg_id}"
                    out_audio_path = os.path.join(audio_dir, f"{safe_name}.wav")
                    sf.write(out_audio_path, audio_array, sr)
                    
                    ref_data = {
                        "utt_id": safe_name, 
                        "audio_path": out_audio_path, 
                        "ref": ref_text.lower(),
                        "halas_human_label": human_label
                    }
                    f.write(json.dumps(ref_data) + "\n")
                    f.flush()
                    
                    for pk in possible_keys:
                        halas_data.pop(pk, None)
                        
                    found_count += 1
                    print(f"✅ Saved: {found_count}/{target_count} audio files...", end="\r")

            if found_count >= target_count:
                break
                    
    print(f"\n✅ Full HALAS dataset audio saved ({found_count} files) to {audio_dir}")

if __name__ == "__main__":
    download_halas()
