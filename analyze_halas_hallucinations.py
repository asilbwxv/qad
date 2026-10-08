import json
import argparse
from metrics import compute_wer

def analyze_hallucinations(final_jsonl, refs_jsonl):
    refs = {}
    halas_labels = {}
    with open(refs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            refs[data["utt_id"]] = data["ref"]
            halas_labels[data["utt_id"]] = data.get("halas_human_label", "")
            
    total_files = 0
    severe_hallucinations = 0
    
    with open(final_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            uid = data["utt_id"]
            hyp = data["final"]
            
            if uid in refs:
                total_files += 1
                ref = refs[uid]
                wer = compute_wer(hyp, ref)
                
                # Detect massive divergence on known hallucination files
                if wer > 1.0:
                    severe_hallucinations += 1

    hallucination_rate = (severe_hallucinations / max(1, total_files)) * 100
    print(f"\n📊 Evaluated {total_files} HALAS audio files.")
    print(f"🚨 Severe Hallucinations Survived: {severe_hallucinations}")
    print(f"📈 Hallucination Rate: {hallucination_rate:.2f}%\n")
    return hallucination_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True, help="Path to asr_final.jsonl")
    parser.add_argument("--refs", required=True, help="Path to references.jsonl")
    args = parser.parse_args()
    analyze_hallucinations(args.final, args.refs)
