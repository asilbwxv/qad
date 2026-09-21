import json
import argparse
from metrics import compute_wer

def inspect_failures(candidates_file, final_file, refs_file):
    print("\n" + "="*60)
    print("🔍 EXTRACTING QUALITATIVE FAILURE CASES")
    print("="*60)
    
    refs = {}
    with open(refs_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            uid = data.get("utt_id") or data.get("image_id")
            refs[uid] = data.get("ref") or data.get("refs")
            
    cands_map = {}
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            uid = data.get("utt_id") or data.get("image_id")
            cands_map[uid] = [c["text"] for c in data["candidates"]]
            
    failures = []
    with open(final_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            uid = data.get("utt_id") or data.get("image_id")
            hyp = data["final"]
            ref = refs.get(uid, "")
            
            if isinstance(ref, str):
                wer = compute_wer(hyp, ref)
                if wer > 0.8: # High error failure case
                    failures.append({
                        "id": uid,
                        "ground_truth": ref,
                        "selected_output": hyp,
                        "candidates": cands_map.get(uid, [])[:3]
                    })
                    
    print(f"Found {len(failures)} notable high-error cases. Displaying top 3:")
    for item in failures[:3]:
        print(f"\nID: {item['id']}")
        print(f"  [Ground Truth]     : {item['ground_truth']}")
        print(f"  [Selected Output]  : {item['selected_output']}")
        print(f"  [Candidate Pool]   : {item['candidates']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--refs", required=True)
    args = parser.parse_args()
    inspect_failures(args.candidates, args.final, args.refs)
