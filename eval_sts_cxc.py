import json
import argparse
import numpy as np
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    exit(1)

def extract_caption_strings(refs_entry):
    """Extracts a flat list of string captions from COCO refs_entry."""
    sentences = []
    if isinstance(refs_entry, dict):
        if "raw" in refs_entry:
            val = refs_entry["raw"]
            if isinstance(val, list):
                sentences.extend([str(x) for x in val])
            else:
                sentences.append(str(val))
        else:
            for v in refs_entry.values():
                if isinstance(v, list):
                    sentences.extend([str(x) for x in v])
                else:
                    sentences.append(str(v))
    elif isinstance(refs_entry, list):
        for s in refs_entry:
            if isinstance(s, dict) and "raw" in s:
                sentences.append(str(s["raw"]))
            elif isinstance(s, str):
                sentences.append(s)
            elif isinstance(s, dict):
                sentences.append(str(list(s.values())[0]))
    elif isinstance(refs_entry, str):
        sentences.append(refs_entry)
    return sentences

def evaluate_sts(final_jsonl, refs_jsonl):
    print(f"\nLoading Sentence-BERT (CxC Human-Preference Proxy)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    refs = {}
    with open(refs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            raw_id = str(data.get("image_id", ""))
            clean_id = ''.join(filter(str.isdigit, raw_id))
            clean_id = str(int(clean_id)) if clean_id else raw_id
            
            sentences = extract_caption_strings(data.get("refs", []))
            refs[clean_id] = sentences
            
    scores = []
    with open(final_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            raw_id = str(data["image_id"])
            clean_id = ''.join(filter(str.isdigit, raw_id))
            clean_id = str(int(clean_id)) if clean_id else raw_id
            
            hyp = data["final"]
            if clean_id in refs and hyp.strip() and refs[clean_id]:
                hyp_emb = model.encode(hyp, convert_to_tensor=True, device=device)
                ref_embs = model.encode(refs[clean_id], convert_to_tensor=True, device=device)
                
                cosine_scores = util.cos_sim(hyp_emb, ref_embs)
                best_score = float(torch.max(cosine_scores).cpu().numpy())
                scores.append(best_score)
                
    avg_sts = np.mean(scores) * 100 if scores else 0.0
    print(f"✅ Average Human-Aligned STS Score: {avg_sts:.2f}/100\n")
    return avg_sts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True, help="Path to caption_final.jsonl")
    parser.add_argument("--refs", required=True, help="Path to references.jsonl")
    args = parser.parse_args()
    evaluate_sts(args.final, args.refs)
