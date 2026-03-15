import os
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from metrics import CLIPScoreWrapper, caption_mbr_multiref

def main():
    parser = argparse.ArgumentParser(description="QAD/MBR Reranking for Image Captioning")
    parser.add_argument("--inp", required=True, help="Input JSONL from caption_gen.py")
    parser.add_argument("--out", required=True, help="Output JSONL with final chosen captions")
    parser.add_argument("--algo", required=True, choices=["map", "fixed_rr", "mbr", "two_stage_mbr"])
    
    # MBR parameters
    parser.add_argument("--mbr_metric", default="cider", choices=["cider", "spice"])
    
    # QAD / Two-stage parameters
    parser.add_argument("--prune_k", type=int, default=5, help="Number of candidates to keep for stage 2")
    
    args = parser.parse_args()

    # Initialize CLIPScore if we are using referenceless QE grounding
    clip_scorer = CLIPScoreWrapper() if args.algo in ["fixed_rr", "two_stage_mbr"] else None

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            rec = json.loads(line)
            cands = rec.get("candidates", [])
            img_path = rec.get("image_path")
            
            if not cands:
                rec["final"] = ""
                rec["meta"] = {"reason": "no_candidates"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            chosen_idx = 0
            texts = [c["text"] for c in cands]
            
            if args.algo == "map":
                # Assuming first is best if beam search was used
                chosen_idx = 0 
                
            elif args.algo == "fixed_rr":
                # Rank purely by visual grounding (CLIPScore)
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Note: CLIPScoreWrapper expects a tensor. You might need torchvision.transforms here 
                    # depending on the exact PyTorchMetrics implementation details. 
                    # Assuming clip_scorer.metric handles PIL natively or via a transform pipeline.
                    import torchvision.transforms as transforms
                    tensor_img = transforms.ToTensor()(img)
                    
                    clip_scores = clip_scorer.score(tensor_img, texts)
                    chosen_idx = int(np.argmax(clip_scores))
                except Exception as e:
                    print(f"Error computing CLIP for {img_path}: {e}")
                    chosen_idx = 0
                
            elif args.algo == "mbr":
                # Multi-reference consensus MBR using pycocoevalcap
                mbr_scores = caption_mbr_multiref(texts, metric_name=args.mbr_metric)
                chosen_idx = int(np.argmax(mbr_scores))
                    
            elif args.algo == "two_stage_mbr":
                # 1. Prune with visual grounding (CLIP)
                try:
                    img = Image.open(img_path).convert("RGB")
                    import torchvision.transforms as transforms
                    tensor_img = transforms.ToTensor()(img)
                    
                    clip_scores = clip_scorer.score(tensor_img, texts)
                    top_k_indices = np.argsort(clip_scores)[-args.prune_k:]
                    
                    pruned_texts = [texts[i] for i in top_k_indices]
                    
                    # 2. Multi-ref MBR on the visually grounded subset
                    mbr_scores = caption_mbr_multiref(pruned_texts, metric_name=args.mbr_metric)
                    best_in_pruned = int(np.argmax(mbr_scores))
                    
                    chosen_idx = int(top_k_indices[best_in_pruned])
                except Exception as e:
                    print(f"Error in two-stage MBR for {img_path}: {e}")
                    chosen_idx = 0

            out_rec = {
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "final": cands[chosen_idx]["text"],
                "meta": {
                    "chosen_index": chosen_idx,
                    "algo_used": args.algo
                }
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
