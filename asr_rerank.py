import os
import json
import argparse
import numpy as np
from pathlib import Path
from metrics import compute_wer, compute_cer, SemaScoreWrapper, NoRefERWrapper

def compute_expected_risk(candidates, metric="wer", semascore_scorer=None):
    """
    Computes pairwise expected risk/utility for MBR.
    Complexity: O(N^2)
    """
    N = len(candidates)
    if N <= 1:
        return [0.0] * N

    scores = []
    for i in range(N):
        total_score = 0.0
        for j in range(N):
            if i == j:
                continue
            
            # Compute distance or utility between candidate i and candidate j
            if metric == "wer":
                total_score += compute_wer(candidates[i]["text"], candidates[j]["text"])
            elif metric == "cer":
                total_score += compute_cer(candidates[i]["text"], candidates[j]["text"])
            elif metric == "semascore" and semascore_scorer:
                # SeMaScore is a utility (higher is better), so we negate it to treat it as a "risk" to minimize,
                # OR we just keep it as utility and maximize later. We'll treat it as utility here.
                total_score += semascore_scorer.score(candidates[i]["text"], candidates[j]["text"])
                
        scores.append(total_score / (N - 1))
    return scores

def main():
    parser = argparse.ArgumentParser(description="QAD/MBR Reranking for ASR")
    parser.add_argument("--inp", required=True, help="Input JSONL from asr_gen.py")
    parser.add_argument("--out", required=True, help="Output JSONL with final chosen transcripts")
    parser.add_argument("--algo", required=True, choices=["map", "fixed_rr", "tuned_rr", "mbr", "two_stage_mbr"])
    
    # MBR parameters
    parser.add_argument("--mbr_metric", default="wer", choices=["wer", "cer", "semascore"])
    
    # QAD / Reranking parameters
    parser.add_argument("--qe_metric", default="norefer", choices=["norefer"])
    parser.add_argument("--tune_weights", type=str, help="Comma-separated weights matching meta features for tuned_rr")
    parser.add_argument("--tune_features", type=str, help="Comma-separated meta feature names for tuned_rr")
    
    # Two-stage parameters
    parser.add_argument("--prune_k", type=int, default=5, help="Number of candidates to keep for stage 2 of two_stage_mbr")
    
    args = parser.parse_args()

    # Initialize heavy scorers only if needed
    semascore_scorer = SemaScoreWrapper() if args.mbr_metric == "semascore" else None
    norefer_scorer = NoRefERWrapper() if args.qe_metric == "norefer" and args.algo in ["fixed_rr", "two_stage_mbr"] else None

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            rec = json.loads(line)
            cands = rec.get("candidates", [])
            
            if not cands:
                rec["final"] = ""
                rec["meta"] = {"reason": "no_candidates"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            chosen_idx = 0
            
            if args.algo == "map":
                # Baseline: Pick highest logprob
                chosen_idx = max(range(len(cands)), key=lambda i: cands[i]["meta"].get("logprob", -float('inf')))
                
            elif args.algo == "fixed_rr":
                # Quality-Aware Decoding: Rank by a single Quality Estimator
                texts = [c["text"] for c in cands]
                qe_scores = norefer_scorer.score(texts)
                chosen_idx = int(np.argmax(qe_scores))
                
            elif args.algo == "tuned_rr":
                # Quality-Aware Decoding: Linear combination of features
                weights = [float(w) for w in args.tune_weights.split(",")]
                features = args.tune_features.split(",")
                
                best_score = -float('inf')
                for i, c in enumerate(cands):
                    score = sum(w * c["meta"].get(feat, 0.0) for w, feat in zip(weights, features))
                    if score > best_score:
                        best_score = score
                        chosen_idx = i
                        
            elif args.algo == "mbr":
                # Minimum Bayes Risk: O(N^2) expected utility
                scores = compute_expected_risk(cands, args.mbr_metric, semascore_scorer)
                if args.mbr_metric in ["wer", "cer"]:
                    chosen_idx = int(np.argmin(scores)) # Minimize error
                else:
                    chosen_idx = int(np.argmax(scores)) # Maximize utility (SeMaScore)
                    
            elif args.algo == "two_stage_mbr":
                # 1. Prune with fast QE
                texts = [c["text"] for c in cands]
                qe_scores = norefer_scorer.score(texts)
                
                # Get top K indices
                top_k_indices = np.argsort(qe_scores)[-args.prune_k:]
                pruned_cands = [cands[i] for i in top_k_indices]
                
                # 2. MBR on pruned set
                mbr_scores = compute_expected_risk(pruned_cands, args.mbr_metric, semascore_scorer)
                
                if args.mbr_metric in ["wer", "cer"]:
                    best_in_pruned = int(np.argmin(mbr_scores))
                else:
                    best_in_pruned = int(np.argmax(mbr_scores))
                    
                chosen_idx = int(top_k_indices[best_in_pruned])

            out_rec = {
                "utt_id": rec["utt_id"],
                "audio_path": rec["audio_path"],
                "final": cands[chosen_idx]["text"],
                "meta": {
                    "chosen_index": chosen_idx,
                    "algo_used": args.algo
                }
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
