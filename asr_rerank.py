import os
import json
import argparse
import numpy as np
from pathlib import Path
from metrics import compute_wer, compute_cer, SemaScoreWrapper, NoRefERWrapper

def compute_probability_weighted_risk(candidates, metric="wer", temperature=1.0, semascore_scorer=None):
    """
    Computes probability-weighted expected risk for MBR:
    E[Risk(y_i)] = sum_j P(y_j) * Loss(y_i, y_j)
    where P(y_j) = softmax(logprob_j / temperature).
    """
    N = len(candidates)
    if N <= 1:
        return [0.0] * N

    # Extract logprobs and compute softmax weights
    logprobs = np.array([c["meta"].get("logprob", -10.0) for c in candidates])
    # Prevent numerical overflow
    max_lp = np.max(logprobs)
    weights = np.exp((logprobs - max_lp) / temperature)
    weights = weights / np.sum(weights)

    scores = []
    for i in range(N):
        total_risk = 0.0
        for j in range(N):
            if metric == "wer":
                loss = compute_wer(candidates[i]["text"], candidates[j]["text"])
            elif metric == "cer":
                loss = compute_cer(candidates[i]["text"], candidates[j]["text"])
            elif metric == "semascore" and semascore_scorer:
                # SeMaScore is utility, so loss is 1 - utility
                loss = 1.0 - semascore_scorer.score(candidates[i]["text"], candidates[j]["text"])
            else:
                loss = compute_wer(candidates[i]["text"], candidates[j]["text"])
                
            total_risk += weights[j] * loss
            
        scores.append(total_risk)
    return scores

def main():
    parser = argparse.ArgumentParser(description="QAD/MBR Reranking for ASR")
    parser.add_argument("--inp", required=True, help="Input JSONL from asr_gen.py")
    parser.add_argument("--out", required=True, help="Output JSONL with final chosen transcripts")
    parser.add_argument("--algo", required=True, choices=["map", "fixed_rr", "tuned_rr", "mbr", "two_stage_mbr"])
    parser.add_argument("--mbr_metric", default="wer", choices=["wer", "cer", "semascore"])
    parser.add_argument("--qe_metric", default="norefer", choices=["norefer"])
    parser.add_argument("--tune_weights", type=str, help="Comma-separated weights matching meta features for tuned_rr")
    parser.add_argument("--tune_features", type=str, help="Comma-separated meta feature names for tuned_rr")
    parser.add_argument("--prune_k", type=int, default=5, help="Number of candidates to keep for stage 2 of two_stage_mbr")
    args = parser.parse_args()

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
                # THE FIX: MAP must only pick from standard "beam" candidates, not DBS hallucinations
                beam_cands = [i for i, c in enumerate(cands) if c["meta"].get("algo") == "beam"]
                if beam_cands:
                    chosen_idx = max(beam_cands, key=lambda i: cands[i]["meta"].get("logprob", -float('inf')))
                else:
                    chosen_idx = max(range(len(cands)), key=lambda i: cands[i]["meta"].get("logprob", -float('inf')))

                
            elif args.algo == "fixed_rr":
                # Quality-Aware Decoding via NoRefER
                texts = [c["text"] for c in cands]
                qe_scores = norefer_scorer.score(texts)
                chosen_idx = int(np.argmax(qe_scores))
                
            elif args.algo == "mbr":
                # Probability-weighted Minimum Bayes Risk
                risk_scores = compute_probability_weighted_risk(cands, metric=args.mbr_metric, semascore_scorer=semascore_scorer)
                chosen_idx = int(np.argmin(risk_scores)) # Minimize expected risk
                    
            elif args.algo == "two_stage_mbr":
                # 1. Prune candidate pool using NoRefER or logprob
                texts = [c["text"] for c in cands]
                qe_scores = norefer_scorer.score(texts)
                
                # Keep top-K candidates
                top_k_indices = np.argsort(qe_scores)[-args.prune_k:]
                pruned_cands = [cands[i] for i in top_k_indices]
                
                # 2. Probability-Weighted MBR on the pruned subset
                risk_scores = compute_probability_weighted_risk(pruned_cands, metric=args.mbr_metric, semascore_scorer=semascore_scorer)
                best_in_pruned = int(np.argmin(risk_scores))
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
