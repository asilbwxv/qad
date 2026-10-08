import json
import argparse
import numpy as np
from scipy.optimize import minimize
from metrics import compute_wer, compute_cer

def load_data(jsonl_path, refs_path):
    # Load ground truth
    refs = {}
    with open(refs_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            refs[data['utt_id']] = data['ref']
            
    # Load candidates
    dataset = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['utt_id'] in refs:
                dataset.append({
                    'utt_id': data['utt_id'],
                    'candidates': data['candidates'],
                    'ref': refs[data['utt_id']]
                })
    return dataset

def objective_function(weights, dataset, features, target_metric="wer"):
    """
    Given a set of weights, compute the combined score for each candidate,
    pick the best candidate per utterance, and compute the corpus-level target metric.
    """
    total_error = 0.0
    total_refs = 0
    
    for item in dataset:
        cands = item['candidates']
        ref = item['ref']
        
        best_score = -float('inf')
        best_hyp = ""
        
        for cand in cands:
            # Combine features linearly: w1*f1 + w2*f2 ...
            score = sum(w * cand['meta'].get(feat, 0.0) for w, feat in zip(weights, features))
            if score > best_score:
                best_score = score
                best_hyp = cand['text']
                
        if target_metric == "wer":
            # For WER, we want to MINIMIZE the objective, so we just sum errors
            total_error += compute_wer(best_hyp, ref)
            total_refs += 1 # Simplified average. True corpus WER aggregates edits/words.
        # Add CIDEr logic here for captioning...
            
    return total_error / max(1, total_refs)

def main():
    parser = argparse.ArgumentParser(description="Tune weights for T-RR QAD")
    parser.add_argument("--candidates", required=True, help="JSONL of validation candidates with meta scores")
    parser.add_argument("--refs", required=True, help="JSONL of validation references")
    parser.add_argument("--features", required=True, help="Comma-separated features to tune (e.g., avg_logprob,norefer)")
    parser.add_argument("--target", default="wer", choices=["wer", "cer", "cider"], help="Metric to optimize")
    args = parser.parse_args()

    features = args.features.split(",")
    dataset = load_data(args.candidates, args.refs)
    
    print(f"Loaded {len(dataset)} validation items.")
    print(f"Tuning weights for features: {features} targeting {args.target}")

    # Initial guess: equal weighting
    initial_weights = np.ones(len(features)) / len(features)
    
    # We use Powell optimization as it works well for non-differentiable objectives (like discrete rank selection)
    result = minimize(
        objective_function, 
        initial_weights, 
        args=(dataset, features, args.target),
        method='Powell',
        options={'disp': True}
    )
    
    print("\n--- Tuning Complete ---")
    print(f"Best Objective ({args.target}): {result.fun:.4f}")
    print("Optimal Weights:")
    for feat, w in zip(features, result.x):
        print(f"  {feat}: {w:.4f}")

if __name__ == "__main__":
    main()
