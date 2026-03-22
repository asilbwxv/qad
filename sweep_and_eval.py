import os
import json
import subprocess
import argparse
import itertools
from metrics import compute_wer # Assuming this is in your metrics.py

def run_pipeline_step(task, input_dir, output_dir, gen_algos, beam_size, rerank_algo, mbr_metric, limit): 
    """Calls the run_pipeline.py master script with specific parameters."""
    cmd = [
        "python", "run_pipeline.py",
        "--task", task,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--gen_algos", gen_algos,
        "--beam_size", str(beam_size),
        "--rerank_algo", rerank_algo,
        "--mbr_metric", mbr_metric,
        "--limit", str(limit)
    ]
    print(f"\n=> Running Sweep Configuration: Beam={beam_size}, Rerank={rerank_algo}, Metric={mbr_metric}")
    subprocess.run(cmd, check=True)

def evaluate_asr(final_jsonl, refs_jsonl):
    """Computes corpus-level WER by comparing pipeline output to references."""
    refs = {}
    with open(refs_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            refs[data["utt_id"]] = data["ref"]
            
    total_wer = 0.0
    count = 0
    
    with open(final_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = data["utt_id"]
            hyp = data["final"]
            if uid in refs:
                total_wer += compute_wer(hyp, refs[uid])
                count += 1
                
    return (total_wer / count) * 100 if count > 0 else 0.0

def evaluate_caption(final_jsonl, refs_jsonl):
    """Computes corpus-level captioning metrics using pycocoevalcap."""
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
    except ImportError:
        print("❌ pycocoevalcap not found. Please install it (pip install pycocoevalcap) to evaluate captioning.")
        return 0.0

    # 1. Load Ground Truths (gts)
    gts = {}
    with open(refs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            raw_id = str(data.get("image_id", ""))
            
            # Clean ID extraction
            if "_" in raw_id:
                clean_id = str(int(raw_id.split("_")[-1]))
            else:
                clean_id = ''.join(filter(str.isdigit, raw_id))
                clean_id = str(int(clean_id)) if clean_id else raw_id
                
            # Extract the actual text sentence
            caption_text = str(data["refs"]["raw"])
            
            # Group all captions under the same image ID (don't overwrite!)
            if clean_id not in gts:
                gts[clean_id] = []
            gts[clean_id].append(caption_text)

    # 2. Load Predictions (res)
    res = {}
    with open(final_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            raw_id = str(data["image_id"])
            
            # Clean ID extraction
            if "_" in raw_id:
                clean_id = str(int(raw_id.split("_")[-1]))
            else:
                clean_id = ''.join(filter(str.isdigit, raw_id))
                clean_id = str(int(clean_id)) if clean_id else raw_id
                
            # THE FIX: Pass just the string, NOT [{"caption": ...}]
            res[clean_id] = [str(data["final"])]

    # 3. Filter ground truths to only include the images we actually predicted
    gts = {k: v for k, v in gts.items() if k in res}

    # Initialize scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    print("\n📊 Computing Captioning Metrics...")
    metrics_results = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    metrics_results[m] = s
            else:
                metrics_results[method] = score
        except Exception as e:
            print(f"Failed to compute {method}: {e}")

    # Print all metrics
    for metric, score in metrics_results.items():
        print(f"   - {metric}: {score:.4f}")

    # Return CIDEr as the primary optimization metric for the sweep
    return metrics_results.get("CIDEr", 0.0)

def main():
    parser = argparse.ArgumentParser(description="Sweep parameters and evaluate corpus metrics.")
    parser.add_argument("--task", required=True, choices=["asr", "caption"])
    parser.add_argument("--input_dir", required=True, help="Path to audio or image files")
    parser.add_argument("--refs", required=True, help="Path to references.jsonl")
    parser.add_argument("--results_dir", default="./sweep_results", help="Where to save sweep data")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files for quick sweep testing") 
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Define the parameter grid based on the task
    if args.task == "asr":
        grid = {
            "gen_algos": ["beam,nucleus"],
            "beam_sizes": [4, 8],
            "rerank_algos": ["map", "mbr"],
            "mbr_metrics": ["wer"]
        }
    else:
        grid = {
            "gen_algos": ["beam,sample"],
            "beam_sizes": [5],
            "rerank_algos": ["map", "mbr", "fixed_rr"],
            "mbr_metrics": ["cider"]
        }
    
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []

    for i, exp in enumerate(experiments):
        exp_out_dir = os.path.join(args.results_dir, f"exp_{i}")
        
        # 1. Run the pipeline for this configuration
        run_pipeline_step(
            task=args.task,
            input_dir=args.input_dir,
            output_dir=exp_out_dir,
            gen_algos=exp["gen_algos"],
            beam_size=exp["beam_sizes"],
            rerank_algo=exp["rerank_algos"],
            mbr_metric=exp["mbr_metrics"],
            limit=args.limit
        )
        
        # 2. Evaluate the output
        final_file = os.path.join(exp_out_dir, f"{args.task}_final.jsonl")
        
        if args.task == "asr":
            primary_score = evaluate_asr(final_file, args.refs)
            print(f"\n🏆 Result for Exp {i} (Algo={exp['rerank_algos']}): WER = {primary_score:.2f}%")
            exp["wer"] = primary_score
        else:
            primary_score = evaluate_caption(final_file, args.refs)
            print(f"\n🏆 Result for Exp {i} (Algo={exp['rerank_algos']}): CIDEr = {primary_score:.4f}")
            exp["cider"] = primary_score
            
        results.append(exp)

    # 3. Save Summary
    summary_path = os.path.join(args.results_dir, f"{args.task}_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ All sweeps complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
