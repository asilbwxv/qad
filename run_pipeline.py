import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd_list):
    """Executes a shell command cross-platform and handles errors."""
    print(f"\n🚀 Running: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing step. Exiting pipeline.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline for QAD/MBR in ASR & Captioning")
    
    # General arguments
    parser.add_argument("--task", required=True, choices=["asr", "caption"], help="Which task to run")
    parser.add_argument("--input_dir", required=True, help="Directory containing audio or image files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final outputs")
    
    # Generation arguments
    parser.add_argument("--gen_algos", default="beam,nucleus", help="Algorithms for candidate generation")
    parser.add_argument("--beam_size", type=str, default="5")
    parser.add_argument("--model", type=str, default="default", help="Model name or 'whisper'/'nemo' for ASR")
    
    # Reranking arguments
    parser.add_argument("--rerank_algo", required=True, choices=["map", "fixed_rr", "tuned_rr", "mbr", "two_stage_mbr"], help="Reranking algorithm")
    parser.add_argument("--mbr_metric", type=str, default="wer", help="Metric for MBR (wer/cer/semascore for ASR; cider/spice for caption)")
    
    # Tuning arguments (if tuned_rr is selected)
    parser.add_argument("--val_refs", type=str, help="Path to validation references (Required if algo=tuned_rr)")
    parser.add_argument("--tune_features", type=str, default="logprob,norefer", help="Features to tune")
    
    args = parser.parse_args()

    # Setup paths
    os.makedirs(args.output_dir, exist_ok=True)
    candidates_jsonl = os.path.join(args.output_dir, f"{args.task}_candidates.jsonl")
    final_jsonl = os.path.join(args.output_dir, f"{args.task}_final.jsonl")

    # Use the current python executable to ensure cross-platform compatibility
    python_exe = sys.executable

    # ==========================================================
    # STAGE 1: GENERATION
    # ==========================================================
    print(f"\n--- STAGE 1: Generating Diverse Candidates for {args.task.upper()} ---")
    gen_script = "asr_gen.py" if args.task == "asr" else "caption_gen.py"
    
    gen_cmd = [
        python_exe, gen_script,
        "--dir", args.input_dir,
        "--out", candidates_jsonl,
        "--algos", args.gen_algos,
        "--beam_size", args.beam_size
    ]
    
    # Add ASR specific model flag if needed
    if args.task == "asr" and args.model in ["whisper", "nemo"]:
        gen_cmd.extend(["--model_type", args.model])
        
    run_command(gen_cmd)

    # ==========================================================
    # STAGE 2: TUNING (OPTIONAL)
    # ==========================================================
    tuned_weights = None
    if args.rerank_algo == "tuned_rr":
        if not args.val_refs:
            print("\n❌ Error: --val_refs is required for Tuned Reranking (tuned_rr).")
            sys.exit(1)
            
        print(f"\n--- STAGE 2: Tuning Weights using {args.val_refs} ---")
        tune_cmd = [
            python_exe, "tune_weights.py",
            "--candidates", candidates_jsonl,
            "--refs", args.val_refs,
            "--features", args.tune_features,
            "--target", "wer" if args.task == "asr" else "cider"
        ]
        # In a fully integrated flow, tune_weights.py would save the weights to a file or print them.
        # For simplicity, we are assuming it prints them and the user can feed them, 
        # or we pass a hardcoded mock string here for demonstration.
        run_command(tune_cmd)
        
        # Placeholder for parsed weights (in a real scenario, you'd capture stdout from the tune_cmd)
        tuned_weights = "0.5,0.5" 

    # ==========================================================
    # STAGE 3: RERANKING
    # ==========================================================
    print(f"\n--- STAGE 3: Reranking using {args.rerank_algo.upper()} ---")
    rerank_script = "asr_rerank.py" if args.task == "asr" else "caption_rerank.py"
    
    rerank_cmd = [
        python_exe, rerank_script,
        "--inp", candidates_jsonl,
        "--out", final_jsonl,
        "--algo", args.rerank_algo,
        "--mbr_metric", args.mbr_metric
    ]
    
    if args.rerank_algo == "tuned_rr" and tuned_weights:
        rerank_cmd.extend([
            "--tune_features", args.tune_features,
            "--tune_weights", tuned_weights
        ])

    run_command(rerank_cmd)

    print(f"\n✅ Pipeline Complete! Final outputs saved to: {final_jsonl}")

if __name__ == "__main__":
    main()
