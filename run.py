import os
import subprocess
import argparse

def run_experiment(task, input_dir, refs, results_dir, gen_algos, beam_sizes, rerank_algos, mbr_metrics):
    """Helper to call the sweep_and_eval.py script with specific thesis configurations."""
    cmd = [
        "python", "sweep_and_eval.py",
        "--task", task,
        "--input_dir", input_dir,
        "--refs", refs,
        "--results_dir", results_dir
    ]
    print(f"\n=========================================================")
    print(f"🚀 RUNNING THESIS EXPERIMENT BATCH: {task.upper()}")
    print(f"=========================================================")
    
    # We will temporarily modify sweep_and_eval.py's internal grid by passing these 
    # as environment variables, or we can just call run_pipeline.py directly. 
    # For maximum control, we call run_pipeline.py directly here to bypass the hardcoded grid.
    
    for beam in beam_sizes:
        for r_algo in rerank_algos:
            for metric in mbr_metrics:
                out_name = f"exp_{task}_beam{beam}_{r_algo}_{metric}"
                out_dir = os.path.join(results_dir, out_name)
                os.makedirs(out_dir, exist_ok=True)
                
                pipeline_cmd = [
                    "python", "run_pipeline.py",
                    "--task", task,
                    "--input_dir", input_dir,
                    "--output_dir", out_dir,
                    "--gen_algos", gen_algos,
                    "--beam_size", str(beam),
                    "--rerank_algo", r_algo,
                    "--mbr_metric", metric
                ]
                
                print(f"\n---> Executing: Beam={beam} | Algo={r_algo} | Metric={metric}")
                subprocess.run(pipeline_cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Script for Thesis Experiments")
    parser.add_argument("--data_dir", default="./data", help="Base directory where datasets were downloaded")
    parser.add_argument("--results_dir", default="./thesis_results", help="Where to save all experiment outputs")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Paths
    asr_audio = os.path.join(args.data_dir, "librispeech", "audio")
    asr_refs = os.path.join(args.data_dir, "librispeech", "references.jsonl")
    
    cap_images = os.path.join(args.data_dir, "coco", "images")
    cap_refs = os.path.join(args.data_dir, "coco", "references.jsonl")

    # ==========================================================
    # PHASES 1-3: ASR EXPERIMENTS
    # ==========================================================
    if os.path.exists(asr_audio):
        run_experiment(
            task="asr",
            input_dir=asr_audio,
            refs=asr_refs,
            results_dir=os.path.join(args.results_dir, "asr"),
            gen_algos="beam,nucleus",
            beam_sizes=[5, 10], # MAP Baseline vs diversity
            rerank_algos=["map", "mbr", "fixed_rr", "two_stage_mbr"],
            mbr_metrics=["wer", "semascore"]
        )
    else:
        print("⚠️ ASR data not found. Skipping ASR experiments.")

    # ==========================================================
    # PHASE 4: IMAGE CAPTIONING EXPERIMENTS
    # ==========================================================
    if os.path.exists(cap_images):
        run_experiment(
            task="caption",
            input_dir=cap_images,
            refs=cap_refs,
            results_dir=os.path.join(args.results_dir, "caption"),
            gen_algos="beam,sample",
            beam_sizes=[5], # Sweeping temperatures handled internally by caption_gen.py
            rerank_algos=["map", "fixed_rr", "mbr", "two_stage_mbr"],
            mbr_metrics=["cider", "spice"]
        )
    else:
        print("⚠️ Captioning data not found. Skipping Captioning experiments.")

    print("\n✅ ALL THESIS EXPERIMENTS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
