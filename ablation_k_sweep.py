import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def run_k_ablation(task, candidates_jsonl, refs_jsonl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    k_values = [1, 2, 3, 5, 8, 10]
    results = []
    
    print(f"\n=========================================================")
    print(f"🔬 RUNNING K-PRUNING ABLATION STUDY: {task.upper()}")
    print(f"=========================================================")
    
    rerank_script = "asr_rerank.py" if task == "asr" else "caption_rerank.py"
    eval_metric = "wer" if task == "asr" else "cider"
    
    for k in k_values:
        out_jsonl = os.path.join(out_dir, f"k_{k}_final.jsonl")
        
        cmd = [
            "python", rerank_script,
            "--inp", candidates_jsonl,
            "--out", out_jsonl,
            "--algo", "two_stage_mbr",
            "--prune_k", str(k),
            "--mbr_metric", eval_metric
        ]
        
        start_time = time.time()
        subprocess.run(cmd, check=True)
        exec_time = time.time() - start_time
        
        if task == "asr":
            from sweep_and_eval import evaluate_asr
            score = evaluate_asr(out_jsonl, refs_jsonl)
            print(f"---> K={k} | WER: {score:.2f}% | Latency: {exec_time:.2f}s")
            results.append({"K": k, "WER": score, "exec_time_seconds": exec_time})
        else:
            from sweep_and_eval import evaluate_caption
            score = evaluate_caption(out_jsonl, refs_jsonl)
            print(f"---> K={k} | CIDEr: {score:.4f} | Latency: {exec_time:.2f}s")
            results.append({"K": k, "CIDEr": score, "exec_time_seconds": exec_time})
            
    summary_path = os.path.join(out_dir, f"{task}_k_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
        
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Pruning Subset Size (K)', fontsize=12)
    
    if task == "asr":
        ax1.set_ylabel('Word Error Rate (WER) %', color=color, fontsize=12)
        ax1.plot(df['K'], df['WER'], marker='o', color=color, linewidth=2, label="WER %")
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        ax1.set_ylabel('CIDEr Score', color=color, fontsize=12)
        ax1.plot(df['K'], df['CIDEr'], marker='s', color=color, linewidth=2, label="CIDEr")
        ax1.tick_params(axis='y', labelcolor=color)
        
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (s)', color=color, fontsize=12)
    ax2.plot(df['K'], df['exec_time_seconds'], marker='^', color=color, linestyle='--', linewidth=2, label="Latency (s)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Impact of Pruning Parameter K on {task.upper()} Quality & Latency", fontsize=14, pad=15)
    fig.tight_layout()
    plot_path = os.path.join("./plots", f"{task}_k_ablation_plot.png")
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Ablation plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["asr", "caption"])
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--out", default="./results/ablation")
    args = parser.parse_args()
    
    run_k_ablation(args.task, args.candidates, args.refs, args.out)

if __name__ == "__main__":
    main()
