import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(summary_file, task, output_dir):
    """Reads the sweep summary JSON and generates a bar chart."""
    if not os.path.exists(summary_file):
        print(f"❌ Error: Could not find {summary_file}")
        return

    with open(summary_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("❌ Error: The summary file is empty.")
        return

    df = pd.DataFrame(data)
    
    # Set up a clean, academic seaborn style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    if task == "asr":
        # ASR: We want to plot WER (Lower is better)
        # Group by beam_sizes on the X-axis, color by rerank algorithm
        ax = sns.barplot(
            data=df, 
            x="beam_sizes", 
            y="wer", 
            hue="rerank_algos", 
            palette="viridis"
        )
        plt.title("ASR QAD Performance by Beam Size (Lower WER is Better)", fontsize=14, pad=15)
        plt.ylabel("Word Error Rate (WER) %", fontsize=12)
        plt.xlabel("Candidate Pool Size (Beam Size)", fontsize=12)
        
        # Add value labels on top of the bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.2f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 8), 
                            textcoords='offset points',
                            fontsize=10)
                        
    elif task == "caption":
        # Captioning: Plot CIDEr (Higher is better)
        # X-axis is the reranking algorithm
        ax = sns.barplot(
            data=df, 
            x="rerank_algos", 
            y="cider", 
            palette="mako"
        )
        plt.title("Image Captioning QAD Performance (Higher CIDEr is Better)", fontsize=14, pad=15)
        plt.ylabel("CIDEr Score", fontsize=12)
        plt.xlabel("Decoding / Reranking Algorithm", fontsize=12)
        
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.3f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 8), 
                            textcoords='offset points',
                            fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task}_results_plot.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize QAD/MBR Sweep Results")
    parser.add_argument("--summary", required=True, help="Path to the *_sweep_summary.json file")
    parser.add_argument("--task", required=True, choices=["asr", "caption"], help="Task type to format the plot correctly")
    parser.add_argument("--out", default="./plots", help="Directory to save the output PNGs")
    args = parser.parse_args()
    
    plot_results(args.summary, args.task, args.out)

if __name__ == "__main__":
    main()
