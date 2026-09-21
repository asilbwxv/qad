import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(summary_file, task, output_dir):
    """Reads the sweep summary JSON and generates bar charts and Pareto frontiers."""
    if not os.path.exists(summary_file):
        print(f"❌ Error: Could not find {summary_file}")
        return

    with open(summary_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("❌ Error: The summary file is empty.")
        return

    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)
    
    if task == "asr":
        # ---------------------------------------------------------
        # Plot 1: ASR Bar Chart (Facet by Model)
        # ---------------------------------------------------------
        g = sns.catplot(
            data=df, kind="bar",
            x="beam_sizes", y="wer", hue="rerank_algos", col="model_ckpts",
            palette="viridis", height=5, aspect=1.2
        )
        g.set_axis_labels("Candidate Pool Size (Beam Size)", "Word Error Rate (WER) %")
        g.set_titles("{col_name}")
        g.fig.suptitle("ASR QAD Performance (Lower WER is Better)", y=1.05, fontsize=14)
        
        # Add value labels
        for ax in g.axes.flat:
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(format(p.get_height(), '.2f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', 
                                xytext=(0, 8), textcoords='offset points', fontsize=9)
                    
        out_path_bar = os.path.join(output_dir, "asr_wer_barplot.png")
        plt.savefig(out_path_bar, dpi=300, bbox_inches='tight')
        print(f"✅ ASR Bar Plot saved to: {out_path_bar}")
        plt.close()

        # ---------------------------------------------------------
        # Plot 2: ASR Pareto Frontier (Latency vs WER)
        # ---------------------------------------------------------
        if "exec_time_seconds" in df.columns:
            plt.figure(figsize=(9, 6))
            sns.scatterplot(
                data=df, x="exec_time_seconds", y="wer", 
                hue="rerank_algos", style="model_ckpts", s=150, palette="viridis"
            )
            plt.title("ASR Pareto Frontier: Latency vs. WER", fontsize=14, pad=15)
            plt.xlabel("Execution Time (Seconds) - Log Scale", fontsize=12)
            plt.ylabel("Word Error Rate (WER) %", fontsize=12)
            plt.xscale('log') # Log scale because generation = hours, reranking = seconds
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            out_path_scatter = os.path.join(output_dir, "asr_latency_scatter.png")
            plt.savefig(out_path_scatter, dpi=300, bbox_inches='tight')
            print(f"✅ ASR Pareto Scatter Plot saved to: {out_path_scatter}")
            plt.close()
            
    elif task == "caption":
        # ---------------------------------------------------------
        # Plot 1: Captioning Bar Chart
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df, x="beam_sizes", y="cider", hue="rerank_algos", palette="mako"
        )
        plt.title("Image Captioning QAD Performance (Higher CIDEr is Better)", fontsize=14, pad=15)
        plt.ylabel("CIDEr Score", fontsize=12)
        plt.xlabel("Candidate Pool Size (Beam Size)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.3f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 8), textcoords='offset points', fontsize=10)
        
        out_path_bar = os.path.join(output_dir, "caption_cider_barplot.png")
        plt.savefig(out_path_bar, dpi=300, bbox_inches='tight')
        print(f"✅ Caption Bar Plot saved to: {out_path_bar}")
        plt.close()

        # ---------------------------------------------------------
        # Plot 2: Captioning Pareto Frontier (Latency vs CIDEr)
        # ---------------------------------------------------------
        if "exec_time_seconds" in df.columns:
            plt.figure(figsize=(9, 6))
            sns.scatterplot(
                data=df, x="exec_time_seconds", y="cider", 
                hue="rerank_algos", s=150, palette="mako"
            )
            plt.title("Captioning Pareto Frontier: Latency vs. CIDEr", fontsize=14, pad=15)
            plt.xlabel("Execution Time (Seconds) - Log Scale", fontsize=12)
            plt.ylabel("CIDEr Score", fontsize=12)
            plt.xscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            out_path_scatter = os.path.join(output_dir, "caption_latency_scatter.png")
            plt.savefig(out_path_scatter, dpi=300, bbox_inches='tight')
            print(f"✅ Caption Pareto Scatter Plot saved to: {out_path_scatter}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize QAD/MBR Sweep Results")
    parser.add_argument("--summary", required=True, help="Path to the *_sweep_summary.json file")
    parser.add_argument("--task", required=True, choices=["asr", "caption"])
    parser.add_argument("--out", default="./plots", help="Directory to save the output PNGs")
    args = parser.parse_args()
    
    plot_results(args.summary, args.task, args.out)

if __name__ == "__main__":
    main()
