# Quality-Aware Decoding (QAD) Beyond Translation

This repository implements a modular pipeline for **Quality-Aware Decoding (QAD)** and **Minimum Bayes Risk (MBR)** decoding, focusing on Automatic Speech Recognition (ASR) and Image Captioning. 

Standard decoding methods, such as beam search, rely on Maximum A-Posteriori (MAP) estimation to find the most probable sequence according to the model. However, likelihood is often misaligned with human-perceived quality. This project shifts from a likelihood-first approach to a quality-first approach: it generates a diverse pool of candidates and then selects the final output using task-aligned utility functions (like WER/SeMaScore for ASR, and CIDEr/CLIPScore for Image Captioning).

## 📂 Repository Structure

The project is broken down into modular scripts to separate candidate generation, metric evaluation, and reranking logic:

* **`download_datasets.py`**: Automatically fetches the LibriSpeech and MS COCO (Karpathy split) datasets.
* **`metrics.py`**: A centralized wrapper for all scoring libraries (WER, CIDEr, CLIPScore, NoRefER, SeMaScore).
* **`asr_gen.py` & `caption_gen.py`**: Generates diverse candidate pools using temperature sweeps, nucleus sampling, and beam search variations.
* **`tune_weights.py`**: Uses continuous optimization to find the best linear combination of metric weights for Tuned Reranking (T-RR).
* **`asr_rerank.py` & `caption_rerank.py`**: Executes the decision rules: MAP, Fixed Reranking, MBR, and Two-Stage MBR.
* **`run_pipeline.py`**: A cross-platform orchestrator that strings generation and reranking together in one command.
* **`sweep_and_eval.py`**: Automates parameter combinations and calculates corpus-level benchmarks.
* **`visualize_results.py`**: Turns JSON sweep summaries into publication-ready graphs.

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets soundfile pillow requests numpy pandas matplotlib seaborn scipy
pip install jiwer torchmetrics
```

**For Image Captioning evaluation (CIDEr, SPICE, METEOR, BLEU):**
```bash
pip install pycocoevalcap
```
*(Note: Java is required on your system for pycocoevalcap's METEOR/SPICE to work properly).*

---

## 🚀 Step-by-Step Guide

### Step 1: Download Datasets
Before running experiments, download the necessary data. The script will automatically format the references into `references.jsonl` files for evaluation.

**For ASR (LibriSpeech test-clean):**
```bash
python download_datasets.py --task asr --out ./data
```

**For Image Captioning (MS COCO Karpathy split):**
```bash
python download_datasets.py --task caption --out ./data
```

### Step 2: Run a Single End-to-End Pipeline
If you want to test a specific configuration on your data, use the `run_pipeline.py` orchestrator. 

**ASR Example:** Generating candidates with beam search and nucleus sampling, then using Two-Stage MBR (pruning with NoRefER, finalizing with SeMaScore).
```bash
python run_pipeline.py \
    --task asr \
    --input_dir ./data/librispeech/audio \
    --output_dir ./results/asr_test \
    --gen_algos beam,nucleus \
    --beam_size 5 \
    --rerank_algo two_stage_mbr \
    --mbr_metric semascore
```

**Captioning Example:** Generating a diverse pool of captions, then using Multi-Reference MBR (optimizing for CIDEr).
```bash
python run_pipeline.py \
    --task caption \
    --input_dir ./data/coco/images \
    --output_dir ./results/caption_test \
    --gen_algos beam,sample \
    --rerank_algo mbr \
    --mbr_metric cider
```

### Step 3: Sweep Parameters and Evaluate (The Research Workflow)
To compare algorithms (e.g., MAP vs MBR) across different beam sizes and candidate counts, use the sweeper. It will run the full pipeline for multiple configurations and compute the corpus-level metrics (WER for ASR, CIDEr for Captioning).

**Sweep ASR:**
```bash
python sweep_and_eval.py \
    --task asr \
    --input_dir ./data/librispeech/audio \
    --refs ./data/librispeech/references.jsonl \
    --results_dir ./sweep_results/asr
```

**Sweep Image Captioning:**
```bash
python sweep_and_eval.py \
    --task caption \
    --input_dir ./data/coco/images \
    --refs ./data/coco/references.jsonl \
    --results_dir ./sweep_results/caption
```

### Step 4: Visualizing Results
Convert the `sweep_summary.json` generated in Step 3 into bar charts to analyze the quality-compute trade-off.

```bash
# Visualize ASR Sweep (Plots WER)
python visualize_results.py \
    --task asr \
    --summary ./sweep_results/asr/asr_sweep_summary.json \
    --out ./plots

# Visualize Captioning Sweep (Plots CIDEr)
python visualize_results.py \
    --task caption \
    --summary ./sweep_results/caption/caption_sweep_summary.json \
    --out ./plots
```

### Step 5: (Optional) Tuning Weights for QAD
If you wish to use **Tuned Reranking (`tuned_rr`)**, you can optimize the weights of different features (e.g., log-probability and NoRefER) against a target metric (like WER) on a validation set.

1. Generate candidates on a validation set.
2. Run the tuner:
```bash
python tune_weights.py \
    --candidates ./results/asr_test/asr_candidates.jsonl \
    --refs ./data/librispeech/references.jsonl \
    --features logprob,norefer \
    --target wer
```
3. Pass the resulting weights to `run_pipeline.py` using `--tune_weights`.

---

## 🧠 Algorithms Explained

* **`map` (Maximum A-Posteriori):** The standard baseline. Selects the candidate with the highest model probability (usually the top beam search result).
* **`fixed_rr` (Fixed Reranking):** Uses a single Quality Estimation (QE) metric to rank candidates (e.g., NoRefER for ASR, or CLIPScore for visual grounding in captioning).
* **`tuned_rr` (Tuned Reranking):** Uses a tuned linear combination of features (e.g., `w1 * logprob + w2 * QE_score`).
* **`mbr` (Minimum Bayes Risk):** Computes expected utility across the candidate pool. For ASR, it minimizes expected error (WER/CER). For Captioning, it maximizes expected consensus (CIDEr) against the other candidates. *Complexity: O(N^2).*
* **`two_stage_mbr`:** Fuses fast Quality Estimators with heavy MBR. First, scores all candidates with a fast QE (e.g., NoRefER/CLIPScore) to prune the list to the Top-K. Then, it runs the expensive O(K^2) MBR algorithm on the pruned subset.

## 📏 Supported Metrics
**ASR:**
* **WER/CER:** Word/Character Error Rate (Edit-distance).
* **NoRefER:** Referenceless Quality Estimation via LLM.
* **SeMaScore:** Meaning preservation and semantic utility.

**Image Captioning:**
* **CIDEr / SPICE / BLEU / METEOR:** Reference-based n-gram and semantic consensus metrics.
* **CLIPScore:** Referenceless visual-semantic grounding.
