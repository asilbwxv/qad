# Quality-Aware Decoding (QAD) Pipeline

This repository implements a robust, modular pipeline for **Quality-Aware Decoding (QAD)** and **Minimum Bayes Risk (MBR)** decoding. It is designed to prove that likelihood-first decoding (MAP/Beam Search) is often misaligned with human-perceived quality in generative AI.

The pipeline currently supports:
1. **Automatic Speech Recognition (ASR)** using Whisper / NeMo.
2. **Image Captioning** using BLIP.

---

## Installation

**1. Create a virtual environment (Recommended):**
```bash
python -m venv qad_env
source qad_env/bin/activate  # On Windows: qad_env\Scripts\activate
```

**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```
*Note: We strictly use `datasets==2.19.0` to bypass underlying FFmpeg/torchcodec environment issues on Windows.*

**3. Install System Dependencies:**
* **Java:** Required by `pycocoevalcap` to compute SPICE and METEOR metrics for image captioning. Ensure Java is in your system PATH.

---

## Quick Start Guide

### Step 1: Download the Datasets
This script will automatically download the LibriSpeech (ASR) and MS COCO Karpathy splits (Captioning) and format their ground-truth references.

```bash
python download_datasets.py --task all --out ./data
```

### Step 2: Run a "Smoke Test" Example
Before running a massive computation, verify your GPU and Java environment are working by running a tiny, 2-file sweep. 

**Test ASR (Whisper):**
```bash
python sweep_and_eval.py --task asr --input_dir ./data/librispeech/audio --refs ./data/librispeech/references.jsonl --results_dir ./results/asr_test --limit 2
```

**Test Image Captioning (BLIP & Java Metrics):**
```bash
python sweep_and_eval.py --task caption --input_dir ./data/coco/images --refs ./data/coco/references.jsonl --results_dir ./results/caption_test --limit 2
```

### Step 3: Run the Full Experiments
Once the tests pass, you can drop the `--limit` flag and run the full dataset sweeps. *(Warning: Running O(N²) MBR over thousands of files is computationally heavy).*


---

## The Architecture & Pipeline Potential

This repository is built modularly. You do not have to rely on the master orchestrator scripts; you can chain the individual tools together to test highly specific decoding behaviors.

### 1. The Generators (`asr_gen.py`, `caption_gen.py`)
Standard generation scripts usually just output the single best prediction. These generators are built to create **diverse candidate pools** by forcing the models to explore alternative hypotheses.

* **Algorithms available (`--algos`):** * `beam`: Standard beam search. Acts as the mode-seeking baseline.
  * `nucleus` (ASR) / `sample` (Captioning): Triggers temperature sweeping and top-p sampling to force diverse, risky candidates.
* **Tuning generation:** You can control candidate density using `--beam_size`, `--n_samples` (ASR), or `--samples_per_param` (Captioning). 

### 2. The Rerankers (`asr_rerank.py`, `caption_rerank.py`)
Once a diverse pool of candidates is generated, the rerankers act as decision rules to select the final output. 

* **`map` (Maximum A-Posteriori):** The standard baseline. It selects the candidate the model thinks is most probable (highest log-likelihood).
* **`mbr` (Minimum Bayes Risk):** Computes pairwise expected utility across the entire candidate pool. It finds the "consensus" candidate. *Highly accurate, but scales quadratically $O(N^2)$*.
* **`fixed_rr` (Reference-Free Quality Estimation):** Uses a fast, referenceless metric to ground the text (e.g., NoRefER for ASR, or CLIPScore for Image-Text alignment).
* **`two_stage_mbr` (The QAD Fusion):** Solves the $O(N^2)$ bottleneck of MBR. It first uses a fast Quality Estimator to prune the massive candidate pool down to the `--prune_k` best options, and then runs heavy MBR consensus on that pruned subset.
* **`tuned_rr`:** Uses continuous optimization to combine multiple features (e.g., $w_1 \times \text{logprob} + w_2 \times \text{CLIPScore}$). Weights can be optimized using `tune_weights.py`.

### 3. The Supported Metrics
The pipeline relies on a unified wrapper (`metrics.py`) to handle complex evaluations safely.
* **ASR Utilities:** Word Error Rate (WER), Character Error Rate (CER), SeMaScore (Semantic Meaning Preservation), NoRefER.
* **Vision Utilities:** CIDEr (Consensus), SPICE (Semantic Graphing), BLEU, METEOR, CLIPScore (Visual Grounding).

### 4. Extras
To make experimentation seamless, the repository includes these additiona scripts:
* **`run_pipeline.py`**: Runs generation and reranking for a *single* specific configuration.
* **`sweep_and_eval.py`**: Takes a parameter grid (e.g., sweeping beam sizes from 4 to 16, comparing MAP vs MBR) and calculates the corpus-level metrics for all combinations.
