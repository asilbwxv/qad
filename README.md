# Quality-Aware Decoding (QAD) & MBR for Multimodal Generation

This repository contains the complete implementation for **Quality-Aware Decoding (QAD)** and **Minimum Bayes Risk (MBR)** applied to continuous multimodal sequence generation. It investigates test-time compute scaling, structural candidate reranking, and hallucination suppression in **Automatic Speech Recognition (ASR)** and **Image Captioning**.

The codebase evaluates decoding pipelines against severe algorithmic vulnerabilities:
* **The ASR Hallucination Trap:** Exposing the "fluency bias" of text-centric Quality Estimators on noisy acoustic data.
* **The Vision-Language Modality Gap:** Demonstrating the "bag-of-words" degradation inherent to continuous visual embedding optimization (CLIPScore).
* **Two-Stage MBR Fusion:** A hybrid architecture that uses rapid referenceless estimation to prune candidates before executing probability-weighted MBR consensus, eliminating algorithmic bias and the $O(N^2)$ computational bottleneck.

---

## Supported Architectures & Evaluators
* **Generators:** OpenAI Whisper (`whisper-small`, `whisper-large-v3`) and Salesforce BLIP.
* **Decoding Search:** Standard Beam Search, Diverse Beam Search (DBS), Nucleus (Top-$p$) Sampling.
* **Decision Rules:** MAP (Baseline), Pure MBR, Fixed Reranking (F-RR), Tuned Reranking (T-RR), and Two-Stage MBR.
* **Task Metrics:** Word Error Rate (WER), CIDEr, SPICE.
* **Referenceless Quality Estimators (QE):** NoRefER (ASR), CLIPScore (Vision).
* **Human-Aligned Proxies:** Sentence-BERT Semantic Textual Similarity (STS).

---

## Installation

**1. Create a virtual environment:**
```bash
python -m venv qad_env
source qad_env/bin/activate

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
pip install sentence-transformers  # Required for human-preference STS evaluation

```

*Note: We strictly use `datasets==2.19.0` to ensure stability when streaming massive audio corpora.*

**3. Java Runtime:**
Required by `pycocoevalcap` to compute SPICE and METEOR. Ensure Java is installed and in your system PATH.

---

## Datasets and Data Acquisition

The pipeline features automated extraction for all human-aligned multimodal benchmarks:

* **MS COCO (Karpathy Split):** Image captioning multi-reference datasets.
* **LibriSpeech (`test-other`):** Degraded acoustic environments.
* **HALAS (Earnings-22):** 745 uncorrupted audio files featuring human-annotated ASR hallucination labels.
* **Google FLEURS (`pt_br`):** Portuguese benchmark for testing cross-lingual consensus generalization.

**To download and join the datasets:**

```bash
python download_datasets.py --task all --out ./data
python download_halas.py  # Streams Earnings-22 to extract HALAS audio
python download_fleurs_pt.py

```

---

## Execution & Workflow

The architecture mathematically isolates the cost of autoregressive candidate generation from post-hoc reranking evaluation via a JSONL caching protocol.

### 1. Generating Diverse Candidates

```bash
python asr_gen.py --dir ./data/halas/audio --out ./results/halas/candidates.jsonl --algos beam,nucleus --beam_size 8 --ckpt openai/whisper-large-v3

```

### 2. Reranking (MBR / Quality Estimation)

Apply specific decision rules over the cached candidates:

```bash
python asr_rerank.py --inp ./results/halas/candidates.jsonl --out ./results/halas/final_output.jsonl --algo two_stage_mbr --prune_k 2 --mbr_metric wer

```

### 3. Running Automated Parameter Sweeps

The `sweep_and_eval.py` orchestrator automatically iterates over generation strategies, beam sizes, and decision rules:

```bash
python sweep_and_eval.py --task asr --input_dir ./data/halas/audio --refs ./data/halas/references.jsonl --results_dir ./results/halas_full --limit 0

```

---

## Diagnostic Tools & Ablation Sweeps

* **$K$-Pruning Threshold Ablation (`ablation_k_sweep.py`):**
Sweeps the Stage-1 pruning parameter $K \in \{1, 2, 3, 5, 8, 10\}$ to calculate the exact Pareto frontier between WER/CIDEr error minimization and test-time latency.
* **Hallucination Detection (`analyze_halas_hallucinations.py`):**
Cross-references model outputs against the human-verified HALAS annotations to compute the Severe Hallucination Survival Rate.
* **Human-Preference Semantic Evaluation (`eval_sts_cxc.py`):**
Utilizes Sentence-BERT (`all-MiniLM-L6-v2`) to compute continuous Semantic Textual Similarity (STS) scores, tracking human preference alignment beyond discrete $n$-gram overlaps.
* **Qualitative Error Taxonomy (`extract_failure_cases.py`):**
Scans generative pools and extracts definitive failure cases to study acoustic decoupling and modality collapse.

---

## Key Empirical Findings

* **Hallucination Suppression:** On the HALAS benchmark under Diverse Beam Search ($B=8$), standard MAP decoding suffered a 122.04% WER. Two-Stage MBR successfully eliminated ungrounded fabrications, yielding a **42.63 percentage-point absolute WER reduction** and dropping severe human-annotated hallucinations from 20% to 12%.
* **The Modality Gap:** Pure visual optimization via CLIPScore severely degraded grammatical structure (CIDEr collapsed from 0.83 to 0.63). Two-Stage MBR bridged this gap, restoring structural consensus (CIDEr: 0.67) while simultaneously achieving peak human-preference semantic alignment (STS: 59.04/100).
* **Scaling Laws:** In speech, the optimal pruning parameter stabilizes at $K=2$, mitigating fluency bias instantly. Vision-language structures demand a wider consensus window, achieving peak performance at $K=5$.
