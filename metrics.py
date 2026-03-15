import torch
import numpy as np
from typing import List, Dict, Union

# ==========================================
# ASR METRICS
# ==========================================
try:
    import jiwer
except ImportError:
    pass

def compute_wer(hyp: str, ref: str) -> float:
    """Lower is better"""
    return jiwer.wer(ref, hyp)

def compute_cer(hyp: str, ref: str) -> float:
    """Lower is better"""
    return jiwer.cer(ref, hyp)

class SemaScoreWrapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Assuming official semascore library is installed
        # from semascore import Semascore
        # self.scorer = Semascore(device=self.device)
        pass

    def score(self, hyp: str, ref: str) -> float:
        """Higher is better"""
        # return self.scorer.compute(hyp, ref)
        return 0.0 # Placeholder for official library call

class NoRefERWrapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained("aixplain/NoRefER", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("aixplain/NoRefER", trust_remote_code=True)
        self.model.to(self.device).eval()

    def score(self, texts: List[str]) -> List[float]:
        """Higher is better"""
        with torch.inference_mode():
            batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            sc = self.model.score(**batch)
            if isinstance(sc, torch.Tensor):
                return [float(x) for x in sc.detach().cpu().tolist()]
            return [float(x) for x in sc]

# ==========================================
# CAPTIONING METRICS
# ==========================================
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
except ImportError:
    pass

class CLIPScoreWrapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        from torchmetrics.multimodal.clip_score import CLIPScore
        self.device = device
        self.metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)

    def score(self, images, texts: List[str]) -> List[float]:
        """Higher is better. `images` can be PIL images or tensors."""
        # Note: Depending on input, you might need torchvision transforms here.
        # This assumes images are pre-processed tensors matching the text count.
        scores = []
        with torch.inference_mode():
            for text in texts:
                # In practice, you'd batch this. Keeping simple for interface.
                score = self.metric(images, text)
                scores.append(score.item())
        return scores

def caption_mbr_multiref(candidates: List[str], metric_name="cider") -> List[float]:
    """
    Computes multi-reference MBR for captioning.
    For each candidate i, it treats all other candidates (j != i) as the reference set.
    """
    if metric_name == "cider":
        scorer = Cider()
    elif metric_name == "spice":
        scorer = Spice()
    else:
        raise ValueError(f"Unsupported multi-ref metric: {metric_name}")

    N = len(candidates)
    if N <= 1:
        return [0.0] * N

    mbr_scores = []
    for i in range(N):
        hyp = {str(i): [candidates[i]]}
        # All other candidates are the "crowd" consensus references
        refs = {str(i): [candidates[j] for j in range(N) if j != i]}
        
        score, _ = scorer.compute_score(refs, hyp)
        mbr_scores.append(float(score))
        
    return mbr_scores
