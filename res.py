import json
import os

try:
    with open('results/asr_noisy_test/exp_0/asr_candidates.jsonl', encoding='utf-8') as f:
        data = json.loads(f.readline())
        cands = data['candidates']
        
    print('\n=== WHISPER-SMALL CANDIDATE DIVERSITY ===')
    for c in cands[:5]:
        text = c["text"]
        algo = c["meta"].get("algo", "unknown")
        print(f" - {text} (Algo: {algo})")
        
except Exception as e:
    print(f"Error: {e}")