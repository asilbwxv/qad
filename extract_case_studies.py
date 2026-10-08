import json
import random

def extract_asr_case_study(candidates_file):
    print("\n" + "="*50)
    print("ASR CASE STUDY (Looking for DBS Hallucinations)")
    print("="*50)
    with open(candidates_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines[:5]: # Check first 5 utterances
        data = json.loads(line)
        print(f"\nUtt ID: {data['utt_id']}")
        beam_cands = [c['text'] for c in data['candidates'] if c['meta'].get('algo') == 'beam']
        dbs_cands = [c['text'] for c in data['candidates'] if c['meta'].get('algo') == 'dbs']
        
        print(f"Top Beam Output: {beam_cands[0] if beam_cands else 'None'}")
        print(f"DBS Hallucinations:")
        for i, text in enumerate(dbs_cands[-3:]): # Print the most extreme DBS branches
            print(f"  - {text}")

def extract_caption_case_study(candidates_file):
    print("\n" + "="*50)
    print("CAPTIONING CASE STUDY (Looking for Divergence)")
    print("="*50)
    with open(candidates_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines[:5]:
        data = json.loads(line)
        print(f"\nImage ID: {data['image_id']}")
        for i, c in enumerate(data['candidates']):
            algo = c['meta'].get('algo')
            print(f"[{algo.upper()}] {c['text']}")

if __name__ == "__main__":
    # Point this to the Beam=8 whisper-small candidates where WER was 79%
    asr_path = "./results/asr_noisy_test/exp_6/asr_candidates.jsonl"
    # Point this to the Beam=10 captioning candidates
    cap_path = "./results/caption_test/exp_4/caption_candidates.jsonl"
    
    import os
    if os.path.exists(asr_path): extract_asr_case_study(asr_path)
    if os.path.exists(cap_path): extract_caption_case_study(cap_path)
