import json
import random
import numpy as np

def save_to_jsonl(arrays, strings, file_path, shuffle=False):
    """Save arrays and their string representations to JSONL file."""
    data = list(zip(arrays, strings))
    if shuffle:
        random.shuffle(data)
    
    with open(file_path, "w") as f:
        for idx, (arr, string) in enumerate(data):
            json.dump({
                "index": idx,
                "input_arr": arr.tolist(),
                "input_str": string
            }, f)
            f.write("\n")

def process_jsonl(input_file, output_file):
    """Process JSONL file into comparison format."""
    with open(input_file) as f:
        samples = [json.loads(line) for line in f]
    
    # Split and balance samples
    pattern_samples = [s for s in samples if s['index'] % 2 == 0]
    random_samples = [s for s in samples if s['index'] % 2 == 1]
    min_len = min(len(pattern_samples), len(random_samples))
    
    results = []
    for pattern, random_seq in zip(pattern_samples[:min_len], random_samples[:min_len]):
        if random.choice([True, False]):
            sampleA = pattern['input_arr']
            sampleB = random_seq['input_arr']
            answer = "A"
        else:
            sampleA = random_seq['input_arr']
            sampleB = pattern['input_arr']
            answer = "B"
        
        results.append({
            "index": pattern['index'],
            "sampleA": sampleA,
            "sampleB": sampleB,
            "answer": answer
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)