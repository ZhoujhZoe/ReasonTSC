import json
import random
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process time series data')
    parser.add_argument('--input_jsonl', type=str, required=True,
                      help='Path to input JSONL file')
    parser.add_argument('--output_json', type=str, required=True,
                      help='Path to output JSON file')
    return parser.parse_args()

def process_jsonl(input_file, output_file):
    """
    Process JSONL file containing time series data by pairing pattern and random samples
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file
    """
    # Read and parse input file
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]
    
    # Separate pattern and random samples
    pattern_samples = [sample for sample in lines if sample['index'] % 2 == 0]
    random_samples = [sample for sample in lines if sample['index'] % 2 == 1]
    
    # Balance sample counts
    min_len = min(len(pattern_samples), len(random_samples))
    pattern_samples = pattern_samples[:min_len]
    random_samples = random_samples[:min_len]
    
    # Create paired comparisons
    results = []
    for pattern, random_seq in zip(pattern_samples, random_samples):
        if random.choice([True, False]):
            sampleA = ", ".join(map(str, pattern['input_arr']))
            sampleB = ", ".join(map(str, random_seq['input_arr']))
            answer = "A"
        else:
            sampleA = ", ".join(map(str, random_seq['input_arr']))
            sampleB = ", ".join(map(str, pattern['input_arr']))
            answer = "B"
        
        results.append({
            "index": pattern['index'],
            "sampleA": sampleA,
            "sampleB": sampleB,
            "answer": answer
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    args = parse_args()
    process_jsonl(args.input_jsonl, args.output_json)
    print(f"Processing complete. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()