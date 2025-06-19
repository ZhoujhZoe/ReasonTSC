import json
import re
import argparse

# Define patterns for extracting values from LLM output
PATTERNS = {
    "Trend": r"Trend Differences[^\d]*(\d)",
    "Cyclic": r"Cyclic behavior Differences[^\d]*(\d)",
    "Stationarity": r"Stationarity Differences[^\d]*(\d)",
    "Amplitude": r"Amplitude Differences[^\d]*(\d)",
    "Rate of change": r"Rate of change Differences[^\d]*(\d)",
    "Outliers": r"Outliers Differences[^\d]*(\d)",
    "Noise level": r"Noise level Differences[^\d]*(\d)",
    "Volatility": r"Volatility Differences[^\d]*(\d)",
    "Structural Break": r"Structural Break Differences[^\d]*(\d)",
    "Mean Level": r"Mean Level Differences[^\d]*(\d)"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Process LLM output JSON file')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input JSON file containing LLM output')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output JSON file for processed results')
    return parser.parse_args()

def extract_value(text, pattern):
    """Extract numeric value using regex pattern"""
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None

def process_json(data):
    for item in data:
        for conv in item['conversation']:
            if conv['role'] == 'assistant':
                content = conv['content']
                for key, pattern in PATTERNS.items():
                    item[key] = extract_value(content, pattern)
    return data

def main():
    args = parse_args()
    
    with open(args.input_file, 'r', encoding='utf-8') as file:
        original_data = json.load(file)
    
    processed_data = process_json(original_data)
    
    with open(args.output_file, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)
    
    print(f"Processing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()