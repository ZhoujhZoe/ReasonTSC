import json
import argparse

def process_ts_to_json(ts_filename: str, json_filename: str) -> None:
    """
    Process time series data from .ts file to JSON format.
    
    Args:
        ts_filename: Path to input .ts file
        json_filename: Path to output JSON file
    """
    data = []
    
    with open(ts_filename, 'r') as ts_file:
        for idx, line in enumerate(ts_file):
            line = line.strip()
            if not line:
                continue
                
            sample_data, label = line.rsplit(":", 1)
            sample = ','.join([f"{float(num):.2f}" for num in sample_data.split(',')])
            
            data.append({
                "id": idx,
                "label": int(label),
                "sample": sample
            })
    
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process time series data from .ts to JSON format.')
    parser.add_argument('--ts_file', required=True, help='Input .ts file path')
    parser.add_argument('--json_file', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    process_ts_to_json(args.ts_file, args.json_file)