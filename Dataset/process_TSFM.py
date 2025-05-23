import json
import argparse
from typing import Dict, List

# Label mapping configuration
LABEL_MAPPING: Dict[int, int] = {0: 1, 1: 2, 2: 3, 3: 4}

def process_txt_to_json(txt_filename: str, json_filename: str) -> None:
    """
    Process TSFM model output from text file to JSON format.
    
    Args:
        txt_filename: Path to input text file
        json_filename: Path to output JSON file
    """
    data = []
    
    with open(txt_filename, 'r') as txt_file:
        for line in txt_file:
            line = line.strip()
            if not line or line.startswith("Batch_idx"):
                continue
            
            parts = line.split(', ', 3)  # Split into max 4 parts
            idx = int(parts[0])
            predicted_label = int(parts[1])
            real_label = int(parts[2])
            logits = [round(float(x), 2) for x in parts[3].strip('[]').split(", ")]
            
            # Apply label mapping
            mapped_predicted = LABEL_MAPPING.get(predicted_label, predicted_label)
            mapped_label = LABEL_MAPPING.get(real_label, real_label)
            mapped_logits = [0.0] * len(logits)
            
            for i, logit in enumerate(logits):
                mapped_idx = LABEL_MAPPING.get(i, i)
                mapped_logits[mapped_idx - 1] = logit
            
            data.append({
                "id": idx, 
                "label": mapped_label,
                "predicted": mapped_predicted,
                "logits": f"[{', '.join(map(str, mapped_logits))}]"
            })
    
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TSFM model output to JSON format.')
    parser.add_argument('--txt_file', required=True, help='Input text file path')
    parser.add_argument('--json_file', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    process_txt_to_json(args.txt_file, args.json_file)