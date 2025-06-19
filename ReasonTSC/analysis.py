import json
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze model prediction results')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to input JSON file containing prediction data')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output JSON file for analyzed results')
    return parser.parse_args()

def load_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_data(output_file, data):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def extract_llm_prediction(last_content):
    """Extract LLM predicted label from conversation content"""
    true_label_matches = re.findall(r'\*\*True Label.*(\d+)', last_content)
    return int(true_label_matches[-1]) if true_label_matches else None

def extract_llm_confidence(last_content):
    """Extract LLM confidence score from conversation content"""
    confidence_matches = re.findall(r'\*\*Confidence.*(\d+\.\d{2})', last_content)
    return float(confidence_matches[-1]) if confidence_matches else None

def analyze_predictions(data):
    """Analyze and compare prediction results between models"""
    correct_llm = 0  
    correct_predicted = 0  
    consistent = 0  
    overriden_acc_case = 0
    llm_mismatch_idx = []
    predicted_mismatch_idx = []
    inconsistent_idx = [] 

    for item in data:
        last_content = item['conversation'][-1]['content']

        if last_content is None:
            print(f"Skipping item with idx {item['idx']} due to None content")
            continue

        item['LLM_predicted'] = extract_llm_prediction(last_content)
        item['LLM_confidence'] = extract_llm_confidence(last_content)

        # Check if LLM prediction matches true label
        if item['LLM_predicted'] is not None and item['LLM_predicted'] == item['label']:
            correct_llm += 1
        else:
            llm_mismatch_idx.append(item['idx'])
            item['llm_mismatch'] = 1  

        # Check if specific model prediction matches true label
        if item['predicted'] == item['label']:
            correct_predicted += 1  
        else:
            predicted_mismatch_idx.append(item['idx'])
            item['specific_mismatch'] = 1
        
        # Check consistency between LLM and specific model predictions
        if item['LLM_predicted'] is not None and item['LLM_predicted'] == item['predicted']:
            consistent += 1  
        else:
            inconsistent_idx.append(item['idx'])
            item['inconsistent'] = 1
            if 'specific_mismatch' in item and item['specific_mismatch'] == 1:
                overriden_acc_case += 1
                if 'llm_mismatch' in item and item['llm_mismatch'] == 1:
                    overriden_acc_case -= 1

    # Calculate metrics
    total = len(data)
    llm_accuracy = correct_llm / total * 100
    predicted_accuracy = correct_predicted / total * 100
    consistency_rate = consistent / total * 100
    overriden_acc = overriden_acc_case / (total - consistent) * 100 if (total - consistent) > 0 else 0

    print(f"LLM prediction accuracy: {llm_accuracy:.2f}%")
    print(f"Specific model prediction accuracy: {predicted_accuracy:.2f}%")
    print(f"Inconsistency rate between LLM and specific model: {100-consistency_rate:.2f}%")
    print(f"Indices with inconsistent predictions: {inconsistent_idx}")
    print(f"Total override cases: {total-consistent}")
    print(f"Correct override cases: {overriden_acc_case}")
    print(f"Override accuracy: {overriden_acc:.2f}%")

    return data

def main():
    args = parse_arguments()
    data = load_data(args.input_file)
    analyzed_data = analyze_predictions(data)
    save_data(args.output_file, analyzed_data)

if __name__ == "__main__":
    main()