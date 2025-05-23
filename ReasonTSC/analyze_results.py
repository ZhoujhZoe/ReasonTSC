import json
import re
import argparse
from typing import Dict, List, Any

class ResultAnalyzer:
    def __init__(self, input_file: str, output_file: str = None):
        """
        Initialize the analyzer with input and output file paths
        """
        self.input_file = input_file
        self.output_file = output_file or f"analyzed_{input_file}"
        self.data = []
        self.stats = {
            'correct_llm': 0,
            'correct_predicted': 0,
            'consistent': 0,
            'overriden_acc_case': 0,
            'llm_mismatch_idx': [],
            'predicted_mismatch_idx': [],
            'inconsistent_idx': []
        }

    def load_data(self) -> None:
        """Load JSON data from input file"""
        with open(self.input_file, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def extract_llm_prediction(self, content: str) -> int:
        """Extract LLM predicted label from content"""
        matches = re.findall(r'\*\*True Label.*(\d+)', content)
        return int(matches[-1]) if matches else None

    def extract_confidence(self, content: str) -> float:
        """Extract confidence score from content"""
        matches = re.findall(r'\*\*Confidence.*(\d+\.\d{2})', content)
        return float(matches[-1]) if matches else None

    def analyze_results(self) -> None:
        """Analyze the loaded data and calculate statistics"""
        for item in self.data:
            last_content = item['conversation'][-1]['content']
            
            if last_content is None:
                print(f"Skipping item with idx {item['idx']} due to None content")
                continue

            # Extract predictions and confidence
            item['LLM_predicted'] = self.extract_llm_prediction(last_content)
            item['LLM_confidence'] = self.extract_confidence(last_content)

            # Check LLM prediction accuracy
            if item['LLM_predicted'] == item['label']:
                self.stats['correct_llm'] += 1
            else:
                self.stats['llm_mismatch_idx'].append(item['idx'])
                item['llm_mismatch'] = 1

            # Check specific model prediction accuracy
            if item['predicted'] == item['label']:
                self.stats['correct_predicted'] += 1
            else:
                self.stats['predicted_mismatch_idx'].append(item['idx'])
                item['specific_mismatch'] = 1

            # Check consistency between LLM and specific model
            if item['LLM_predicted'] == item['predicted']:
                self.stats['consistent'] += 1
            else:
                self.stats['inconsistent_idx'].append(item['idx'])
                item['inconsistent'] = 1
                if item.get('specific_mismatch') and item.get('llm_mismatch'):
                    self.stats['overriden_acc_case'] += 1

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        total = len(self.data)
        return {
            'llm_accuracy': self.stats['correct_llm'] / total * 100,
            'predicted_accuracy': self.stats['correct_predicted'] / total * 100,
            'inconsistency_rate': 100 - (self.stats['consistent'] / total * 100),
            'overriden_accuracy': self.stats['overriden_acc_case'] / max(1, (total - self.stats['consistent'])) * 100
        }

    def save_results(self) -> None:
        """Save analyzed data to output file"""
        with open(self.output_file, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)

    def print_report(self, metrics: Dict[str, float]) -> None:
        """Print analysis report"""
        print(f"LLM prediction accuracy: {metrics['llm_accuracy']:.2f}%")
        print(f"Specific model accuracy: {metrics['predicted_accuracy']:.2f}%")
        print(f"Inconsistency rate: {metrics['inconsistency_rate']:.2f}%")
        print(f"Inconsistent indices: {self.stats['inconsistent_idx']}")
        print(f"Override cases: {len(self.data) - self.stats['consistent']}")
        print(f"Override accuracy: {metrics['overriden_accuracy']:.2f}%")

    def run_analysis(self) -> None:
        """Run complete analysis pipeline"""
        self.load_data()
        self.analyze_results()
        metrics = self.calculate_metrics()
        self.print_report(metrics)
        self.save_results()

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM prediction results')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    args = parser.parse_args()

    analyzer = ResultAnalyzer(args.input, args.output)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()