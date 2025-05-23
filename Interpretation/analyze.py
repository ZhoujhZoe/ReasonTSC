import openai
from openai import OpenAI
import json
import time
import argparse
from typing import Dict, List, Any

class ReasonTSCAnalyzer:
    """
    A class to analyze time series data using LLM for interpretability experiments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.client = self._initialize_client()
        
        # Task descriptions
        self.first_round_description = ("Your task is to analyze and determine whether there are any highly pronounced "
                                       "and distinctly typical temporal patterns across these categories. Only if such "
                                       "patterns are exceptionally clear and consistently representative, mark it as 1; "
                                       "otherwise, mark it as 0.")
        
        self.first_round_task = """### Analysis Task
            Compare and summarize the significant differences in the time series patterns across categories based on the following characteristics. 
            Break the series into meaningful segments (e.g. early, middle, late) if applicable. Only mark a characteristic as 1 if the differences are very clear and typical. 
            Explicitly state if no differences are observed. 

            ### Answer Format:
            - **Trend Differences**: 0/1. [Describe clear and typical trends (upward/downward) and how they differ across categories, or state if none are found.]
            - **Cyclic behavior Differences**: 0/1. [Describe clear and typical differences in cyclic or periodic patterns, or state if none are found.]
            - **Stationarity Differences**: 0/1. [Describe clear and typical stability or shifts in the time series, or state if none are found.]
            - **Amplitude Differences**: 0/1. [Describe clear and typical constant or fluctuating amplitudes, or state if none are found.]
            - **Rate of change Differences**: 0/1. [Describe clear and typical differences in the speed of change (rapid, moderate, slow), or state if none are found.]
            - **Outliers Differences**: 0/1. [Identify clear and typical distinct outliers or anomalies, or state if none are found.]
            - **Noise level Differences**: 0/1. [Describe clear and typical the amount of random fluctuations or noise across categories, or state if none are found.]
            - **Volatility Differences**: 0/1. [Describe clear and typical differences in variability or fluctuations, or state if none are found.]
            - **Structural Break Differences**: 0/1. [Identify clear and typical significant shifts or breaks in the time series, or state if none are found.]
            - **Mean Level Differences**: 0/1. [Identify clear and typical the average values across categories, or state if none are found.]
        """

    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client with API key."""
        # Note: In production, use environment variables for API keys
        return OpenAI(
            api_key="your_api_key_here",  # Replace with proper secure handling
            base_url=""
        )

    @staticmethod
    def load_json(file_path: str) -> Any:
        """Load JSON data from file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def gpt_chat(self, content: str, conversation: List[Dict[str, str]]) -> str:
        """
        Send chat request to GPT model.
        
        Args:
            content: The message content to send
            conversation: Conversation history
            
        Returns:
            Response content from the model
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config['gpt_model'],
                temperature=0.2,
                messages=conversation + [{"role": "user", "content": content}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API request failed: {e}")
            time.sleep(5)
            return ""

    def get_round1_prompt_prefix(self) -> str:
        """Generate the prefix for the first round prompt."""
        subset = self.load_json(self.config['config_file'])[self.config['subset_id']]
        subset_description = subset["subset_description"]
        task_description = subset_description + " " + self.first_round_description

        categories = subset["classes"]
        length = subset["length"]
        dataset_details = ("### Dataset Details:\n"
                         f"- **Categories**: {str(categories)}\n"
                         f"- **Sequence Length**: {str(length)} time points")

        return task_description + '\n' + dataset_details + '\n'

    def round_1(self, conversation: List[Dict[str, str]], prompt: str) -> List[Dict[str, str]]:
        """
        Execute first round of analysis.
        
        Args:
            conversation: Conversation history
            prompt: The prompt to send
            
        Returns:
            Updated conversation history
        """
        answer = self.gpt_chat(prompt, conversation)
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ])
        return conversation

    def process_samples(self) -> None:
        """Main method to process all samples and save results."""
        round1_prompt_prefix = self.get_round1_prompt_prefix()
        category_sample = self.load_json(self.config['subset_sample_file'])

        results = []
        for idx, sample in enumerate(category_sample):
            conversation = []
            time_series_samples = "### Time Series Samples by Category:\n"
            
            for i in range(sample["classes"]):
                time_series_samples += f"{i+1}. **Category {i+1}**: {sample['sample'][i]}\n"

            full_prompt = round1_prompt_prefix + time_series_samples + self.first_round_task
            conversation = self.round_1(conversation, full_prompt)
            
            results.append({
                "idx": sample["id"],
                "class": sample["classes"],
                "conversation": conversation
            })
            
            print(f"Processed sample {idx + 1}/{len(category_sample)}")

        # Save all results at once
        with open(self.config['output_file'], 'w') as f:
            json.dump(results, f, indent=4)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ReasonTSC Time Series Analysis")
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    parser.add_argument('--subset_id', type=int, required=True, help='Subset ID to analyze')
    parser.add_argument('--samples', required=True, help='Path to sample data JSON file')
    parser.add_argument('--model', default="gpt-4o-mini", help='GPT model to use')
    parser.add_argument('--output', required=True, help='Output file path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = {
        'config_file': args.config,
        'subset_id': args.subset_id,
        'subset_sample_file': args.samples,
        'gpt_model': args.model,
        'output_file': args.output
    }
    
    analyzer = ReasonTSCAnalyzer(config)
    analyzer.process_samples()

if __name__ == "__main__":
    main()