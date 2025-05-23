import json
import time
from typing import List, Dict, Any
from openai import OpenAI
from prompts import PROMPTS

class ReasonTSC:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ReasonTSC classifier with configuration.
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.get("openai_api_key", ""),  # Should be passed via config or env variable
            base_url=config.get("openai_base_url", "https://api.openai.com/v1")
        )
        
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def gpt_chat(self, content: str, conversation: List[Dict[str, str]]) -> str:
        """
        Send message to GPT model and get response.
        
        Args:
            content: The message content to send
            conversation: Conversation history
            
        Returns:
            Response content from GPT
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["gpt_model"],
                temperature=0.2,
                messages=conversation + [{"role": "user", "content": content}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API request failed: {e}")
            time.sleep(5)
            return ""
    
    def build_round_prompt(self, round_num: int, subset_data: Dict[str, Any]) -> str:
        """
        Construct prompt for the specified round.
        
        Args:
            round_num: Round number (1, 2, or 3)
            subset_data: Dataset subset information
            
        Returns:
            Constructed prompt string
        """
        if round_num == 1:
            return self._build_round1_prompt(subset_data)
        elif round_num == 2:
            return self._build_round2_prompt(subset_data)
        else:
            return self._build_round3_prompt_prefix(subset_data)
    
    def _build_round1_prompt(self, subset: Dict[str, Any]) -> str:
        """Build prompt for round 1 analysis."""
        task_description = subset["subset_description"] + " " + PROMPTS["first_round_description"]
        
        details = (
            f"### Dataset Details:\n"
            f"- **Categories**: {subset['classes']}\n"
            f"- **Sequence Length**: {subset['length']} time points"
        )
        
        samples1 = subset["categories"]["sample1"]
        samples2 = subset["categories"]["sample2"]
        time_series_samples = "### Time Series Samples (2 Samples per Category):\n"
        for i in range(subset["classes"]):
            time_series_samples += (
                f"{i+1}. **Category {i+1}**: \n"
                f"- **Sample 1**: {samples1[i]}\n"
                f"- **Sample 2**: {samples2[i]}\n"
            )
            
        return f"{task_description}\n{details}\n{time_series_samples}{PROMPTS['first_round_task']}"
    
    def _build_round2_prompt(self, subset: Dict[str, Any]) -> str:
        """Build prompt for round 2 analysis."""
        task_description = subset["subset_description"] + " " + PROMPTS["second_round_description"]
        
        details = (
            f"### Dataset Details:\n"
            f"- **Categories**: {subset['classes']}\n"
            f"- **Sequence Length**: {subset['length']} time points\n"
            f"### Model Details:\n"
            f"- **Classification Accuracy**: {subset['model_accuracy'][self.config['specific_model']]}"
        )
        
        true_samples = subset["model_samples"][self.config["specific_model"]]["true_samples"]
        false_samples = subset["model_samples"][self.config["specific_model"]]["false_samples"]
        
        classification_examples = (
            "### Classification Examples:\n"
            f"- **Case 1**: True Label: {true_samples[0]['label']}, Model Result: {true_samples[0]['predicted']}, "
            f"Category Logits: {true_samples[0]['logits']}, Time Series: [{true_samples[0]['time_series']}]\n"
            f"- **Case 2**: True Label: {false_samples[0]['label']}, Model Result: {false_samples[0]['predicted']}, "
            f"Category Logits: {false_samples[0]['logits']}, Time Series: [{false_samples[0]['time_series']}]\n"
            f"- **Case 3**: True Label: {false_samples[1]['label']}, Model Result: {false_samples[1]['predicted']}, "
            f"Category Logits: {false_samples[1]['logits']}, Time Series: [{false_samples[1]['time_series']}]\n"
        )
        
        return f"{task_description}\n{details}\n{classification_examples}{PROMPTS['second_round_task']}"
    
    def _build_round3_prompt_prefix(self, subset: Dict[str, Any]) -> str:
        """Build prefix for round 3 prompt."""
        details = (
            f"### Dataset Details:\n"
            f"- **Categories**: {subset['classes']}\n"
            f"- **Sequence Length**: {subset['length']} time points\n"
            f"### Model Details:\n"
            f"- **Classification Accuracy**: {subset['model_accuracy'][self.config['specific_model']]}"
        )
        return f"{PROMPTS['third_round_description']}\n{details}"
    
    def process_round(self, conversation: List[Dict[str, str]], prompt: str) -> List[Dict[str, str]]:
        """
        Process a single round of conversation.
        
        Args:
            conversation: Current conversation history
            prompt: Prompt for this round
            
        Returns:
            Updated conversation history
        """
        answer = self.gpt_chat(prompt, conversation)
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ])
        return conversation
    
    def run(self):
        """Execute the full classification pipeline."""
        subset_data = self.load_json(self.config["config_file"])
        subset = subset_data[self.config["subset_id"]]
        
        model_output = self.load_json(self.config["specific_model_output_file"])
        test_samples = self.load_json(self.config["subset_sample_file"])
        
        results = []
        
        for idx, sample_data in enumerate(model_output):
            conversation = []
            
            # Round 1
            round1_prompt = self.build_round_prompt(1, subset)
            conversation = self.process_round(conversation, round1_prompt)
            
            # Round 2
            round2_prompt = self.build_round_prompt(2, subset)
            conversation = self.process_round(conversation, round2_prompt)
            
            # Round 3
            round3_prefix = self.build_round_prompt(3, subset)
            task_prompt = (
                "### Classification Task:\n"
                f"- **Task**: Model Result: {sample_data['predicted']}, "
                f"Category Logits: {sample_data['logits']}, "
                f"Time Series: [{test_samples[sample_data['id']]['sample']}]\n"
            )
            round3_prompt = f"{round3_prefix}\n{task_prompt}{PROMPTS['third_round_task']}"
            conversation = self.process_round(conversation, round3_prompt)
            
            results.append({
                "idx": sample_data["id"],
                "label": sample_data["label"],
                "predicted": sample_data["predicted"],
                "conversation": conversation
            })
            
            print(f"Processed sample {idx}")
        
        # Save results
        with open(self.config["output_file"], 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Example configuration - should be passed via command line or config file
    config = {
        "config_file": "./multi_prompt_UEA.json",
        "subset_id": 3,
        "specific_model": "chronos", # "chronos" or "moment_fullfinetune"
        "specific_model_output_file": "./scripts/Chronos_LibrasDimension1_gpt4omini.json",
        "subset_sample_file": "./scripts/LibrasDimension1_TEST.json",
        "gpt_model": "gpt-4o-mini",
        "output_file": "./output_file.json",
        "openai_api_key": "",  # Should be set via environment variable
        "openai_base_url": ""
    }
    
    classifier = ReasonTSC(config)
    classifier.run()