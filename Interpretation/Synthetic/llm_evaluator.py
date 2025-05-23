import json
from dataclasses import dataclass
from typing import Literal
import openai
import time

@dataclass
class EvaluationConfig:
    model: str = "gpt-4"
    temperature: float = 0.2
    max_retries: int = 3
    retry_delay: int = 5

class LLMEvaluator:
    """Evaluate time series patterns using LLMs."""
    
    PROMPT_TEMPLATES = {
        'trend': "Compare the two provided time series samples and select the one that demonstrates a more typical and well-defined trend pattern, specifically a sustained and clear directional trend (either upward or downward) throughout the series.", 
        'frequency': "Compare the two provided time series samples and select the one that exhibits a more typical and well-defined frequency or cyclical pattern, characterized by consistent and regular periodic behavior or repetitive cycles throughout the series.",
        'amplitude': "Compare the two provided time series samples and select the one that demonstrates a more typical and well-defined amplitude pattern, characterized by consistent and pronounced variations in value range, indicative of strong oscillations or signal intensity.",
        'pattern': "Compare the two provided time series samples and select the one that exhibits more typical and well-defined patterns, such as trends, seasonality, or cyclical behavior."
    }
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = openai.OpenAI()
    
    def evaluate_samples(self, sample_file: str, pattern_type: Literal['trend', 'frequency', 'amplitude', 'pattern']):
        """Evaluate samples from JSON file."""
        with open(sample_file) as f:
            samples = json.load(f)
        
        results = []
        for sample in samples:
            response = self._get_llm_response(
                sample['sampleA'],
                sample['sampleB'],
                pattern_type
            )
            results.append({
                **sample,
                "response": response
            })
        return results
    
    def _get_llm_response(self, sampleA, sampleB, pattern_type):
        prompt = self._build_prompt(sampleA, sampleB, pattern_type)
        
        for _ in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API request failed: {e}")
                time.sleep(self.config.retry_delay)
        return None
    
    def _build_prompt(self, sampleA, sampleB, pattern_type):
        return (
            f"{self.PROMPT_TEMPLATES[pattern_type]}\n"
            f"- Case A: {sampleA}\n"
            f"- Case B: {sampleB}\n"
            "### Answer Format:\n"
            "- Option: [A or B]\n"
            "- Explanation: [Your reasoning]"
        )