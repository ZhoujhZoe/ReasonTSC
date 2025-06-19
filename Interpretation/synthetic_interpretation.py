import openai
from openai import OpenAI
import json
import time
import argparse
import os

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Time Series Pattern Analysis')
    parser.add_argument('--gpt_model', type=str, help='LLM model to use')
    parser.add_argument('--sample_file', type=str, required=True, 
                      help='Path to sample data JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output JSON file')
    parser.add_argument('--prompt_type', type=str, required=True,
                      choices=['trend', 'frequency', 'amplitude', 'pattern'],
                      help='Type of pattern analysis to perform')
    return parser.parse_args()

TREND_PREFIX = """
Compare the two provided time series samples and select the one that demonstrates a more typical and well-defined trend pattern,
specifically a sustained and clear directional trend (either upward or downward) throughout the series.
"""
FREQUENCY_PREFIX = """
Compare the two provided time series samples and select the one that exhibits a more typical and well-defined frequency or cyclical pattern,
characterized by consistent and regular periodic behavior or repetitive cycles throughout the series.
"""
AMPLITUDE_PREFIX = """
Compare the two provided time series samples and select the one that demonstrates a more typical and well-defined amplitude pattern,
characterized by consistent and pronounced variations in value range, indicative of strong oscillations or signal intensity.
"""
PATTERN_PREFIX = """
Compare the two provided time series samples and select the one that exhibits more 
typical and well-defined patterns, such as trends, seasonality, or cyclical behavior.
"""
ANSWER_FORMAT = """
### Answer Format:
- **Option**: [Case A or Case B]
- **Explanation**: [Reason for choosing this time series sample and the specific pattern observed]
"""

def get_prompt_prefix(prompt_type):
    """Get the appropriate prompt prefix based on type"""
    prefixes = {
        'trend': TREND_PREFIX,
        'frequency': FREQUENCY_PREFIX,
        'amplitude': AMPLITUDE_PREFIX,
        'pattern': PATTERN_PREFIX
    }
    return prefixes.get(prompt_type, PATTERN_PREFIX)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def initialize_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def gpt_chat(client, model, content):
    """Send request to OpenAI API and return response"""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"role": "user", "content": content}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        time.sleep(5)
        return None


def llm_chat(client, model, conversation, prompt):
    """Execute LLM conversation and update conversation history"""
    llm_answer = gpt_chat(client, model, prompt)
    if llm_answer:
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": llm_answer})
    return conversation


def main():
    args = parse_args()
    client = initialize_openai_client()
    sample_data = load_json(args.sample_file)
    prompt_prefix = get_prompt_prefix(args.prompt_type)

    with open(args.output_file, 'w') as f:
        f.write("[\n")
        
        for idx, sample in enumerate(sample_data):
            conversation = []
            time_series = f"- **Case A**: {sample['sampleA']}\n- **Case B**: {sample['sampleB']}"
            prompt = prompt_prefix + '\n' + time_series + '\n' + ANSWER_FORMAT
            
            conversation = llm_chat(client, args.gpt_model, conversation, prompt)
            
            output_data = {
                "idx": idx,
                "label": sample["answer"],
                "conversation": conversation
            }

            json.dump(output_data, f, indent=4)
            if idx < len(sample_data) - 1:
                f.write(",\n")
            
            print(f"Processed sample {idx}")

        f.write("\n]")

if __name__ == "__main__":
    main()