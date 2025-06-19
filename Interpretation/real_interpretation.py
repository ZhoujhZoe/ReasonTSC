import openai
from openai import OpenAI
import json
import time
import argparse
import os

# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Pattern Analysis using GPT')
    parser.add_argument('--config_file', type=str, default='../ReasonTSC/multi_prompt_UEA.json', help='Path to configuration JSON file')
    parser.add_argument('--subset_id', type=int, help='ID of the subset to analyze')
    parser.add_argument('--subset_sample_file', type=str, help='Path to sample data JSON file')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o-mini', help='OpenAI GPT model to use')
    parser.add_argument('--output_file', type=str, help='Path to output JSON file')
    return parser.parse_args()
    
# Task descriptions
FIRST_ROUND_DESCRIPTION = '''
Your task is to analyze and determine whether there are any highly pronounced and distinctly typical temporal patterns across these categories.
Only if such patterns are exceptionally clear and consistently representative, mark it as 1; otherwise, mark it as 0.
'''

FIRST_ROUND_TASK = '''
### Analysis Task\nCompare and summarize the significant differences in the time series patterns across categories based on the following characteristics.
Break the series into meaningful segments (e.g. early, middle, late) if applicable. Only mark a characteristic as 1 if the differences are very clear and typical. Explicitly state if no differences are observed.
### Answer Format:\n- **Trend Differences**: 0/1. [Describe clear and typical trends (upward/downward) and how they differ across categories, or state if none are found.]
- **Cyclic behavior Differences**: 0/1. [Describe clear and typical differences in cyclic or periodic patterns, or state if none are found.]
- **Stationarity Differences**: 0/1. [Describe clear and typical stability or shifts in the time series, or state if none are found.]
- **Amplitude Differences**: 0/1. [Describe clear and typical constant or fluctuating amplitudes, or state if none are found.]
- **Rate of change Differences**: 0/1. [Describe clear and typical differences in the speed of change (rapid, moderate, slow), or state if none are found.]
- **Outliers Differences**: 0/1. [Identify clear and typical distinct outliers or anomalies, or state if none are found.]
- **Noise level Differences**: 0/1. [Describe clear and typical the amount of random fluctuations or noise across categories, or state if none are found.]
- **Volatility Differences**: 0/1. [Describe clear and typical differences in variability or fluctuations, or state if none are found.]
- **Structural Break Differences**: 0/1. [Identify clear and typical significant shifts or breaks in the time series, or state if none are found.]
- **Mean Level Differences**: 0/1. [Identify clear and typical the average values across categories, or state if none are found.]
'''

def load_json(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def initialize_openai_client():
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAI(api_key=api_key)

def gpt_chat(client, model, content, conversation):
    """Send request to OpenAI API and return response"""
    try:
        response = client.chat.completions.create(
            model=model, 
            temperature=0.2,
            messages=conversation + [{"role": "user", "content": content}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        time.sleep(5)
        return None


def get_round1_prompt_prefix(config_file, subset_id):
    """Generate the prefix for the first round prompt"""
    subset_file = load_json(config_file)
    subset = subset_file[subset_id]

    subset_description = subset["subset_description"]
    task_description = subset_description + " " + FIRST_ROUND_DESCRIPTION

    categories = subset["classes"]
    length = subset["length"]
    dataset_details = "### Dataset Details:\n- **Categories**: " + str(categories) + "\n- **Sequence Length**: " + str(length) + " time points"
    
    return task_description + '\n' + dataset_details + '\n'


def round_1(client, model, conversation, round1_prompt):
    round1_answer = gpt_chat(client, model, round1_prompt, conversation)
    if round1_answer:
        conversation.append({"role": "user", "content": round1_prompt})
        conversation.append({"role": "assistant", "content": round1_answer})
    return conversation


def main():
    args = parse_args()
    client = initialize_openai_client()
    round1_prompt_prefix = get_round1_prompt_prefix(args.config_file, args.subset_id)
    category_sample = load_json(args.subset_sample_file)

    with open(args.output_file, 'w') as f:
        f.write("[\n") 

        for idx in range(len(category_sample)):
            conversation = []
            sample_data = category_sample[idx]

            time_series_samples = "### Time Series Samples by Category:\n"
            for i in range(sample_data["classes"]):
                time_series_samples += f"{i+1}. **Category {i+1}**: {sample_data['sample'][i]}\n"

            round3_prompt = round1_prompt_prefix + time_series_samples + FIRST_ROUND_TASK
            conversation = round_1(client, args.gpt_model, conversation, round3_prompt)
            
            output_data = {
                "idx": sample_data["id"],
                "class": sample_data["classes"],
                "conversation": conversation
            }

            json.dump(output_data, f, indent=4)
            if idx < len(category_sample) - 1:
                f.write(",\n") 

            print(f"Processed sample {idx}")

        f.write("\n]")

if __name__ == "__main__":
    main()