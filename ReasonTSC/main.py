import openai
from openai import OpenAI
import json
import time
import argparse
import os
from scripts.prompts import (
    first_round_description,
    second_round_description,
    third_round_description,
    first_round_task,
    second_round_task,
    third_round_task
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='ReasonTSC')
    parser.add_argument('--config_file', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--subset_id', type=int, required=True, help='Subset ID to process')
    parser.add_argument('--specific_model', type=str, required=True, help='plug-in model')
    parser.add_argument('--specific_model_output_file', type=str, required=True, help='Path to plug-in model results')
    parser.add_argument('--subset_sample_file', type=str, required=True, help='Path to subset dataset')
    parser.add_argument('--gpt_model', type=str, required=True, help='GPT model to use')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def initialize_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def gpt_chat(client, model, content, conversation, temperature=0.2):
    try:
        response = client.chat.completions.create(
            model=model, 
            temperature=temperature,
            messages=conversation + [{"role": "user", "content": content}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        time.sleep(5)
        return None

def get_round1_prompt(config_file, subset_id):
    subset_file = load_json(config_file)
    subset = subset_file[subset_id]

    subset_description = subset["subset_description"]
    task_description = subset_description + " " + first_round_description

    categories = subset["classes"]
    length = subset["length"]
    dataset_details = "### Dataset Details:\n- **Categories**: " + str(categories) + "\n- **Sequence Length**: " + str(length) + " time points"
    
    time_series_samples = "### Time Series Samples (2 Samples per Category):\n"
    samples1 = subset["categories"]["sample1"]
    samples2 = subset["categories"]["sample2"]
    for i in range(categories):
        time_series_samples += f"{i+1}. **Category {i+1}**: \n- **Sample 1**: {samples1[i]}\n- **Sample 2**: {samples2[i]}\n"

    return task_description + '\n' + dataset_details + '\n' + time_series_samples + first_round_task


def round_1(client, gpt_model, conversation, round1_prompt):
    round1_answer = gpt_chat(client, gpt_model, round1_prompt, conversation)
    if round1_answer:
        conversation.append({"role": "user", "content": round1_prompt})
        conversation.append({"role": "assistant", "content": round1_answer})
    return conversation


def get_round2_prompt(config_file, subset_id, specific_model):
    subset_file = load_json(config_file)
    subset = subset_file[subset_id]

    subset_description = subset["subset_description"]
    task_description = subset_description + " " + second_round_description

    categories = subset["classes"]
    length = subset["length"]
    dataset_details = "### Dataset Details:\n- **Categories**: " + str(categories) + "\n- **Sequence Length**: " + str(length) + " time points"
    accuracy = subset["model_accuracy"][specific_model]
    model_details = "### Model Details:\n- **Classification Accuracy**: " + accuracy
    details = dataset_details + '\n' + model_details

    classification_examples = "### Classification Examples:\n"
    true_samples = subset["model_samples"][specific_model]["true_samples"]
    false_samples = subset["model_samples"][specific_model]["false_samples"]
    classification_examples += f"- **Case {1}**: True Label: {true_samples[0]['label']}, Model Result: {true_samples[0]['predicted']}, Category Logits: {true_samples[0]['logits']}, Time Series: [{true_samples[0]['time_series']}]\n"
    classification_examples += f"- **Case {2}**: True Label: {false_samples[0]['label']}, Model Result: {false_samples[0]['predicted']}, Category Logits: {false_samples[0]['logits']}, Time Series: [{false_samples[0]['time_series']}]\n"
    classification_examples += f"- **Case {3}**: True Label: {false_samples[1]['label']}, Model Result: {false_samples[1]['predicted']}, Category Logits: {false_samples[1]['logits']}, Time Series: [{false_samples[1]['time_series']}]\n"

    return task_description + '\n' + details + '\n' + classification_examples + second_round_task


def round_2(client, gpt_model, conversation, round2_prompt):
    round2_answer = gpt_chat(client, gpt_model, round2_prompt, conversation)
    if round2_answer:
        conversation.append({"role": "user", "content": round2_prompt})
        conversation.append({"role": "assistant", "content": round2_answer})
    return conversation


def get_round3_prompt_prefix(config_file, subset_id, specific_model):
    subset_file = load_json(config_file)
    subset = subset_file[subset_id]

    categories = subset["classes"]
    length = subset["length"]
    dataset_details = "### Dataset Details:\n- **Categories**: " + str(categories) + "\n- **Sequence Length**: " + str(length) + " time points"
    accuracy = subset["model_accuracy"][specific_model]
    model_details = "### Model Details:\n- **Classification Accuracy**: " + accuracy
    details = dataset_details + '\n' + model_details

    return third_round_description + '\n' + details


def round_3(client, gpt_model, conversation, round3_prompt):
    round3_answer = gpt_chat(client, gpt_model, round3_prompt, conversation)
    if round3_answer:
        conversation.append({"role": "user", "content": round3_prompt})
        conversation.append({"role": "assistant", "content": round3_answer})
    return conversation


def main():
    args = parse_arguments()
    client = initialize_openai_client()
    
    round1_prompt = get_round1_prompt(args.config_file, args.subset_id)
    round2_prompt = get_round2_prompt(args.config_file, args.subset_id, args.specific_model)
    round3_prompt_prefix = get_round3_prompt_prefix(args.config_file, args.subset_id, args.specific_model)

    specific_model_output = load_json(args.specific_model_output_file)
    subset_sample = load_json(args.subset_sample_file)

    with open(args.output_file, 'w') as f:
        f.write("[\n") 

        for idx in range(len(specific_model_output)):
            conversation = []
            conversation = round_1(client, args.gpt_model, conversation, round1_prompt)
            conversation = round_2(client, args.gpt_model, conversation, round2_prompt)

            sample_id = specific_model_output[idx]["id"]
            label = specific_model_output[idx]["label"]
            predicted = specific_model_output[idx]["predicted"]
            logits = specific_model_output[idx]["logits"]
            sample = subset_sample[sample_id]["sample"]
            task_prompt = "### Classification Task:\n" + f"- **Task**: Model Result: {predicted}, Category Logits: {logits}, Time Series: [{sample}]\n"

            round3_prompt = round3_prompt_prefix + '\n' + task_prompt + third_round_task

            conversation = round_3(client, args.gpt_model, conversation, round3_prompt)
            
            output_data = {
                    "idx": sample_id,
                    "label": label,
                    "predicted": predicted,
                    "conversation": conversation
            }

            if idx < len(specific_model_output) - 1:
                json.dump(output_data, f, indent=4)
                f.write(",\n") 
            else:
                json.dump(output_data, f, indent=4)

            print(f"sample {idx} processed")

if __name__ == "__main__":
    main()