# ReasonTSC
A novel framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC.

## Installation
To set up the environment, run:
```bash
pip install -r requirements.txt
```
Configure your OpenAI API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Time Series Foundation Models (TSFMs)
Model Options
* MOMENT time series foundation model: https://github.com/moment-timeseries-foundation-model/moment
* Chronos Forecasting Model for Embedding: https://github.com/amazon-science/chronos-forecasting
### Using Chronos Embeddings with SVM
```bash
python chronos_embeddings.py \
    --model_path ./path/to/chronos-model \
    --output_file ./results/predictions.txt \
    --batch_size 8 \
    --pooling mean
```
Arguments:

* --model_path: Path to Chronos model
* --output_file: Path to save predictions (default: ./output/predictions.txt)
* --batch_size: Batch size for embedding generation (default: 1)
* --pooling: Embedding pooling strategy - 'mean' or 'first' (default: mean)

### Fine-tuning MOMENT for classification
```bash
python MOMENT_fullfinetune.py \
    --num_class 5 \
    --model_path "./path/to/MOMENT" \
    --output_file "./MOMENT_output.txt" \
    --epochs 20
```

## Dataset Preparation
### Dataset Sources
Download UCR/UEA datasets from: 
https://www.timeseriesclassification.com/ 
### Data Processing
Convert time series data from .ts format to JSON:
```bash
python process_dataset.py \
        --ts_file ./EpilepsyDimension1_TEST.ts \
        --json_file ./EpilepsyDimension1_TEST.json
```
Convert TSFM model output to JSON:
```bash
python process_TSFM.py \
    --txt_file ./raw_output/Chronos_EpilepsyDimension1.txt \
    --json_file ./processed_output/Chronos_EpilepsyDimension1.json
```

## Running ReasonTSC
```bash
python main.py --config config.json
```
Arguments:

* --config_file: Contains dataset and framework information
* --subset_id: Matches the dataset ID in config_file
* --specific_model: Choose between moment_fullfinetune or chronos
* specific_model_output_file: Processed TSFM classification output
* subset_sample_file: Processed UCR/UEA dataset
* gpt_model: LLM selection for ReasonTSC
* output_file: Path to save LLM output

Analyze results:
```bash
python analyze_results.py --input output.json
```

## Interpretation
### Synthetic data interpretation
Process the generated data into comparison format:
```bash
python -m data_generation.utils process_jsonl input.jsonl output.json
```
Evaluate samples using GPT-4:
```bash
from evaluation.llm_evaluator import LLMEvaluator, EvaluationConfig
config = EvaluationConfig(model="gpt-4")
evaluator = LLMEvaluator(config)
results = evaluator.evaluate_samples("synthesis_pattern.json", "pattern")
```
### UCR & UEA data interpretation
```bash
python analyze.py \
    --config ./multi_prompt_UEA.json \
    --subset_id 4 \
    --samples ./scripts/EpilepsyDimension1_category.json \
    --model gpt-4o-mini \
    --output ./results/PenDigitsDimension1_analysis.json
```

