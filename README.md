# ReasonTSC
ReasonTSC is a framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC.

Our work is detailed in the paper *"Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision"* available on [arXiv](https://arxiv.org/pdf/2506.00807).

To set up the environment
```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key-here'
```

## Data Preparation
### Dataset Source
We benchmark ReasonTSC using the [UCR/UEA Time Series Classification Archive](https://www.timeseriesclassification.com/), a standard dataset collection for evaluating classification algorithms. It covers diverse scenarios with varying numbers of classes.

### Dataset processing
- The raw dataset is provided in `.ts` format. We convert the time series data from `.ts` to `JSON` for further processing. 
- Since time series samples often have long sequence lengths and LLMs are insensitive to high-precision decimals, we truncate values to `three decimal places` to optimize context window usage.
- We use the `first dimension` of the multivariate UEA datasets to address the token limit restrictions imposed by LLM input queries.
```bash
cd ./Data
python process_dataset.py
```

### TSFM output processing
The raw classification results (including predictions and per-category logits) of TSFMs are in the `.txt` format, we also convert it to `JSON` for ReasonTSC. 
```bash
cd ./Data
python process_TSFM.py
```
The resulting JSON structure
```json
[
    {
        "id": 0,
        "label": 10,
        "predicted": 10,
        "logits": "[8.3, 3.76, 6.21, 3.73, 0.69, 2.73, 0.7, 1.71, 7.29, 9.32]"
    }
]
```

## Time Series Foundation Models (TSFMs)
We select two prominent time series foundation models as the plug-in classifiers: 
- [MOMENT](https://github.com/moment-timeseries-foundation-model/moment). An encoder-only T5-based model, fully fine-tuned per task.

  Model used: [MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large) (341M).
- [Chronos](https://github.com/amazon-science/chronos-forecasting). An encoder-decoder model originally for forecasting; we adapt its pretrained encoder to extract embeddings for an SVM classifier.
  
   Model used: [chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large).
- ReasonTSC can effectively leverage LLMs' understanding of time series patterns through multi-turn reasoning to correct incorrect predictions by plug-in models. The framework is extensible to other TSFMs.

### Fine-tuning MOMENT for classification
```bash
cd ./TSFM
python MOMENT_fullfinetune.py \
    --num-class 5 \
    --epochs 20 \
    --output-file ./results.txt \
    --model-path /path/to/MOMENT
```

### Training an SVM-based classifier with Pre-trained Chronos Embeddings
```bash
cd ./TSFM
python chronos_embeddings.py \
    --output_file ./output.txt \
    --model_path chronos-model-path \
    --train_file DodgerLoopDay_TRAIN.ts \
    --test_file DodgerLoopDay_TEST.ts
```

## Running ReasonTSC 
```bash
cd ./ReasonTSC
python main.py \
  --config_file ./scripts/multi_prompt_UEA.json \
  --subset_id 3 \
  --specific_model chronos \ # Options: moment/chronos
  --specific_model_output_file ./scripts/Chronos_LibrasDimension1.json \
  --subset_sample_file ./scripts/LibrasDimension1_TEST.json \
  --gpt_model gpt-4o-mini \
  --output_file ./LibrasDimension1_gpt4o_chronos.json
```
**Arguments Explanation:**
| Argument | Description |
|----------|-------------|
| `--config_file` | Path to JSON configuration file containing dataset and framework specifications |
| `--subset_id` | Numeric identifier for the target dataset subset in config_file |
| `--specific_model` | TSFM selection (`moment` or `chronos`) |
| `--specific_model_output_file` | Path to plug-in TSFM results |
| `--subset_sample_file` | Path to processed UCR/UEA dataset samples |
| `--gpt_model` | LLM selection for ReasonTSC analysis |
| `--output_file` | Path for saving LLM output results |

**Analyzing Results:**
```bash
python analyze_results.py \
  --input_file ./LLM_outputs.json \
  --output_file ./processed_LLM_outputs.json
```
















## TS Pattern Interpretation Study
### Interpreting pattens on synthetic data
```bash
cd ./Interpretation
# Generate synthetic testing dataset
python synthetic_generating.py \
    --pattern_type pattern \
    --output_file ./synthesis_trend.jsonl \
    --num_significant 150 \
    --num_non_significant 150 \
    --length 80 \
    --shuffle
    
# Convert to multiple-choice format
python process_data.py \
    --input_jsonl ./synthesis_trend.jsonl \
    --output_json ./synthesis_trend.json

# Run pattern interpretation
python synthetic_interpretation.py \
    --gpt_model gpt-4 \
    --sample_file ./synthetic_dataset/synthesis_trend.json \
    --output_file trend_results.json \
    --prompt_type trend # Options: trend/frequency/amplitude/pattern
```
### Interpreting pattens on UCR/UEA Archive
```bash
# Run UCR/UEA pattern interpretation
python real_interpretation.py \
    --config_file ../ReasonTSC/multi_prompt_UEA.json \
    --subset_id 4 \
    --subset_sample_file ./EpilepsyDimension1_category.json \
    --gpt_model gpt-4o-mini \
    --output_file ./output.json

# Analysis LLM outputs
python output_analysis.py \
    --input_file ./input_file.json \
    --output_file ./output_file.json
```


