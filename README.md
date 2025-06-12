# ReasonTSC
ReasonTSC is a framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC.

Our work is detailed in the paper *"Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision"* available on [arXiv](https://arxiv.org/pdf/2506.00807). The complete code for ReasonTSC is currently under optimization, and we will release the updated version soon.

To set up the environment
```bash
pip install -r requirements.txt
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
UCR & UEA data interpretation
```bash
python analyze.py \
    --config ./multi_prompt_UEA.json \
    --subset_id 4 \
    --samples ./scripts/EpilepsyDimension1_category.json \
    --model gpt-4o-mini \
    --output ./results/PenDigitsDimension1_analysis.json
```

