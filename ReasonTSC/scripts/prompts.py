
first_round_description = """
You will be provided with two time series samples from each category of this dataset.
Your first task is to analyze and compare the significant pattern differences across these categories.
"""

second_round_description = """
Your second task is to analyze the time series data and refine your understanding 
based on the classification results and logits (model probabilities for each category) provided by a domain-specific model.
"""

third_round_description = """
### Task Description:\nBased on your refined understanding, your third task is to perform the time series classification task on the new data sample.
You will use your updated analysis of time series patterns along with the result and category logits (model probabilities for each category) from the domain-specific model to make a final classification decision.
"""

first_round_task = """
### Analysis Task\nCompare and summarize the significant differences in the time series patterns across categories based on the following characteristics.
Explicitly state if no differences are observed. Break the series into meaningful segments (e.g., early, middle, late) if applicable.\n
### Answer Format:
- **Trend Differences:**: [Describe trends (upward/downward) and how trends differ across categories, or state if no trends are observed.]
- **Cyclic behavior Differences**: [Describe differences in cyclic or periodic patterns, or state if none are found.]
- **Stationarity Differences**: [Describe stability or shifts in the time series, or state if none are found.]
- **Amplitude Differences**: [Compare constant or fluctuating amplitudes, or state if no differences]
- **Rate of change Differences**: [describe the speed of change across categories (rapid, moderate, slow), or state if none are found.]
- **Outliers Differences**: [Identify distinct outliers or anomalies, or state if none are found.]
"""

second_round_task = """
### Analysis Task:\nRefine your understanding of the time series patterns, considering the model's classification results and logits. Identify any necessary adjustments to your initial analysis.
### Answer Format:
- **Classification Analysis:**: [Evaluate the logits' confidence and alignment with categories.]
- **Time Series Understanding Adjustment:**: [Adjust your understanding of time series patterns based on the model's results.]
"""

third_round_task = """
Please think step by step: 
1. **Analyze the Time Series Pattern**: [Examine the time series data for trends, cyclic behavior, stationarity, amplitude, rate of change, and outliers. Compare these characteristics across the categories to identify any significant patterns or differences.]
2. **Interpret the Model's Results**: [Evaluate the model's classification result and logits. Assess the confidence level of the model's prediction and how well it aligns with the observed time series patterns.]
3. **Make a Preliminary Prediction**: [Based on your analysis of the time series pattern and the model's results, make an initial classification decision. Provide a brief explanation for this decision.]
4. **Review Alternative Classifications**: [Consider if there are any other plausible categories that could fit the observed time series pattern. Evaluate the strengths and weaknesses of these alternative classifications compared to your initial prediction.]
5. **Final Classification Decision**: [After reviewing all possibilities, make your final classification decision.]
**Answer Format**:
- **True Label**: [Your Final Classification Result]
- **Confidence**: [Your Classification Confidence 0.XX]
"""