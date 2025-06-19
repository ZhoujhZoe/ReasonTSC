import numpy as np
import random
import json
import argparse
from serialize import SerializerSettings, serialize_arr

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate synthetic time series data')
    parser.add_argument('--pattern_type', type=str, required=True,
                      choices=['trend', 'frequency', 'amplitude', 'pattern'],
                      help='Type of pattern to generate')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output JSONL file')
    parser.add_argument('--num_significant', type=int, default=200,
                      help='Number of significant pattern samples')
    parser.add_argument('--num_non_significant', type=int, default=200,
                      help='Number of non-significant pattern samples')
    parser.add_argument('--length', type=int, default=100,
                      help='Length of each time series')
    parser.add_argument('--shuffle', action='store_true',
                      help='Shuffle the output data')
    return parser.parse_args()

def get_generator(pattern_type):
    """Get the appropriate generation function based on pattern type"""
    generators = {
        'trend': generate_time_series_trend,
        'frequency': generate_time_series_frequency,
        'amplitude': generate_time_series_amplitude,
        'pattern': generate_time_series_pattern
    }
    return generators[pattern_type]


def save_to_jsonl(input_arrs, input_strs, file_path, shuffle=False):
    """
    Convert input arrays and strings to JSONL format with optional shuffling
    
    Args:
        input_arrs: List of numpy arrays
        input_strs: List of serialized strings
        file_path: Output file path
        shuffle: Whether to shuffle the data
    """
    if shuffle:
        combined = list(zip(input_arrs, input_strs))
        random.shuffle(combined)
        input_arrs, input_strs = zip(*combined)
    
    with open(file_path, "w") as f:
        for idx, (arr, string) in enumerate(zip(input_arrs, input_strs)):
            data = {
                "index": idx,
                "input_arr": arr.tolist(),
                "input_str": string
            }
            f.write(json.dumps(data) + "\n")

def generate_time_series_trend(num_significant=10, num_non_significant=10, length=25):
    """
        Generate time series data, with some having significant trends and others without significant trends, alternately arranged.

        Parameters:
            num_significant (int): The number of time series with significant trends.
            num_non_significant (int): The number of time series without significant trends.
            length (int): The length of each time series.

        Returns:
            list: A list of numpy arrays containing the generated time series, with significant and non-significant trends alternately arranged.
            list: A corresponding list of labels ("significant" or "non-significant").
    """
    def generate_significant_trend(length, start=0, slope=1, noise_level=1):
        x = np.arange(length)
        trend = start + slope * x
        noise = np.random.normal(0, noise_level, size=length)
        return trend + noise

    def generate_non_significant_trend(length, baseline=10, noise_level=1):
        noise = np.random.normal(0, noise_level, size=length)
        return baseline + noise

    significant_trends = [
        generate_significant_trend(length, start=np.random.randint(0, 5), slope=np.random.uniform(0.2, 0.4)) for _ in
        range(num_significant)]
    non_significant_trends = [generate_non_significant_trend(length, baseline=np.random.randint(5, 15)) for _ in
                              range(num_non_significant)]
    
    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_trends[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_trends[i])
            labels.append("non-significant")

    return time_series_data, labels


def generate_time_series_frequency(num_significant=10, num_non_significant=10, length=25):
    """
        Generate time series data, with some having significant frequency/cyclicity and others without significant frequency, alternately arranged.

        Parameters:
            num_significant (int): The number of time series with significant frequency.
            num_non_significant (int): The number of time series without significant frequency.
            length (int): The length of each time series.

        Returns:
            list: A list of numpy arrays containing the generated time series, with significant and non-significant frequencies alternately arranged.
            list: A corresponding list of labels ("significant" or "non-significant").
    """
    def generate_significant_frequency(length, amplitude=3, frequency=0.2, noise_level=0.01):
        x = np.arange(length)
        signal = amplitude * np.sin(2 * np.pi * frequency * x)
        noise = np.random.normal(0, noise_level, size=length)
        return signal + noise

    def generate_non_significant_frequency(length, baseline=10, noise_level=3):
        noise = np.random.normal(0, noise_level, size=length)
        return baseline + noise

    significant_frequencies = [
        generate_significant_frequency(length, amplitude=np.random.uniform(3, 4),
                                       frequency=np.random.uniform(0.1, 0.2))
        for _ in range(num_significant)]
    non_significant_frequencies = [generate_non_significant_frequency(length, baseline=0)
                                   for _ in range(num_non_significant)]

    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_frequencies[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_frequencies[i])
            labels.append("non-significant")

    return time_series_data, labels


def generate_time_series_amplitude(num_significant=10, num_non_significant=10, length=25):
    """
        Generate time series data, with some having significant amplitude variations and others without significant amplitude variations, alternately arranged.

        Parameters:
            num_significant (int): The number of time series with significant amplitude.
            num_non_significant (int): The number of time series without significant amplitude.
            length (int): The length of each time series.

        Returns:
            list: A list of numpy arrays containing the generated time series, with significant and non-significant amplitudes alternately arranged.
            list: A corresponding list of labels ("significant" or "non-significant").
    """
    def generate_significant_amplitude(length, base_frequency=0.2, amplitude=2, noise_level=0.1):
        x = np.arange(length)
        signal = amplitude * np.sin(2 * np.pi * base_frequency * x)
        noise = np.random.normal(0, noise_level, size=length)
        return signal + noise

    def generate_non_significant_amplitude(length, baseline=0, noise_level=1):
        return np.random.normal(baseline, noise_level, size=length)

    significant_amplitudes = [
        generate_significant_amplitude(length, base_frequency=np.random.uniform(0.1, 0.2),
                                       amplitude=np.random.uniform(4, 5))
        for _ in range(num_significant)]
    non_significant_amplitudes = [generate_non_significant_amplitude(length) for _ in range(num_non_significant)]

    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_amplitudes[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_amplitudes[i])
            labels.append("non-significant")

    return time_series_data, labels


def generate_time_series_pattern(num_significant=10, num_non_significant=10, length=50):
    """
        Generate time series data, with some having significant patterns/shapes and others without significant patterns, alternately arranged.

        Parameters:
            num_significant (int): The number of time series with significant patterns.
            num_non_significant (int): The number of time series without significant patterns.
            length (int): The length of each time series.

        Returns:
            list: A list of numpy arrays containing the generated time series, with significant and non-significant patterns alternately arranged.
            list: A corresponding list of labels ("significant" or "non-significant").
    """
    def generate_significant_pattern(length, beta_0 = 10, beta_1 = 0.05, A_1 = 5, f_1 = 1 / 12, phi_1 = np.pi / 2, A_2 = 3, f_2 = 1 / 20, phi_2 = np.pi / 3, noise_level=0):
        x = np.arange(length)
        trend = beta_0 + beta_1 * x
        seasonality = A_1 * np.sin(2 * np.pi * f_1 * x + phi_1)
        cyclical = A_2 * np.cos(2 * np.pi * f_2 * x + phi_2)
        noise = np.random.normal(0, noise_level, size=length)
        return trend + seasonality + cyclical + noise

    def generate_non_significant_pattern(length, noise_level=6):
        return np.random.normal(10, noise_level, size=length)

    significant_patterns = [
        generate_significant_pattern(length, beta_0 = np.random.randint(5, 15), beta_1 = np.random.uniform(0.01, 0.1), A_1 = np.random.uniform(5, 8), f_1 = np.random.uniform(0.08, 0.1), phi_1 = np.random.uniform(0, np.pi), A_2 = np.random.uniform(1, 2), f_2 = np.random.uniform(0.02, 0.04), phi_2 = np.random.uniform(0, np.pi))
        for _ in range(num_significant)]
    non_significant_patterns = [generate_non_significant_pattern(length) for _ in range(num_non_significant)]

    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_patterns[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_patterns[i])
            labels.append("non-significant")

    return time_series_data, labels


def main():
    args = parse_args()
    
    # Get the appropriate generator function
    generator = get_generator(args.pattern_type)
    
    # Generate time series data
    time_series_list, labels_list = generator(
        num_significant=args.num_significant,
        num_non_significant=args.num_non_significant,
        length=args.length
    )
    
    # Round values and serialize
    time_series_list = [np.round(arr, 2) for arr in time_series_list]
    str_slices = [serialize_arr(arr, SerializerSettings()) for arr in time_series_list]
    
    # Save to JSONL
    save_to_jsonl(time_series_list, str_slices, args.output_file, args.shuffle)
    
    print(f"Generated {len(time_series_list)} samples with {args.pattern_type} patterns")
    print(f"Data saved to {args.output_file}")

if __name__ == "__main__":
    main()