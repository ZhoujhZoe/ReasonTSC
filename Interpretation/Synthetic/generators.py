import numpy as np
from serialization import SerializerSettings
from utils import save_to_jsonl

class TimeSeriesGenerator:
    """Base class for time series data generation."""
    
    def __init__(self, num_significant=10, num_non_significant=10, length=25):
        self.num_significant = num_significant
        self.num_non_significant = num_non_significant
        self.length = length
    
    def generate(self):
        """Generate time series data with alternating significant/non-significant samples."""
        significant = self._generate_significant()
        non_significant = self._generate_non_significant()
        
        time_series = []
        labels = []
        for i in range(max(self.num_significant, self.num_non_significant)):
            if i < self.num_significant:
                time_series.append(significant[i])
                labels.append("significant")
            if i < self.num_non_significant:
                time_series.append(non_significant[i])
                labels.append("non-significant")
        return time_series, labels
    
    def _generate_significant(self):
        raise NotImplementedError
        
    def _generate_non_significant(self):
        raise NotImplementedError

class TrendGenerator(TimeSeriesGenerator):
    """Generate time series with/without significant trends."""
    
    def _generate_significant(self):
        return [self._generate_trend() for _ in range(self.num_significant)]
    
    def _generate_non_significant(self):
        return [self._generate_noise() for _ in range(self.num_non_significant)]
    
    def _generate_trend(self, start=0, slope=1, noise_level=1):
        x = np.arange(self.length)
        trend = start + slope * x
        noise = np.random.normal(0, noise_level, size=self.length)
        return trend + noise
    
    def _generate_noise(self, baseline=10, noise_level=1):
        noise = np.random.normal(0, noise_level, size=self.length)
        return baseline + noise

class FrequencyGenerator(TimeSeriesGenerator):
    """Generate time series with/without significant frequencies."""
    
    def _generate_significant(self):
        return [self._generate_wave() for _ in range(self.num_significant)]
    
    def _generate_non_significant(self):
        return [self._generate_noise() for _ in range(self.num_non_significant)]
    
    def _generate_wave(self, amplitude=3, frequency=0.2, noise_level=0.01):
        x = np.arange(self.length)
        signal = amplitude * np.sin(2 * np.pi * frequency * x)
        noise = np.random.normal(0, noise_level, size=self.length)
        return signal + noise
    
    def _generate_noise(self, baseline=0, noise_level=3):
        noise = np.random.normal(0, noise_level, size=self.length)
        return baseline + noise

def generate_example_data():
    """Example data generation workflow."""
    generator = TrendGenerator(num_significant=200, num_non_significant=200, length=100)
    time_series, labels = generator.generate()
    time_series = [np.round(arr, 2) for arr in time_series]
    
    settings = SerializerSettings()
    str_slices = [serialize_arr(arr, settings) for arr in time_series]
    save_to_jsonl(time_series, str_slices, "./trend_synthesis.jsonl")
    print("Data written to JSONL")