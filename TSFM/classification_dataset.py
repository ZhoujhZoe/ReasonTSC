import numpy as np
from sklearn.preprocessing import StandardScaler
from data import load_from_tsfile
from typing import Tuple


class ClassificationDataset:
    """Dataset class for time series classification tasks."""
    
    def __init__(self, data_split: str = "train", seq_len: int = 512):
        """
        Initialize dataset.
        
        Args:
            data_split: 'train' or 'test'
            seq_len: Length to pad/truncate time series to
        """
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.data_split = data_split
        
        # Initialize paths (should be parameterized in real use)
        self.train_path = "./data/DodgerLoopDay_TRAIN.ts"
        self.test_path = "./data/DodgerLoopDay_TEST.ts"
        
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess time series data."""
        train_data, train_labels = load_from_tsfile(self.train_path)
        test_data, test_labels = load_from_tsfile(self.test_path)
        
        self.train_labels, self.test_labels = self._remap_labels(train_labels, test_labels)
        
        if self.data_split == "train":
            self.data, self.labels = train_data, self.train_labels
        else:
            self.data, self.labels = test_data, self.test_labels
            
        self._normalize_and_reshape()

    def _remap_labels(self, train_labels: np.ndarray, test_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remap labels to consecutive integers starting from 0."""
        unique_labels = np.unique(train_labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        train_labels = np.vectorize(label_map.get)(train_labels)
        test_labels = np.vectorize(label_map.get)(test_labels)
        return train_labels, test_labels

    def _normalize_and_reshape(self) -> None:
        """Normalize and reshape time series data."""
        n_samples = self.data.shape[0]
        series_len = self.data.shape[2]
        
        # Flatten, normalize, then reshape back
        self.data = self.data.reshape(-1, series_len)
        self.data = self.scaler.fit_transform(self.data)
        self.data = self.data.reshape(n_samples, series_len).T

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get time series sample by index."""
        timeseries = self.data[:, index]
        label = self.labels[index].astype(int)
        
        # Pad/truncate to seq_len
        timeseries_len = len(timeseries)
        input_mask = np.ones(self.seq_len)
        input_mask[:self.seq_len - timeseries_len] = 0
        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))
        
        return np.expand_dims(timeseries, axis=0), input_mask, label