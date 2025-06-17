import numpy as np
from sklearn.preprocessing import StandardScaler
from data import load_from_tsfile

class ClassificationDataset:
    def __init__(self, data_split="train", train_file="DodgerLoopDay_TRAIN.ts", test_file="DodgerLoopDay_TEST.ts"):
        """
        Initialize the classification dataset
        
        Args:
            data_split (str): Split of the dataset, 'train' or 'test'
            train_file (str): Path to training data file
            test_file (str): Path to test data file
        """
        self.seq_len = 512
        self.train_file_path_and_name = train_file
        self.test_file_path_and_name = test_file
        self.data_split = data_split

        # Read and process data
        self._read_data()

    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        """Transform labels to consecutive integers starting from 0"""
        labels = np.unique(train_labels)
        transform = {l: i for i, l in enumerate(labels)}
        
        train_labels = np.vectorize(transform.get)(train_labels)
        print("Unique train labels:", np.unique(train_labels))
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def __len__(self):
        """Return number of time series in the dataset"""
        return self.num_timeseries

    def _read_data(self):
        """Load and preprocess the data"""
        self.scaler = StandardScaler()

        # Load data from files
        self.train_data, self.train_labels = load_from_tsfile(self.train_file_path_and_name)
        self.test_data, self.test_labels = load_from_tsfile(self.test_file_path_and_name)

        # Transform labels
        self.train_labels, self.test_labels = self._transform_labels(
            self.train_labels, self.test_labels
        )

        # Select appropriate split
        if self.data_split == "train":
            self.data = self.train_data
            self.labels = self.train_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        # Reshape and normalize data
        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]

        self.data = self.data.reshape(-1, self.len_timeseries)
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.data = self.data.reshape(self.num_timeseries, self.len_timeseries)
        self.data = self.data.T

    def __getitem__(self, index):
        """Get a single time series with its label and mask"""
        assert index < self.__len__()

        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = self.labels[index,].astype(int)
        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))

        return np.expand_dims(timeseries, axis=0), input_mask, labels