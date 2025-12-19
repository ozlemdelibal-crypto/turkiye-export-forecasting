"""
Dataset utilities for time series forecasting
"""

import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(data, seq_len, horizon=1):
    """
    Create sliding window sequences for time series forecasting
    
    This function transforms a 1D time series into input-output pairs
    using a sliding window approach.
    
    Args:
        data: 1D numpy array of time series values (normalized)
        seq_len: Length of input sequence (lookback window)
        horizon: Number of future steps to predict
    
    Returns:
        X: Input sequences of shape (num_samples, seq_len)
        y: Target values of shape (num_samples, horizon)
    
    Example:
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        seq_len = 3
        horizon = 2
        
        X[0] = [0, 1, 2]  ->  y[0] = [3, 4]
        X[1] = [1, 2, 3]  ->  y[1] = [4, 5]
        X[2] = [2, 3, 4]  ->  y[2] = [5, 6]
        ...
    """
    X, y = [], []
    
    # Calculate maximum starting index
    # We need seq_len points for input and horizon points for output
    max_start = len(data) - seq_len - horizon + 1
    
    # Check if we have enough data
    if max_start <= 0:
        raise ValueError(
            f"Not enough data! seq_len ({seq_len}) + horizon ({horizon}) "
            f"is larger than data length ({len(data)}). "
            f"Please reduce seq_len or horizon, or use more data."
        )
    
    # Create sequences
    for i in range(max_start):
        # Input sequence: [i, i+1, ..., i+seq_len-1]
        X.append(data[i: i + seq_len])
        
        # Output sequence: [i+seq_len, i+seq_len+1, ..., i+seq_len+horizon-1]
        y.append(data[i + seq_len: i + seq_len + horizon])
    
    # Convert to numpy arrays
    X = np.array(X, dtype="float32")  # shape: (num_samples, seq_len)
    y = np.array(y, dtype="float32")  # shape: (num_samples, horizon)
    
    return X, y


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data
    
    This class wraps numpy arrays into a PyTorch Dataset for use with DataLoader.
    
    Args:
        X: Input sequences (numpy array)
        y: Target values (numpy array)
    """
    def __init__(self, X, y):
        """
        Initialize dataset
        
        Args:
            X: Input sequences of shape (num_samples, seq_len)
            y: Target values of shape (num_samples, horizon)
        """
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
        # Verify shapes match
        assert len(self.X) == len(self.y), "X and y must have same number of samples"
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (input_sequence, target_value)
        """
        return self.X[idx], self.y[idx]


def denormalize(data, mean, std):
    """
    Denormalize data back to original scale
    
    Args:
        data: Normalized data (numpy array or torch tensor)
        mean: Mean value used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Denormalized data in original scale
    """
    return data * std + mean


def normalize(data, mean=None, std=None):
    """
    Normalize data using z-score normalization
    
    Args:
        data: Raw data (numpy array)
        mean: Mean value (if None, compute from data)
        std: Standard deviation (if None, compute from data)
    
    Returns:
        normalized_data: Normalized data
        mean: Mean value used
        std: Standard deviation used
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    
    normalized_data = (data - mean) / (std + 1e-8)
    
    return normalized_data, mean, std


if __name__ == '__main__':
    # Test the functions
    print("Testing dataset utilities...")
    print("="*70)
    
    # Create sample time series
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype='float32')
    
    print("Original data:")
    print(data)
    print()
    
    # Normalize
    data_norm, mean, std = normalize(data)
    print(f"Normalized data (mean={mean:.2f}, std={std:.2f}):")
    print(data_norm)
    print()
    
    # Create sequences
    seq_len = 3
    horizon = 2
    X, y = create_sequences(data_norm, seq_len, horizon)
    
    print(f"Created sequences (seq_len={seq_len}, horizon={horizon}):")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"\nFirst 3 samples:")
    for i in range(3):
        print(f"  X[{i}] = {X[i]} -> y[{i}] = {y[i]}")
    print()
    
    # Create dataset
    dataset = TimeSeriesDataset(X, y)
    print(f"Dataset size: {len(dataset)}")
    
    # Test indexing
    sample_x, sample_y = dataset[0]
    print(f"Sample 0:")
    print(f"  Input shape: {sample_x.shape}")
    print(f"  Target shape: {sample_y.shape}")
    print(f"  Input tensor: {sample_x}")
    print(f"  Target tensor: {sample_y}")
    print()
    
    # Test denormalization
    denorm_data = denormalize(data_norm, mean, std)
    print("Denormalized data (should match original):")
    print(denorm_data)
    print()
    
    # Verify denormalization
    is_close = np.allclose(data, denorm_data, rtol=1e-5)
    print(f"Denormalization correct: {is_close}")
    print()
    
    print("="*70)
    print("All tests passed!")
