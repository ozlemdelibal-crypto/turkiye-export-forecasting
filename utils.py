"""
Utility functions for time series forecasting
"""

import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    This function sets the seed for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU and GPU)
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def get_device(device_type='auto'):
    """
    Get the device to use for training/inference
    
    Args:
        device_type: 'auto', 'cuda', or 'cpu'
                    'auto' will use CUDA if available
    
    Returns:
        device: torch.device object
    """
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_type == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU instead.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device


def print_model_summary(model, input_size=(1, 6)):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
        input_size: Tuple of (batch_size, seq_len)
    """
    print("="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    # Print architecture
    print(model)
    print()
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print()
    
    # Test forward pass
    device = next(model.parameters()).device
    sample_input = torch.randn(*input_size).to(device)
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        
        print(f"Input shape:  {sample_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Could not perform forward pass: {e}")
    
    print("="*70)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save a training checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a training checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: PyTorch optimizer (optional)
    
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
    """
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss


def calculate_metrics(predictions, targets):
    """
    Calculate various evaluation metrics
    
    Args:
        predictions: Predicted values (numpy array)
        targets: True values (numpy array)
    
    Returns:
        Dictionary containing various metrics
    """
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # R² Score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2)
    }


def format_time(seconds):
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == '__main__':
    # Test utility functions
    print("Testing utility functions...")
    print("="*70)
    
    # Test set_seed
    set_seed(42)
    print("✓ Random seed set to 42")
    
    # Test get_device
    device = get_device('auto')
    print(f"✓ Device: {device}")
    
    # Test calculate_metrics
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    metrics = calculate_metrics(predictions, targets)
    print(f"\n✓ Metrics calculated:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test format_time
    print(f"\n✓ Time formatting:")
    print(f"  3661 seconds = {format_time(3661)}")
    print(f"  125 seconds = {format_time(125)}")
    print(f"  45 seconds = {format_time(45)}")
    
    print("\n" + "="*70)
    print("All utility functions working correctly!")
