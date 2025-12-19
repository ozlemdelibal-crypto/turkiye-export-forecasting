"""
Training Script for Turkey Export Time Series Forecasting
This script trains LSTM, GRU, and TCN models to predict monthly export values
"""

import os
import time
import random
import gc
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import LSTMModel, GRUModel, TCNModel
from dataset import TimeSeriesDataset, create_sequences
from utils import set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train time series forecasting models')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                       default='MonthlyExport_Türkiye (6).xls',
                       help='Path to data file')
    parser.add_argument('--target_country', type=str, default='World',
                       help='Target country for prediction')
    parser.add_argument('--seq_len', type=int, default=6,
                       help='Sequence length (lookback window)')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['lstm', 'gru', 'tcn', 'all'],
                       help='Model type to train')
    parser.add_argument('--model_size', type=str, default='all',
                       choices=['small', 'medium', 'large', 'all'],
                       help='Model size variant')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    return parser.parse_args()

def load_and_preprocess_data(data_path, target_country):
    """
    Load and preprocess export data
    
    Args:
        data_path: Path to Excel file
        target_country: Target country name
    
    Returns:
        values_normalized: Normalized time series data
        mean_val: Mean value for denormalization
        std_val: Standard deviation for denormalization
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Read Excel file (HTML format)
    df_list = pd.read_html(data_path)
    df_raw = df_list[3].copy()  # 4th table contains the data
    
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Available countries: {len(df_raw)}")
    
    # Check if target country exists
    if target_country not in df_raw["Importers"].values:
        raise ValueError(f"Country '{target_country}' not found in data!")
    
    # Extract data for target country
    df_country = df_raw[df_raw["Importers"] == target_country].copy()
    date_columns = [col for col in df_country.columns if col != "Importers"]
    
    print(f"\nTarget country: {target_country}")
    print(f"Number of months: {len(date_columns)}")
    print(f"Date range: {date_columns[0]} → {date_columns[-1]}")
    
    # Convert to time series
    values_raw = df_country[date_columns].values.flatten().astype("float32")
    
    print(f"\nTime series length: {len(values_raw)} months")
    print(f"Min: ${values_raw.min():,.0f} | Max: ${values_raw.max():,.0f}")
    print(f"Mean: ${values_raw.mean():,.0f}")
    
    # Normalize data
    mean_val = values_raw.mean()
    std_val = values_raw.std()
    values_normalized = (values_raw - mean_val) / (std_val + 1e-8)
    
    print(f"\nNormalization:")
    print(f"Mean: ${mean_val:,.2f} | Std: ${std_val:,.2f}")
    
    return values_normalized, mean_val, std_val

def split_data(X_all, y_all, train_ratio=0.70, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Args:
        X_all: Input sequences
        y_all: Target values
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    
    Returns:
        Dictionary containing train, val, test splits
    """
    N = len(X_all)
    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)
    
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, N)
    
    print(f"\nData split:")
    print(f"Train: {len(train_indices)} samples ({train_ratio*100:.0f}%)")
    print(f"Val  : {len(val_indices)} samples ({val_ratio*100:.0f}%)")
    print(f"Test : {len(test_indices)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return {
        'train': (X_all[train_indices], y_all[train_indices]),
        'val': (X_all[val_indices], y_all[val_indices]),
        'test': (X_all[test_indices], y_all[test_indices]),
        'indices': {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def train_model(model, train_loader, val_loader, criterion, 
                num_epochs, learning_rate, patience, device, save_path):
    """
    Train a model with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device to train on
        save_path: Path to save best model
    
    Returns:
        Dictionary containing training history
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    train_start = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            # Save best model
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping! Best epoch: {best_epoch}")
            break
    
    train_time = time.time() - train_start
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_time': train_time
    }

def get_model_configs(model_type, model_size, seq_len, horizon):
    """
    Get model configurations based on type and size
    
    Args:
        model_type: 'lstm', 'gru', 'tcn', or 'all'
        model_size: 'small', 'medium', 'large', or 'all'
        seq_len: Sequence length (for TCN)
        horizon: Prediction horizon
    
    Returns:
        List of model configurations
    """
    configs = []
    
    # LSTM configurations
    if model_type in ['lstm', 'all']:
        lstm_configs = [
            {"family": "LSTM", "variant": "small", "params": {"hidden_size": 32, "num_layers": 1, "dropout": 0.1}},
            {"family": "LSTM", "variant": "medium", "params": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}},
            {"family": "LSTM", "variant": "large", "params": {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}},
        ]
        if model_size == 'all':
            configs.extend(lstm_configs)
        else:
            configs.extend([c for c in lstm_configs if c['variant'] == model_size])
    
    # GRU configurations
    if model_type in ['gru', 'all']:
        gru_configs = [
            {"family": "GRU", "variant": "small", "params": {"hidden_size": 32, "num_layers": 1, "dropout": 0.1}},
            {"family": "GRU", "variant": "medium", "params": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}},
            {"family": "GRU", "variant": "large", "params": {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}},
        ]
        if model_size == 'all':
            configs.extend(gru_configs)
        else:
            configs.extend([c for c in gru_configs if c['variant'] == model_size])
    
    # TCN configurations
    if model_type in ['tcn', 'all']:
        tcn_configs = [
            {"family": "TCN", "variant": "small", "params": {"c1": 8, "c2": 16, "c3": 32, "c4": 64, "dropout": 0.1, "seq_len": seq_len}},
            {"family": "TCN", "variant": "medium", "params": {"c1": 16, "c2": 32, "c3": 64, "c4": 128, "dropout": 0.2, "seq_len": seq_len}},
            {"family": "TCN", "variant": "large", "params": {"c1": 32, "c2": 64, "c3": 128, "c4": 256, "dropout": 0.3, "seq_len": seq_len}},
        ]
        if model_size == 'all':
            configs.extend(tcn_configs)
        else:
            configs.extend([c for c in tcn_configs if c['variant'] == model_size])
    
    return configs

def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"TURKEY EXPORT TIME SERIES FORECASTING - TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Target country: {args.target_country}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Horizon: {args.horizon}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and preprocess data
    values_normalized, mean_val, std_val = load_and_preprocess_data(
        args.data_path, args.target_country
    )
    
    # Save normalization parameters
    norm_params = {
        'mean': float(mean_val),
        'std': float(std_val)
    }
    np.savez(
        os.path.join(args.save_dir, 'normalization_params.npz'),
        **norm_params
    )
    print(f"\nNormalization parameters saved to: {args.save_dir}/normalization_params.npz")
    
    # Create sequences
    print(f"\n{'='*70}")
    print("CREATING SEQUENCES")
    print(f"{'='*70}")
    X_all, y_all = create_sequences(values_normalized, args.seq_len, args.horizon)
    print(f"Total samples: {len(X_all)}")
    print(f"X shape: {X_all.shape} | y shape: {y_all.shape}")
    
    # Split data
    data_splits = split_data(X_all, y_all)
    
    # Save indices
    indices_path = os.path.join(args.save_dir, 'split_indices.npz')
    np.savez(indices_path, **data_splits['indices'])
    print(f"Split indices saved to: {indices_path}")
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(*data_splits['train'])
    val_dataset = TimeSeriesDataset(*data_splits['val'])
    test_dataset = TimeSeriesDataset(*data_splits['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nDataLoaders ready:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Get model configurations
    model_configs = get_model_configs(
        args.model_type, args.model_size, args.seq_len, args.horizon
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING {len(model_configs)} MODEL(S)")
    print(f"{'='*70}")
    
    results = []
    
    # Train each model
    for cfg in model_configs:
        family = cfg["family"]
        variant = cfg["variant"]
        params = cfg["params"]
        
        model_name = f"{family}_{variant}"
        print(f"\n{'#'*70}")
        print(f"MODEL: {model_name}")
        print(f"{'#'*70}")
        
        # Create model
        if family == "LSTM":
            model = LSTMModel(
                input_size=1,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                output_size=args.horizon
            ).to(device)
        elif family == "GRU":
            model = GRUModel(
                input_size=1,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                output_size=args.horizon
            ).to(device)
        elif family == "TCN":
            model = TCNModel(
                c1=params["c1"],
                c2=params["c2"],
                c3=params["c3"],
                c4=params["c4"],
                dropout=params["dropout"],
                seq_len=params["seq_len"],
                output_size=args.horizon
            ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        
        # Train model
        save_path = os.path.join(args.save_dir, f"{model_name}_best.pt")
        train_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            device=device,
            save_path=save_path
        )
        
        print(f"\nTraining completed!")
        print(f"Best val loss: {train_results['best_val_loss']:.5f}")
        print(f"Best epoch: {train_results['best_epoch']}")
        print(f"Training time: {train_results['train_time']:.2f}s")
        print(f"Model saved: {save_path}")
        
        # Save training results
        result = {
            'model_name': model_name,
            'family': family,
            'variant': variant,
            'num_params': num_params,
            'best_val_loss': train_results['best_val_loss'],
            'best_epoch': train_results['best_epoch'],
            'train_time': train_results['train_time']
        }
        results.append(result)
        
        # Clean up
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.save_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Total models trained: {len(results)}")
    print(f"Results saved to: {results_path}")
    print(f"\nBest model (by validation loss):")
    best_model = results_df.loc[results_df['best_val_loss'].idxmin()]
    print(f"  {best_model['model_name']}: {best_model['best_val_loss']:.5f}")

if __name__ == '__main__':
    main()
