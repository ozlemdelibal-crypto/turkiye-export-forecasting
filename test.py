"""
Testing Script for Turkey Export Time Series Forecasting
This script evaluates trained models on test data
"""

import os
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import LSTMModel, GRUModel, TCNModel
from dataset import TimeSeriesDataset, create_sequences
from utils import set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test time series forecasting models')
    
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
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['lstm', 'gru', 'tcn'],
                       help='Model type')
    parser.add_argument('--model_size', type=str, required=True,
                       choices=['small', 'medium', 'large'],
                       help='Model size')
    
    # Other parameters
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing normalization params and indices')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()

def load_data(data_path, target_country, seq_len, horizon, results_dir):
    """
    Load and preprocess test data
    
    Returns:
        test_loader: DataLoader for test data
        mean_val: Mean for denormalization
        std_val: Std for denormalization
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load normalization parameters
    norm_path = os.path.join(results_dir, 'normalization_params.npz')
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization parameters not found: {norm_path}")
    
    norm_params = np.load(norm_path)
    mean_val = norm_params['mean']
    std_val = norm_params['std']
    
    print(f"Loaded normalization parameters:")
    print(f"  Mean: ${mean_val:,.2f}")
    print(f"  Std: ${std_val:,.2f}")
    
    # Read data
    df_list = pd.read_html(data_path)
    df_raw = df_list[3].copy()
    
    if target_country not in df_raw["Importers"].values:
        raise ValueError(f"Country '{target_country}' not found in data!")
    
    df_country = df_raw[df_raw["Importers"] == target_country].copy()
    date_columns = [col for col in df_country.columns if col != "Importers"]
    
    print(f"\nTarget country: {target_country}")
    print(f"Number of months: {len(date_columns)}")
    
    # Convert to time series and normalize
    values_raw = df_country[date_columns].values.flatten().astype("float32")
    values_normalized = (values_raw - mean_val) / (std_val + 1e-8)
    
    # Create sequences
    X_all, y_all = create_sequences(values_normalized, seq_len, horizon)
    
    # Load split indices
    indices_path = os.path.join(results_dir, 'split_indices.npz')
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"Split indices not found: {indices_path}")
    
    indices = np.load(indices_path)
    test_indices = indices['test']
    
    print(f"\nTest set size: {len(test_indices)} samples")
    
    # Create test dataset
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return test_loader, mean_val, std_val

def load_model(model_path, model_type, model_size, seq_len, horizon, device):
    """
    Load trained model
    
    Returns:
        model: Loaded PyTorch model
    """
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    # Model configurations
    configs = {
        'lstm': {
            'small': {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1},
            'medium': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
            'large': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
        },
        'gru': {
            'small': {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1},
            'medium': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
            'large': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
        },
        'tcn': {
            'small': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dropout': 0.1},
            'medium': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dropout': 0.2},
            'large': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 256, 'dropout': 0.3}
        }
    }
    
    params = configs[model_type][model_size]
    
    # Create model
    if model_type == 'lstm':
        model = LSTMModel(
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            output_size=horizon
        )
    elif model_type == 'gru':
        model = GRUModel(
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            output_size=horizon
        )
    elif model_type == 'tcn':
        model = TCNModel(
            c1=params['c1'],
            c2=params['c2'],
            c3=params['c3'],
            c4=params['c4'],
            dropout=params['dropout'],
            seq_len=seq_len,
            output_size=horizon
        )
    
    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model: {model_type.upper()}_{model_size}")
    print(f"Parameters: {num_params:,}")
    print(f"Loaded from: {model_path}")
    
    return model

def evaluate_model(model, test_loader, mean_val, std_val, horizon, device):
    """
    Evaluate model on test data
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    model.eval()
    all_preds_norm = []
    all_targets_norm = []
    
    # Inference time measurement
    inference_times = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            start = time.time()
            outputs = model(X_batch)
            inference_times.append(time.time() - start)
            
            all_preds_norm.append(outputs.cpu().numpy())
            all_targets_norm.append(y_batch.cpu().numpy())
    
    all_preds_norm = np.vstack(all_preds_norm)
    all_targets_norm = np.vstack(all_targets_norm)
    
    # Normalized metrics
    mse_norm = np.mean((all_preds_norm - all_targets_norm) ** 2)
    mae_norm = np.mean(np.abs(all_preds_norm - all_targets_norm))
    
    # Denormalize
    all_preds_real = all_preds_norm * std_val + mean_val
    all_targets_real = all_targets_norm * std_val + mean_val
    
    # Real-scale metrics
    mse_real = np.mean((all_preds_real - all_targets_real) ** 2)
    mae_real = np.mean(np.abs(all_preds_real - all_targets_real))
    rmse_real = np.sqrt(mse_real)
    
    # MAPE
    mape = np.mean(
        np.abs((all_targets_real - all_preds_real) / (all_targets_real + 1e-8))
    ) * 100
    
    # R² Score
    ss_res = np.sum((all_targets_real - all_preds_real) ** 2)
    ss_tot = np.sum((all_targets_real - all_targets_real.mean()) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Speed metrics
    avg_inference_time = np.mean(inference_times)
    total_inference_time = sum(inference_times)
    
    # Print results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    
    print(f"\n⏱️  Speed Metrics:")
    print(f"   Total inference time: {total_inference_time:.3f}s")
    print(f"   Avg batch time: {avg_inference_time*1000:.2f}ms")
    
    print(f"\n📊 Accuracy Metrics (Real $ Scale):")
    print(f"   RMSE: ${rmse_real:,.2f}")
    print(f"   MAE:  ${mae_real:,.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2_score:.4f}")
    
    print(f"\n📉 Normalized Scale:")
    print(f"   MSE: {mse_norm:.5f}")
    print(f"   MAE: {mae_norm:.5f}")
    
    print(f"\n📈 Sample Predictions (first 5):")
    for i in range(min(5, len(all_preds_real))):
        pred = all_preds_real[i, 0]
        target = all_targets_real[i, 0]
        error = abs(pred - target)
        error_pct = (error / target) * 100
        print(f"   Sample {i+1}: Pred=${pred:,.0f} | True=${target:,.0f} | "
              f"Error=${error:,.0f} ({error_pct:.2f}%)")
    
    print(f"{'='*70}")
    
    results = {
        'rmse': float(rmse_real),
        'mae': float(mae_real),
        'mape': float(mape),
        'r2_score': float(r2_score),
        'mse_norm': float(mse_norm),
        'mae_norm': float(mae_norm),
        'inference_time': float(total_inference_time),
        'avg_batch_time': float(avg_inference_time)
    }
    
    return results, all_preds_real, all_targets_real

def main():
    """Main testing function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("TURKEY EXPORT TIME SERIES FORECASTING - TESTING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model_type.upper()}_{args.model_size}")
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    test_loader, mean_val, std_val = load_data(
        args.data_path,
        args.target_country,
        args.seq_len,
        args.horizon,
        args.results_dir
    )
    
    # Load model
    model = load_model(
        args.model_path,
        args.model_type,
        args.model_size,
        args.seq_len,
        args.horizon,
        device
    )
    
    # Evaluate
    results, predictions, targets = evaluate_model(
        model, test_loader, mean_val, std_val, args.horizon, device
    )
    
    # Save results
    output_dir = os.path.dirname(args.model_path)
    model_name = f"{args.model_type}_{args.model_size}"
    
    # Save metrics
    results_path = os.path.join(output_dir, f'{model_name}_test_results.csv')
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save predictions
    pred_path = os.path.join(output_dir, f'{model_name}_predictions.npz')
    np.savez(pred_path, predictions=predictions, targets=targets)
    print(f"Predictions saved to: {pred_path}")
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETED!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
