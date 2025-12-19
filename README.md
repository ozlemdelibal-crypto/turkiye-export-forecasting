# 🇹🇷 Turkey Export Time Series Forecasting

Deep Learning models (LSTM, GRU, TCN) for forecasting Turkey's monthly export values by country.

## 📊 Project Overview

This project uses time series forecasting to predict Turkey's monthly export values to different countries/regions. The data spans from March 2024 to October 2025 (20 months) and includes exports to 218 different importers.

Three deep learning architectures are implemented and compared:
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)
- **TCN** (Temporal Convolutional Network)

Each model comes in three sizes: **small**, **medium**, and **large**.

## 📁 Project Structure

```
├── train.py              # Training script
├── test.py               # Testing/evaluation script
├── models.py             # Neural network architectures
├── dataset.py            # Dataset and data preprocessing utilities
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── results/             # Directory for trained models and results
│   ├── *_best.pt        # Trained model weights
│   ├── normalization_params.npz
│   ├── split_indices.npz
│   └── training_results.csv
└── data/                # Data directory (not included in repo)
    └── MonthlyExport_Türkiye (6).xls
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (optional, but recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/turkey-export-forecasting.git
cd turkey-export-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data file in the project directory or specify its path.

### Training

Train all models (LSTM, GRU, TCN in all sizes):

```bash
python train.py --data_path "MonthlyExport_Türkiye (6).xls" --target_country "World"
```

Train a specific model:

```bash
python train.py --model_type lstm --model_size medium --num_epochs 100
```

### Testing

Evaluate a trained model:

```bash
python test.py --model_path results/GRU_medium_best.pt --model_type gru --model_size medium
```

## 📊 Model Architectures

### LSTM (Long Short-Term Memory)

LSTM is designed to handle long-term dependencies in sequential data using memory cells and three gates (forget, input, output).

- **Small**: 32 hidden units, 1 layer (~4.5K parameters)
- **Medium**: 64 hidden units, 2 layers (~50K parameters)
- **Large**: 128 hidden units, 2 layers (~199K parameters)

### GRU (Gated Recurrent Unit)

GRU simplifies LSTM by using only two gates (update and reset), resulting in fewer parameters while maintaining similar performance.

- **Small**: 32 hidden units, 1 layer (~3.4K parameters)
- **Medium**: 64 hidden units, 2 layers (~38K parameters)
- **Large**: 128 hidden units, 2 layers (~150K parameters)

### TCN (Temporal Convolutional Network)

TCN uses dilated causal convolutions to capture long-range dependencies without recurrence.

- **Small**: 8-16-32-64 channels (~8.5K parameters)
- **Medium**: 16-32-64-128 channels (~33K parameters)
- **Large**: 32-64-128-256 channels (~131K parameters)

## 🎯 Training Parameters

Key hyperparameters (can be modified via command-line arguments):

- **Sequence Length** (`--seq_len`): 6 months (default)
- **Horizon** (`--horizon`): 1 month ahead prediction
- **Batch Size** (`--batch_size`): 8
- **Learning Rate** (`--learning_rate`): 0.001
- **Max Epochs** (`--num_epochs`): 100
- **Early Stopping Patience** (`--patience`): 15 epochs

## 📈 Results

The models are evaluated using the following metrics:

- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **MAPE** (Mean Absolute Percentage Error): Lower is better
- **R²** (Coefficient of Determination): Higher is better (max 1.0)

### Best Model Performance (Example)

Based on initial experiments on World exports:

| Model | RMSE ($1000) | MAPE (%) | R² | Parameters |
|-------|--------------|----------|-----|------------|
| GRU_medium | 952,589 | 3.35% | -0.067 | 37,889 |
| TCN_medium | 1,012,919 | 3.57% | -0.207 | 33,153 |
| LSTM_small | 1,036,235 | 3.56% | -0.263 | 4,513 |

*Note: Negative R² indicates the model performs worse than a simple mean baseline on this small test set.*

## 🔧 Command-Line Arguments

### train.py

```
--data_path          Path to data file (default: MonthlyExport_Türkiye (6).xls)
--target_country     Target country (default: World)
--seq_len            Sequence length (default: 6)
--horizon            Prediction horizon (default: 1)
--batch_size         Batch size (default: 8)
--num_epochs         Maximum epochs (default: 100)
--learning_rate      Learning rate (default: 0.001)
--patience           Early stopping patience (default: 15)
--model_type         Model type: lstm, gru, tcn, all (default: all)
--model_size         Model size: small, medium, large, all (default: all)
--save_dir           Results directory (default: results)
--device             Device: auto, cuda, cpu (default: auto)
--seed               Random seed (default: 42)
```

### test.py

```
--model_path         Path to trained model .pt file (required)
--model_type         Model type: lstm, gru, tcn (required)
--model_size         Model size: small, medium, large (required)
--data_path          Path to data file
--target_country     Target country
--seq_len            Sequence length (must match training)
--horizon            Prediction horizon (must match training)
--results_dir        Directory with normalization params (default: results)
--device             Device: auto, cuda, cpu (default: auto)
```

## 🧪 Example Usage

### 1. Train only GRU models

```bash
python train.py --model_type gru --model_size all
```

### 2. Train a small LSTM model with custom parameters

```bash
python train.py --model_type lstm --model_size small --num_epochs 50 --batch_size 16
```

### 3. Test a trained model

```bash
python test.py --model_path results/GRU_medium_best.pt --model_type gru --model_size medium
```

### 4. Train for a different country

```bash
python train.py --target_country "Germany" --seq_len 8
```

## 📝 Data Format

The input data should be an Excel file (.xls) from Trade Map containing:

- **Column 1**: Importers (country names)
- **Columns 2-N**: Monthly export values in USD thousands
  - Format: "Exported value in YYYY-MM"

Example:
```
Importers | Exported value in 2024-M03 | Exported value in 2024-M04 | ...
World     | 22648720                    | 19292590                    | ...
Germany   | 1956644                     | 1652806                     | ...
```

## 🔬 Model Details

### Normalization

Data is normalized using z-score normalization:
```
normalized = (value - mean) / std
```

Normalization parameters are saved in `results/normalization_params.npz`.

### Data Splitting

- **Training**: 70% of sequences
- **Validation**: 15% of sequences
- **Test**: 15% of sequences

Split indices are saved in `results/split_indices.npz` for reproducibility.

### Loss Function

Mean Squared Error (MSE) is used as the loss function:
```
MSE = (1/n) * Σ(y_pred - y_true)²
```

### Optimization

- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Reduces LR by factor 0.5 when validation loss plateaus (patience=5)
- **Early Stopping**: Stops training if no improvement for 15 epochs

## 🐛 Troubleshooting

### "Not enough data" error

Reduce `--seq_len` or `--horizon` values. With only 20 months of data, very large sequence lengths will leave too few samples.

### Out of memory (GPU)

- Reduce `--batch_size`
- Use a smaller model size (`--model_size small`)
- Use CPU instead (`--device cpu`)

### Poor model performance

- Increase `--num_epochs`
- Try different model types or sizes
- Adjust `--seq_len` to capture more temporal patterns
- Collect more data if possible

## 📚 References

- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU Paper](https://arxiv.org/abs/1406.1078)
- [TCN Paper](https://arxiv.org/abs/1803.01271)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

[Your Name]  
[Your Email]  
[Your GitHub]

## 🙏 Acknowledgments

- Data source: Trade Map (International Trade Centre)
- Built with PyTorch and Python

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com]

---

**Note**: This project is for educational purposes. Model performance depends heavily on data quality and quantity. With only 20 months of data, predictions should be interpreted cautiously.
