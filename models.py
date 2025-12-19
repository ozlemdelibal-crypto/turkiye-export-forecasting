"""
Neural Network Models for Time Series Forecasting
Contains LSTM, GRU, and TCN architectures
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM-based time series forecasting model
    
    Args:
        input_size: Number of input features per timestep (default: 1 for univariate)
        hidden_size: Number of LSTM hidden units
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability between LSTM layers
        output_size: Number of output values (horizon)
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # dropout only if num_layers > 1
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len)
        
        Returns:
            output: Predictions of shape (batch, output_size)
        """
        # Reshape input: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # LSTM forward pass
        # out: (batch, seq_len, hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        out, (hidden, cell) = self.lstm(x)
        
        # Take the output of the last timestep
        # out[:, -1, :]: (batch, hidden_size)
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        # (batch, hidden_size) -> (batch, output_size)
        out = self.fc(out)
        
        return out


class GRUModel(nn.Module):
    """
    GRU-based time series forecasting model
    
    GRU (Gated Recurrent Unit) is similar to LSTM but with fewer parameters.
    It uses update and reset gates instead of forget, input, and output gates.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of GRU hidden units
        num_layers: Number of stacked GRU layers
        dropout: Dropout probability between GRU layers
        output_size: Number of output values (horizon)
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len)
        
        Returns:
            output: Predictions of shape (batch, output_size)
        """
        # Reshape: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # GRU forward pass
        out, hidden = self.gru(x)
        
        # Take last timestep output
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for time series forecasting
    
    TCN uses dilated causal convolutions to capture long-range dependencies.
    Each layer has exponentially increasing dilation rates.
    
    Args:
        c1, c2, c3, c4: Number of channels in each convolutional layer
        dropout: Dropout probability
        seq_len: Input sequence length (not directly used but kept for compatibility)
        output_size: Number of output values (horizon)
    """
    def __init__(self, c1=16, c2=32, c3=64, c4=128, dropout=0.2, seq_len=96, output_size=1):
        super(TCNModel, self).__init__()
        
        # First convolutional block (dilation=1)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=c1,
            kernel_size=3,
            padding=1,      # padding = dilation * (kernel_size - 1) / 2
            dilation=1
        )
        self.bn1 = nn.BatchNorm1d(c1)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional block (dilation=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Third convolutional block (dilation=4)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(c3)
        self.dropout3 = nn.Dropout(dropout)
        
        # Fourth convolutional block (dilation=8)
        self.conv4 = nn.Conv1d(c3, c4, kernel_size=3, padding=8, dilation=8)
        self.bn4 = nn.BatchNorm1d(c4)
        self.dropout4 = nn.Dropout(dropout)
        
        # Global average pooling
        # Reduces sequence dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected output layer
        self.fc = nn.Linear(c4, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len)
        
        Returns:
            output: Predictions of shape (batch, output_size)
        """
        # Reshape: (batch, seq_len) -> (batch, 1, seq_len)
        # Conv1d expects (batch, channels, length)
        x = x.unsqueeze(1)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        
        # Global average pooling
        # (batch, c4, seq_len) -> (batch, c4, 1)
        x = self.global_pool(x)
        
        # Remove the last dimension
        # (batch, c4, 1) -> (batch, c4)
        x = x.squeeze(-1)
        
        # Fully connected layer
        # (batch, c4) -> (batch, output_size)
        x = self.fc(x)
        
        return x


# Model parameter counts for reference
def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models with sample input
    batch_size = 4
    seq_len = 6
    horizon = 1
    
    # Create sample input
    x = torch.randn(batch_size, seq_len)
    
    print("Testing models...")
    print("="*70)
    
    # Test LSTM
    lstm_small = LSTMModel(hidden_size=32, num_layers=1, dropout=0.1, output_size=horizon)
    lstm_output = lstm_small(x)
    print(f"LSTM small:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {lstm_output.shape}")
    print(f"  Parameters: {count_parameters(lstm_small):,}")
    print()
    
    # Test GRU
    gru_medium = GRUModel(hidden_size=64, num_layers=2, dropout=0.2, output_size=horizon)
    gru_output = gru_medium(x)
    print(f"GRU medium:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {gru_output.shape}")
    print(f"  Parameters: {count_parameters(gru_medium):,}")
    print()
    
    # Test TCN
    tcn_large = TCNModel(c1=32, c2=64, c3=128, c4=256, dropout=0.3, seq_len=seq_len, output_size=horizon)
    tcn_output = tcn_large(x)
    print(f"TCN large:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {tcn_output.shape}")
    print(f"  Parameters: {count_parameters(tcn_large):,}")
    print()
    
    print("="*70)
    print("All models passed the test!")
