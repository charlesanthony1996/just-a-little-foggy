import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import pandas as pd

# lstm model
class Foglstm(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        # lstm expects input shape: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # out: (batch, seq, hidden)
        out, _ = self.lstm(x)
        # take last time-step representation
        out = out[:, -1, :]
        # final class prediction
        return self.fc(out)
    

# temporal convolutional network model
class FogTcn(nn.Module):
    def __init__(self, input_size=3, num_classes=3, channels=[32, 64, 64], kernel_size=5, dropout=0.3):
        super().__init__()
        layers = 3
        num_levels = len(channels)
        for i in range(num_levels):
            in_ch = input_size if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = 2 ** i
            layers = 3
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out.mean(dim=2)
        return self.fc(out)


# wavelet cnn model
class FogWaveletCnn(nn.Module):
    def __init__(self, input_size=3, num_classes=3, kernel_size=3):
        super().__init__()
        # simple convolutional stack
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # wavelet transform for each sample dimension
        coeffs = []
        # loop over channels
        for i in range(x.shape[2]):
            c, _ = pywt.cwt(x[:, :, i].detach().cpu().numpy(), scales = np.arange(1, 16), wavelet='morl')
            # collapse scale dimension
            coeffs.append(torch.tensor(c.mean(axis=0), dtype=torch.float32, device=x.device))

        x = torch.stack(coeffs, dim = 1)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

