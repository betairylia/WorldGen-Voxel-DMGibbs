import torch
from torch import nn
import torch.nn.functional as F

import math

from .base import ModelBase

import torch
import torch.nn as nn

'''
Basic convolutional block
'''
def conv_block(in_channels, out_channels, layers_per_level=2, kernel_size=3, padding=1, use_batchnorm=True, dropout_prob=0.0):

    layers = []

    # Simple conv-BN-ReLU-Dropout block

    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))
    
    return nn.Sequential(*layers)

'''
Residual block with single layer
'''
class SingleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(SingleResidualBlock, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels, *args, **kwargs)
        self.use_residual = in_channels == out_channels

    def forward(self, x):
        if self.use_residual:
            return x + self.conv_block(x)
        else:
            return self.conv_block(x)

'''
Full residual block with multiple single residual blocks
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers_per_level=2, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.res_blocks = nn.Sequential(
            *[SingleResidualBlock(in_channels if i == 0 else out_channels, out_channels, *args, **kwargs) for i in range(layers_per_level)]
        )

    def forward(self, x):
        return self.res_blocks(x)

'''
Encoder
'''
class Encoder1D(nn.Module):
    def __init__(self, num_alphabet, num_filters=64, num_levels=3, layers_per_level=2, use_batchnorm=True, dropout_prob=0.0):
        super(Encoder1D, self).__init__()

        self.num_levels = num_levels

        # Encoding layers
        self.enc = nn.ModuleList()
        for i in range(num_levels):
            in_channels = num_alphabet if i == 0 else num_filters * 2 ** i
            out_channels = num_filters * 2 ** (i + 1)
            self.enc.append(ResidualBlock(
                in_channels, out_channels, 
                layers_per_level, 
                kernel_size = 5 if i == 0 else 3,
                padding = 2 if i == 0 else 1,
                use_batchnorm=use_batchnorm, 
                dropout_prob=dropout_prob
            ))

        self.final_out_channels = num_filters * 2 ** num_levels

        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        encodings = []

        # Encoding
        for i in range(self.num_levels):
            x = self.enc[i](x)
            encodings.append(x)
            x = self.pool(x)

        return encodings

class UNet1D_Forwarder(nn.Module):
    def __init__(self, num_alphabet, num_filters=64, num_levels=3, layers_per_level=2, use_batchnorm=True, dropout_prob=0.0):
        super(UNet1D_Forwarder, self).__init__()

        self.encoder = Encoder1D(num_alphabet + 1, num_filters, num_levels, layers_per_level, use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)

        # Decoding layers
        self.dec = nn.ModuleList()
        for i in range(num_levels - 1, 0, -1):
            in_channels = num_filters * 2 ** (i + 1) + num_filters * (2 ** i)
            out_channels = num_filters * 2 ** i
            self.dec.append(ResidualBlock(in_channels, out_channels, layers_per_level, use_batchnorm=use_batchnorm, dropout_prob=dropout_prob))

        self.final = nn.Conv1d(num_filters * 2, num_alphabet, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        
        # Shape of x: (batch_size, seq_len, num_alphabet)
        # Shape of t: float

        # Append t as a channel to each component of x
        t = torch.ones(x.shape[0], x.shape[1], 1, device = x.device) * t
        x = torch.cat((x, t), dim=2)

        x = torch.transpose(x, 1, 2)  # (batch_size, num_alphabet + 1, seq_len)

        # Feed-Forward
        # Encoding
        encodings = self.encoder(x)

        # Decoding
        x = encodings[-1]
        for i in range(self.encoder.num_levels - 1):
            x = self.upsample(x)
            x = torch.cat((x, encodings[-(i + 2)]), dim=1)
            x = self.dec[i](x)

        out = self.final(x)
        out = torch.transpose(out, 1, 2)  # (batch_size, seq_len, num_alphabet)

        return out, None

class ConvCritic(nn.Module):

    def __init__(self, seq_length, num_alphabet, num_filters=64, num_levels=3, layers_per_level=2, use_batchnorm=True, dropout_prob=0.0):

        super(ConvCritic, self).__init__()

        self.encoder = Encoder1D(num_alphabet + 1, num_filters, num_levels, layers_per_level, use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.readout = nn.Conv1d(self.encoder.final_out_channels, 2, kernel_size=1)  # 2 output channels for score and entropy

    def forward(self, x, t):

        # Shape of x: (batch_size, seq_len, num_alphabet)
        # Shape of t: float

        # Append t as a channel to each component of x
        t = torch.ones(x.shape[0], x.shape[1], 1, device = x.device) * t
        x = torch.cat((x, t), dim=2)

        x = torch.transpose(x, 1, 2)  # (batch_size, num_alphabet + 1, seq_len)

        x = self.encoder(x)[-1]
        return self.forward_readout(x)

    def forward_readout(self, x):

        # Readout and Global Average Pooling
        y = self.readout(x)
        y = torch.mean(y, dim=2)  # Apply mean along the sequence dimension

        return y[:, 0], y[:, 1]  # score, entropy
