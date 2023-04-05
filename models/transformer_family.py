import torch
from torch import nn
import torch.nn.functional as F

import math

from .base import ModelBase

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerForwarder(ModelBase):
    
    def __init__(self, seq_length, num_alphabet, **transformer_kwargs):
        
        super(TransformerForwarder, self).__init__(seq_length=seq_length, num_alphabet=num_alphabet)
        
        # Simply use a learnable matrix to embed the input
        self.embed = nn.Linear(num_alphabet, transformer_kwargs['d_model'], bias=False)
        self.embed_temperature = nn.Linear(1, transformer_kwargs['d_model'], bias=True)

        # Temperature parameter
        self.sample_temperature = nn.Parameter(torch.tensor(1.0))

        # Use no decoder as we have no target sequence.
        # Not sure how will this perform.

        # Dirty, Ughhhhhh
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_kwargs['d_model'],
            nhead=transformer_kwargs['nhead'],
            dim_feedforward=transformer_kwargs['dim_feedforward'] if 'dim_feedforward' in transformer_kwargs else 2048,
            dropout=transformer_kwargs['dropout'] if 'dropout' in transformer_kwargs else 0.1,
            activation=transformer_kwargs['activation'] if 'activation' in transformer_kwargs else 'relu',
            layer_norm_eps=transformer_kwargs['layer_norm_eps'] if 'layer_norm_eps' in transformer_kwargs else 1e-5,
            batch_first=transformer_kwargs['batch_first'] if 'batch_first' in transformer_kwargs else False,
            norm_first=transformer_kwargs['norm_first'] if 'norm_first' in transformer_kwargs else False,
            device=transformer_kwargs['device'] if 'device' in transformer_kwargs else None,
            dtype=transformer_kwargs['dtype'] if 'dtype' in transformer_kwargs else None,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_kwargs['num_layers'] if 'num_layers' in transformer_kwargs else 6,
            norm=transformer_kwargs['norm'] if 'norm' in transformer_kwargs else None,
            mask_check=transformer_kwargs['mask_check'] if 'mask_check' in transformer_kwargs else True,

            # Set to true if you have a variable sequence length
            enable_nested_tensor=transformer_kwargs['enable_nested_tensor'] if 'enable_nested_tensor' in transformer_kwargs else False,
        )

        self.pos_encoder = PositionalEncoding(
            transformer_kwargs['d_model'],
            dropout=transformer_kwargs['dropout'] if 'dropout' in transformer_kwargs else 0.1
        )

    def forward(self, x, t):

        # x: (batch_size, seq_length, num_alphabet)
        # t: (1,)

        bs = x.shape[0]

        x = x.permute(1, 0, 2) # (seq_length, batch_size, num_alphabet)
        x = self.embed(x) # (seq_length, batch_size, d_model)
        t = self.embed_temperature(torch.ones(bs, 1, device = x.device) * t) # (1, d_model)

        # TODO: Should we normalize the embedding?
        x = F.normalize(x, dim=2) # (seq_length, batch_size, d_model)

        # Append t to the beginning of the sequence
        x = torch.cat([t.unsqueeze(0), x], dim=0) # (seq_length + 1, batch_size, d_model)

        # Positional embedding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Remove the first token (temperature)
        x = x[1:, :, :]

        return self.forward_readout(x)

    def forward_readout(self, x):

        # Split the temperature and logits
        x_logits = x[:, :, :]

        # Retrieve the logits by computing the cosine similarity between the output and embeddings
        logits = self.sample_temperature * torch.matmul(x_logits, self.embed.weight.T) # (seq_length, batch_size, num_alphabet)

        # Reshape to (batch_size, seq_length, num_alphabet)
        logits = logits.permute(1, 0, 2)

        return logits, None # output, latent

class TransformerCritic(TransformerForwarder):

    def __init__(self, seq_length, num_alphabet, **transformer_kwargs):
        
        super(TransformerCritic, self).__init__(seq_length=seq_length, num_alphabet=num_alphabet, **transformer_kwargs)
        self.readout = nn.Linear(transformer_kwargs['d_model'], 2)

    def forward_readout(self, x):

        # Readout and Global Average Pooling
        y = self.readout(x)
        y = torch.mean(y, dim=0)

        return y[:, 0], y[:, 1] # score, entropy
