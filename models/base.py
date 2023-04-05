import torch
from torch import nn
import torch.nn.functional as F

class ModelBase(nn.Module):
    
    def __init__(self, seq_length, num_alphabet, **kwargs):
    
        super(ModelBase, self).__init__()

        self.seq_length = seq_length
        self.num_alphabet = num_alphabet
    
    def forward(self, x):
        raise NotImplementedError

class PlaceholderMLP(ModelBase):
    
    def __init__(self, seq_length, num_alphabet, hidden_size, hidden_layers):

        super(PlaceholderMLP, self).__init__(seq_length=seq_length, num_alphabet=num_alphabet)

        self.input_size = 1 + seq_length * num_alphabet
        self.output_size = seq_length * num_alphabet
        self.hidden_size = hidden_size
        self.latent_dim = hidden_size
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))

        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x, t):

        x = x.flatten(1)
        x = torch.cat((torch.ones(x.shape[0], 1, device = x.device) * t, x), dim=1)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z = x

        y = self.layers[-1](x)
        y = y.reshape(-1, self.seq_length, self.num_alphabet)

        return y, z # output, latent

class PlaceholderMLP_critic(ModelBase):

    def __init__(self, seq_length, num_alphabet, head_hidden_size, head_hidden_layers):

        super(PlaceholderMLP_critic, self).__init__(seq_length=seq_length, num_alphabet=num_alphabet)

        self.input_size = 1 + seq_length * num_alphabet
        self.output_size = 2
        self.head_hidden_size = head_hidden_size
        self.head_hidden_layers = head_hidden_layers
        self.head = nn.ModuleList()
        self.head.append(nn.Linear(self.input_size, self.head_hidden_size))

        for _ in range(self.head_hidden_layers - 1):
            self.head.append(nn.Linear(self.head_hidden_size, self.head_hidden_size))
        self.head.append(nn.Linear(self.head_hidden_size, self.output_size))

    def forward(self, x, t):
        
        x = x.flatten(1)
        x = torch.cat((torch.ones(x.shape[0], 1, device = x.device) * t, x), dim=1)

        for layer in self.head[:-1]:
            x = F.relu(layer(x))
        y = self.head[-1](x)

        return y[:, 0], y[:, 1] # score, entropy
