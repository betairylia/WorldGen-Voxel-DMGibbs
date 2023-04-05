from .base import *
from .transformer_family import *
from .conv_family import *

models_forwarder = {
    "placeholder": lambda margs : base.PlaceholderMLP(margs.seq_length, margs.num_alphabet, 1024, 2),
    "transformer-basic": lambda margs : transformer_family.TransformerForwarder(
        seq_length=margs.seq_length,
        num_alphabet=margs.num_alphabet,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
    ),
    "conv-minimal": lambda margs : conv_family.UNet1D_Forwarder(
        num_alphabet=margs.num_alphabet,
        num_filters=64,
        num_levels=2,
        layers_per_level=1,
        use_batchnorm=False,
        dropout_prob=0.0
    ),
    "conv-basic": lambda margs : conv_family.UNet1D_Forwarder(
        num_alphabet=margs.num_alphabet,
        num_filters=64,
        num_levels=4,
        layers_per_level=2,
        use_batchnorm=True,
        dropout_prob=0.1
    )
}

models_critic = {
    "placeholder": lambda margs : base.PlaceholderMLP_critic(margs.seq_length, margs.num_alphabet, 1024, 1),
    "transformer-basic": lambda margs : transformer_family.TransformerCritic(
        seq_length=margs.seq_length,
        num_alphabet=margs.num_alphabet,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
    ),
    "conv-minimal": lambda margs : conv_family.ConvCritic(
        seq_length=margs.seq_length,
        num_alphabet=margs.num_alphabet,
        num_filters=64,
        num_levels=4,
        layers_per_level=1,
        use_batchnorm=False,
        dropout_prob=0.0
    ),
    "conv-basic": lambda margs : conv_family.ConvCritic(
        seq_length=margs.seq_length,
        num_alphabet=margs.num_alphabet,
        num_filters=64,
        num_levels=7,
        layers_per_level=2,
        use_batchnorm=True,
        dropout_prob=0.1
    )
}

def GetModel(model_args):
    f = models_forwarder[model_args.modelname](model_args)
    c = models_critic[model_args.modelname](model_args)
    return f, c
