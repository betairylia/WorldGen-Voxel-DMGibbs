import torch
import numpy as np
import torch.nn.functional as F

from models import GetModel
from datagen import DataPool
from WGRutils import WGREnv, WandBScatterMaxlinePlot

import os
import wandb
import argparse
from datetime import datetime
from termcolor import colored

EPS = 1e-8

def temperature_schedule(args, timestep):
    normalized_time = timestep / args.timesteps
    return args.invtemp_min + normalized_time * (args.invtemp_max - args.invtemp_min)

def query_score(env, seq):
    return env.getEnergy(seq, False)

def init_wandb(args):
    run = wandb.init(project="WGR-DMGIBBS", config=args)
    return run

def main(args):
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the data pool
    print(colored("[INFO] Initializing the data pool...", "green"))
    pool = DataPool(args.batch_size, args.seq_length, args.timesteps)
    env = WGREnv(args, args.seq_length)
    # pool.initialize()

    # Initialize the model
    print(colored("[INFO] Initializing the model...", "green"))
    forwarder, critic = GetModel(args)

    print(colored("[MODEL] Forward generator:", "red"))
    print(forwarder)
    print(colored("[MODEL] Score / entropy critic:", "red"))
    print(critic)

    # Move the model to the device
    print(colored("[INFO] Moving the model to the device...", "green"))
    forwarder = forwarder.to(device)
    critic = critic.to(device)

    if args.resume is not None:

        print(colored("[INFO] Resuming from checkpoint %s..." % args.resume, "green"))

        checkpoint = torch.load(args.resume)

        forwarder.load_state_dict(checkpoint["forwarder"])
        critic.load_state_dict(checkpoint["critic"])

        print(colored("[INFO] Resumed from checkpoint %s." % args.resume, "yellow"))
    
    else:

        print(colored("[ERROR] No checkpoint assigned! aborting ...", "red"))
        exit()

    print(colored("[INFO] Finished initialization.", "yellow"))

    forwarder.eval()
    critic.eval()

    ###############################################################################
    # Data preparation
    ###############################################################################

    # Obtain a batch to start
    uniform_batch = pool.get_and_store_uniform_batch(args.batch_size)
    uniform_batch = uniform_batch.to(device)
    uniform_batch = F.one_hot(uniform_batch, num_classes=args.num_alphabet).float()

    previous_batch = uniform_batch

    for it in range(args.timesteps):

        ###############################################################################
        # Feed-forward
        ###############################################################################

        temperature = temperature_schedule(args, it)

        # Obtain forwarder output (probablities)
        generated_batch, _ = forwarder(previous_batch, temperature)

        # Sample a batch of sequences with torch.distribution from forwarder output logits
        sampled_batch_seq = torch.distributions.Categorical(logits=generated_batch).sample()

        # One-hot encode the sampled batch sequence
        sampled_batch = F.one_hot(sampled_batch_seq, num_classes=args.num_alphabet).float()

        sampled_batch_seq = sampled_batch_seq.cpu()

        # Obtain energy from the scoring server
        true_energy = query_score(env, sampled_batch_seq.numpy()).to(device)
        print(
            colored(
                "[EVALUATION] Time: {:3d}, Energy: {:9.6f}"
                .format(it, true_energy.mean()),
            "blue"
        ))

        previous_batch = sampled_batch

    # Visualization thru WGR
    print(sampled_batch_seq)
    env.getEnergy(sampled_batch_seq.numpy(), True)

    # Press any key to exit
    input("Visualization in progress. \nPress any key to exit...")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='WGR-DMGIBBS')

    parser.add_argument('--serverIP', type=str, default='127.0.0.1')
    parser.add_argument('--serverPort', type=int, default=4445)
    parser.add_argument('--env', type=str, default='testCity')

    parser.add_argument('--seq_length', type=int, default=4096)
    parser.add_argument('--timesteps', type=int, default=25)
    parser.add_argument('--delta_time', type=int, default=1)
    parser.add_argument('--num_alphabet', type=int, default=256)

    parser.add_argument('--invtemp_min', type=float, default=0.0)
    parser.add_argument('--invtemp_max', type=float, default=1.0)
    parser.add_argument('--energy_scale', type=float, default=0.001)
    parser.add_argument('--dont_invert_score', action='store_true')

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--modelname', type=str, default='placeholder')

    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # Get a unique name by datetime
    args.savedir = "saved_models/%s/%s[%s]" % (args.env, args.modelname, datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.savepath = os.path.join(args.savedir, "latest.pt")

    main(args)
