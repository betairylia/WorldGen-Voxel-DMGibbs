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
    run = wandb.init(project="WGR-DMGIBBS", config=args, name=args.run_name)
    return run

def main(args):
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the data pool
    print(colored("[INFO] Initializing the data pool...", "green"))
    pool = DataPool(args.pool_size, args.seq_length, args.timesteps)
    env = WGREnv(args, args.seq_length)
    # pool.initialize()

    # Initialize the model
    print(colored("[INFO] Initializing the model...", "green"))
    forwarder, critic = GetModel(args)

    print(colored("[MODEL] Forward generator:", "red"))
    print(forwarder)
    print(colored("[MODEL] Score / entropy critic:", "red"))
    print(critic)

    # Initialize the optimizer
    print(colored("[INFO] Initializing the optimizer...", "green"))
    optimizer_f = torch.optim.AdamW(forwarder.parameters(), lr=args.base_lr * args.batch_size / 64, weight_decay=args.weight_decay)
    optimizer_c = torch.optim.AdamW(critic.parameters(), lr=args.base_lr * args.batch_size / 64, weight_decay=args.weight_decay)

    # Move the model to the device
    print(colored("[INFO] Moving the model to the device...", "green"))
    forwarder = forwarder.to(device)
    critic = critic.to(device)

    # Initialize the wandb
    print(colored("[INFO] Initializing the Weights and Biases (wandb)...", "green"))
    wandb_run = init_wandb(args)

    # Initialize the visualization
    print(colored("[INFO] Initializing the visualization...", "green"))
    visualization = WandBScatterMaxlinePlot(args, wandb_run)

    # Enable gradient and parameter logging for the model
    print(colored("[INFO] Enabling gradient and parameter logging...", "green"))
    wandb.watch(forwarder)
    wandb.watch(critic)
    trueenergy_log = [None for _ in range(args.timesteps + args.delta_time)]

    # Create args.savedir if it doesn't exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    print(colored("[INFO] Finished initialization.", "yellow"))

    it = 0
    if args.resume is not None:

        print(colored("[INFO] Resuming from checkpoint...", "green"))

        checkpoint = torch.load(args.resume)

        forwarder.load_state_dict(checkpoint["forwarder"])
        critic.load_state_dict(checkpoint["critic"])
        optimizer_f.load_state_dict(checkpoint["optimizer_f"])
        optimizer_c.load_state_dict(checkpoint["optimizer_c"])

        pool = checkpoint["pool"]
        trueenergy_log = checkpoint["trueenergy_log"]
        it = checkpoint["iteration"]

        print(colored("[INFO] Resumed from checkpoint.", "yellow"))

    forwarder.train()
    critic.train()

    # Training loop
    # TODO: Refactor

    while it < args.n_iters:

        ###############################################################################
        # Data preparation
        ###############################################################################

        # Obtain a batch from the pool
        uniform_batch = pool.get_and_store_uniform_batch(args.batch_size)
        previous_batch, timestep = pool.get_batch(args.batch_size, (it % args.timesteps if it < args.timesteps else -1))

        # Move the batches to the device
        previous_batch = previous_batch.to(device)
        uniform_batch = uniform_batch.to(device)

        # One-hot the batches
        previous_batch = F.one_hot(previous_batch, num_classes=args.num_alphabet).float()
        uniform_batch = F.one_hot(uniform_batch, num_classes=args.num_alphabet).float()

        temperature = temperature_schedule(args, timestep)

        ###############################################################################
        # Loss for forwarder
        ###############################################################################

        optimizer_f.zero_grad()

        # Obtain forwarder output (probablities)
        generated_batch, _ = forwarder(previous_batch, temperature)

        # Sample a sequence from forwarder output probablities via Gumbel-Softmax
        # https://arxiv.org/pdf/1611.01144.pdf
        sampled_batch = F.gumbel_softmax(generated_batch, tau=1.0, hard=True)

        # Obtain critic score
        energy, entropy_score = critic(sampled_batch, temperature)
        loss_f = torch.mean(entropy_score + temperature_schedule(args, timestep + args.delta_time) * energy)

        # De-onehot the sampled sequence, then store the sequence in the pool
        sampled_batch_seq = torch.argmax(sampled_batch, dim = -1).detach().cpu()
        if timestep + args.delta_time < args.timesteps:
            pool.store_data(sampled_batch_seq, timestep + args.delta_time)
        
        # Backpropagate & update
        loss_f.backward()
        optimizer_f.step()

        # TODO: Seperately train the forwarder and critic?
        ###############################################################################
        # Loss for critic
        ###############################################################################

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        # 1 redundant forward pass for the critic. Can we avoid this?
        pred_energy, entropy_score_generated = critic(sampled_batch.detach(), temperature)

        # Obtain energy from the scoring server
        true_energy = query_score(env, sampled_batch_seq.numpy()).to(device)

        # Obtain uniform entropy score for MINE
        _, entropy_score_uniform = critic(uniform_batch, temperature)
        
        loss_g = F.smooth_l1_loss(pred_energy, true_energy)
        loss_h = torch.mean(entropy_score_generated) - torch.log(torch.mean(torch.exp(entropy_score_uniform)) + EPS)
        loss_critic = loss_g - loss_h

        # Backpropagate & update
        loss_critic.backward()
        optimizer_c.step()
        
        ###############################################################################
        # Logging
        ###############################################################################

        estimated_entropy = - loss_h
        trueenergy_log[timestep + args.delta_time] = true_energy.mean().item()

        # Log losses and true score to wandb
        wandb.log({
            "Loss/loss_f": loss_f.item(),
            "Loss/loss_g": loss_g.item(),
            "Loss/loss_h": loss_h.item(),
            "Loss/loss_critic": loss_critic.item(),

            "estimated_energy": pred_energy.mean().item(),
            "estimated_entropy": estimated_entropy.item(),
            "raw_true_energy": true_energy.mean().item(),
            "true_energy_at_10": trueenergy_log[10],
            "true_energy_at_T": trueenergy_log[-1],
        }, step = it)

        visualization.add_point(timestep + args.delta_time, true_energy.mean().item(), it)

        if it % visualization.plot_interval == 0:
            visualization.log(it)

        if it % 100 == 0:
            print(
                colored(
                    "[TRAINING] Iter: {:8d}, Loss_f: {:9.6f}, Loss_g: {:9.6f}, Loss_h: {:9.6f}, Loss_critic: {:9.6f}, True score: {:9.6f}"
                    .format(it, loss_f.item(), loss_g.item(), loss_h.item(), loss_critic.item(), true_energy.mean()),
                "blue"
            ))

        ###############################################################################
        # Save model
        ###############################################################################

        if (it % args.ckpt_interval == 0 and it > 0) or it == args.n_iters - 1:

            torch.save({
                'iteration': it,
                'forwarder': forwarder.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_f': optimizer_f.state_dict(),
                'optimizer_c': optimizer_c.state_dict(),
                'pool': pool,
                'trueenergy_log': trueenergy_log,
            }, args.savepath)
            print(colored("[INFO] Saved checkpoint at %s" % (args.savepath), "yellow"))

        it += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='WGR-DMGIBBS')

    parser.add_argument('--serverIP', type=str, default='127.0.0.1')
    parser.add_argument('--serverPort', type=int, default=4445)
    parser.add_argument('--env', type=str, default='testCity')

    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--pool_size', type=int, default=16384)
    parser.add_argument('--seq_length', type=int, default=4096)
    parser.add_argument('--timesteps', type=int, default=25)
    parser.add_argument('--delta_time', type=int, default=1)
    parser.add_argument('--num_alphabet', type=int, default=256)

    parser.add_argument('--dont_invert_score', action='store_true')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--invtemp_min', type=float, default=0.0)
    parser.add_argument('--invtemp_max', type=float, default=1.0)
    parser.add_argument('--energy_scale', type=float, default=0.001)

    parser.add_argument('--plot_interval', type=int, default=1024)
    parser.add_argument('--ckpt_interval', type=int, default=1024)

    parser.add_argument('--modelname', type=str, default='placeholder')
    parser.add_argument('--run_comment', type=str, default='uncommented')

    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # Pre-multiply the energy scale
    # args.invtemp_min *= args.energy_scale
    # args.invtemp_max *= args.energy_scale

    # Get a unique name by datetime
    args.run_name = "%s-%s-%s-%s" % (args.env, args.modelname, args.run_comment, datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.savedir = "saved_models/%s/%s-%s[%s]" % (args.env, args.modelname, args.run_comment, datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.savepath = os.path.join(args.savedir, "latest.pt")

    main(args)
