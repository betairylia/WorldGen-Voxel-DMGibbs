import random
import math
import numpy as np

import socket
from struct import pack

import wandb
import plotly.graph_objs as go

import torch
from termcolor import colored

class WGREnv:
    
    def __init__(self, args, seqLength = 8192):
        
        self.name = "WGR-DMGIBBS"
        
        self.ipaddr = args.serverIP
        self.port = args.serverPort
        
        self.population_size = args.batch_size
        
        self.state = None
        self.done = False

        self.energy_scale = -1.0 * args.energy_scale
        if args.dont_invert_score:
            self.energy_scale *= -1.0
        
        # TODO: Get parameters from server
        self.geneLength = seqLength
        self.opsLength = seqLength
        self.opsCount = 2048
        self.population = None
        
        self.connection = socket.socket()
        self.connection.settimeout(10.0)
        self.connection.connect((self.ipaddr, self.port))
        
    # Get energy from the server
    def getEnergy(self, opSeqPopulation, visualize = False):

        # Prepare & send the data to be sent to the server
        ubyte = opSeqPopulation.astype(np.uint8)
        self.connection.send(pack("i", ubyte.size))
        self.connection.send(pack("i", ubyte.shape[0]))
        self.connection.send(pack("i", 1 if visualize else 0))
        self.connection.send(ubyte.tobytes())
        
        # Fitness score from the server
        score = np.frombuffer(self.connection.recv(ubyte.shape[0] * 4), dtype = np.float32, count = ubyte.shape[0])

        # Apply a scaling & inverting before returning the energy
        return torch.Tensor([s * self.energy_scale for s in score])

class WandBScatterMaxlinePlot:

    def __init__(self, args, wandbrun = None):
        self.x_data = []
        self.y_data = []
        self.color_data = []
        self.line_x = []
        self.line_y = []

        if wandbrun is None:
            wandbrun = wandb.run
        self.wandbrun = wandbrun

        self.num_x = args.timesteps
        self.plot_interval = args.plot_interval
        self.max_points = self.plot_interval * 2

    def update_min_line(self):
        x_unique = sorted(set(self.x_data[-self.max_points:]))
        min_y_points = [(x, min([y for x_i, y in zip(self.x_data[-self.max_points:], self.y_data[-self.max_points:]) if x_i == x])) for x in x_unique]

        self.line_x, self.line_y = zip(*min_y_points)

    def update_avg_line(self):
        x_unique = sorted(set(self.x_data[-self.max_points:]))
        avg_y_points = []
        for x in x_unique:
            y_values = [y for x_i, y in zip(self.x_data[-self.max_points:], self.y_data[-self.max_points:]) if x_i == x]
            avg_y = sum(y_values) / len(y_values)
            avg_y_points.append((x, avg_y))

        self.line_x, self.line_y = zip(*avg_y_points)

    def log(self, step):
        scatter = go.Scatter(
            x=self.x_data[-self.max_points:],
            y=self.y_data[-self.max_points:],
            mode='markers',
            showlegend=False,
            marker=dict(
                color=self.color_data[-self.max_points:],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Iteration')
            )
        )
        line = go.Scatter(x=self.line_x, y=self.line_y, mode='lines+markers', showlegend=False, opacity=0.4)

        layout = go.Layout(title='Energy over time')
        fig = go.Figure(data=[scatter, line], layout=layout)

        self.wandbrun.log({"energy_plot": wandb.Plotly(fig)}, step=step)
        print(colored(("[INFO] Iter %d - New fitness plot" % step), "green"))

    def add_point(self, x, y, step):
        self.x_data.append(x)
        self.y_data.append(y)
        self.color_data.append(step)

        # self.update_min_line()
        self.update_avg_line()

if __name__ == "__main__":
    
    print("TODO: Test-cases for WGRenv.py")
    
