import wandb
import plotly.graph_objs as go
import numpy as np

class CustomScatterPlot:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.color_data = []
        self.line_x = []
        self.line_y = []

    def update_max_line(self):
        x_unique = sorted(set(self.x_data))
        max_y_points = [(x, max([y for x_i, y in zip(self.x_data, self.y_data) if x_i == x])) for x in x_unique]

        self.line_x, self.line_y = zip(*max_y_points)

    def log(self, step):
        scatter = go.Scatter(
            x=self.x_data[-512:],
            y=self.y_data[-512:],
            mode='markers',
            showlegend=False,
            marker=dict(
                color=self.color_data,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Iteration')
            )
        )
        line = go.Scatter(x=self.line_x, y=self.line_y, mode='lines+markers', showlegend=False)

        layout = go.Layout(title='Custom Scatter Plot with Max Y Line')
        fig = go.Figure(data=[scatter, line], layout=layout)

        wandb.log({"custom_plot": wandb.Plotly(fig)}, step=step)

    def add_point(self, x, y, step):
        self.x_data.append(x)
        self.y_data.append(y)
        self.color_data.append(step)
        self.update_max_line()

wandb.init(project="custom_scatter_plot")
scatter_plot = CustomScatterPlot()

for step in range(100):
    # Simulate new data points
    new_x = np.random.choice(range(1, 6))
    new_y = np.random.uniform(0, 10)

    scatter_plot.add_point(new_x, new_y, step)
    scatter_plot.log(step)