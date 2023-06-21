import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .base import BasePlot


class Animate(BasePlot):
    """Class for animating the diffusion process."""

    def initialize_plot(self):
        """Initialize the plot."""
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'bo', markersize=2, label='C1')
        self.line2, = self.ax.plot([], [], 'ro', markersize=2, label='C2')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('c')
        self.ax.set_xlim(np.min(self.x1), np.max(self.x2))
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Time = ')
        self.ax.legend()

        self.anim = FuncAnimation(self.fig, self.update_plot, frames=len(self.t), interval=200)

    def update_plot(self, frame: int):
        """Update the plot for a given frame."""
        self.line1.set_data(self.x1, self.c1[:, frame])
        self.line2.set_data(self.x2, self.c2[:, frame])
        self.ax.set_title(f'Time = {self.t[frame]}')

    def show(self):
        """Show the plot."""
        self.initialize_plot()
        plt.show()

    def save(self, filename: str, dpi: int = 100):
        """Save the animation as a video file."""
        self.initialize_plot()
        self.anim.save(filename, dpi=dpi)
