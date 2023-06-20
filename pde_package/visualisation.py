"""Visualisation module."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Visualization:
    """Class for visualizing the diffusion process."""

    def __init__(self):
        """
        Initialize the Visualization instance.

        Returns:
            None
        """
        self.fig, self.ax = None, None
        self.line1, self.line2 = None, None
        self.anim = None

    def initialize_plot(self, x1, x2, c1, c2, t):
        """
        Initialize the plot.

        Args:
            x1 (numpy.ndarray): Array of x-coordinates for C1.
            x2 (numpy.ndarray): Array of x-coordinates for C2.
            c1 (numpy.ndarray): Array of concentration values for C1.
            c2 (numpy.ndarray): Array of concentration values for C2.
            t (numpy.ndarray): Array of time values.

        Returns:
            None
        """
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'bo', markersize=2, label='C1')
        self.line2, = self.ax.plot([], [], 'ro', markersize=2, label='C2')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('c')
        self.ax.set_xlim(np.min(x1), np.max(x2))
        self.ax.set_ylim(0, 1)
        self.ax.legend()

        def update_title(frame):
            self.ax.set_title(f'Time = {t[frame]}')

        self.ax.set_title('Time = ')
        self.anim = FuncAnimation(self.fig, update_title, frames=len(t))

    def update(self, frame):
        """
        Update the plot for a given frame.

        Args:
            frame (int): Frame index.

        Returns:
            None
        """
        self.line1.set_data(self.x1, self.c1[:, frame])
        self.line2.set_data(self.x2, self.c2[:, frame])

    def animate(self):
        """
        Animate the plot.

        Returns:
            None
        """
        if self.anim is None:
            self.anim = FuncAnimation(self.fig, self.update, frames=len(self.t), interval=200)

    def show(self):
        """
        Display the plot.

        Returns:
            None
        """
        plt.show()
