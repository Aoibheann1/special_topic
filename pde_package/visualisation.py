"""Visualisation module."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Visualisation:
    """Class for visualising the diffusion process."""

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 c1: np.ndarray, c2: np.ndarray, t: np.ndarray):
        """Initialise the Visualisation instance.

            Args:
        x1 (numpy.ndarray): Array of x-coordinates for C1.
        x2 (numpy.ndarray): Array of x-coordinates for C2.
        c1 (numpy.ndarray): Array of concentration values for C1.
        c2 (numpy.ndarray): Array of concentration values for C2.
        t (numpy.ndarray): Array of time values.
        """
        self.x1 = x1
        self.x2 = x2
        self.c1 = c1
        self.c2 = c2
        self.t = t
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.anim = None

    def initialise_plot(self):
        """Initialise the plot."""
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'bo', markersize=2, label='C1')
        self.line2, = self.ax.plot([], [], 'ro', markersize=2, label='C2')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('c')
        self.ax.set_xlim(np.min(self.x1), np.max(self.x2))
        self.ax.set_ylim(0, 1)
        self.ax.legend()

        def update_title(frame):
            self.ax.set_title(f'Time = {self.t[frame]}')

        self.ax.set_title('Time = ')
        self.anim = FuncAnimation(self.fig, self.update_plot,
                                  frames=len(self.t), interval=200)

    def update_plot(self, frame: int):
        """
        Update the plot for a given frame.

        Args:
            frame (int): Frame index.
            x1 (numpy.ndarray): Array of x-coordinates for C1.
            x2 (numpy.ndarray): Array of x-coordinates for C2.
            c1 (numpy.ndarray): Array of concentration values for C1.
            c2 (numpy.ndarray): Array of concentration values for C2.
        """
        self.line1.set_data(self.x1, self.c1[:, frame])
        self.line2.set_data(self.x2, self.c2[:, frame])

    def animate(self):
        """Animate the plot."""
        plt.show()

    def show(self):
        """Display the plot."""
        if not plt.isinteractive():
            plt.show()
