"""Module for animating the diffusion process.

This module provides the `Animate` class, which animates the diffusion process.
It utilizes the `matplotlib` library to generate the animation based on the
provided concentration data.

Classes:
- Animate: Class for animating the diffusion process.

"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .base import BasePlot


class Animate(BasePlot):
    """Class for animating the diffusion process."""

    def animate_solution(self):
        """
        Create and animate the diffusion process.

        This method sets up the figure, axes, and initial plot configuration.
        It creates an animation by updating the plot data for each frame based
        on the provided time steps.

        """
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'bo', markersize=2, label='C1')
        self.line2, = self.ax.plot([], [], 'ro', markersize=2, label='C2')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('C')
        self.ax.set_xlim(np.min(self.x1), np.max(self.x2))
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Time = ')
        self.ax.legend()

        self.anim = FuncAnimation(self.fig, self.update_plot,
                                  frames=len(self.t), interval=200)

    def update_plot(self, frame: int):
        """
        Update the plot for a given frame.

        Args:
            frame (int): The frame index.

        This method updates the plot data for the specified frame.
        """
        self.line1.set_data(self.x1, self.c1[:, frame])
        self.line2.set_data(self.x2, self.c2[:, frame])
        self.ax.set_title(f'Time = {self.t[frame]:.2e}s')

    def show(self):
        """
        Show the animation plot.

        This method displays the animated plot on the screen.
        """
        self.animate_solution()
        plt.show()

    def save(self, filename: str, dpi: int = 100):
        """
        Save the animation plot as a video file.

        Args:
            filename (str): The filename of the output video file.
            dpi (int): The resolution of the video in dots per inch.

        This method saves the animation as a video file.
        """
        self.animate_solution()
        self.anim.save(filename, dpi=dpi)
