import matplotlib.pyplot as plt

from .base import BasePlot


class SpecifiedTimePlot(BasePlot):
    """Class for plotting a single instance of the solution."""

    def plot_solution(self, time_index: int):
        """Plot the solution at a specified time index."""
        plt.plot(self.x1, self.c1[:, time_index], 'bo', markersize=2, label='C1')
        plt.plot(self.x2, self.c2[:, time_index], 'ro', markersize=2, label='C2')
        plt.xlabel('x')
        plt.ylabel('c')
        plt.title(f'Plot of concentration at t={self.t[time_index]}')
        plt.legend()

    def show(self, time_index: int):
        """Show the plot."""
        self.plot_solution(time_index)
        plt.show()

    def save(self, filename: str, time_index: int, dpi: int = 100):
        """Save the plot as an image file."""
        self.plot_solution(time_index)
        plt.savefig(filename, dpi=dpi)
