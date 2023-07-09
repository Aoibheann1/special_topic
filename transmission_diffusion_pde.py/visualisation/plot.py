"""docstring."""
import numpy as np
import matplotlib.pyplot as plt
from .base import BasePlot


class SpecifiedTimePlot(BasePlot):
    """Class for plotting a single instance of the solution."""

    def calculate_time_index(self, time_fraction: float) -> int:
        """Calculate the time index corresponding to a specified time fraction.

        Args:
            time_fraction (float): Fraction of the end time.

        Returns:
            int: Time index.

        Raises:
            ValueError: If the time fraction is outside the range [0, 1].
        """
        if not 0 <= time_fraction <= 1:
            raise ValueError("Invalid time fraction. The fraction should be "
                             "between 0 and 1.")

        desired_time = self.t[-1] * time_fraction

        # Calculate the fractional indices for the desired time
        fractional_indices = np.interp(desired_time, self.t,
                                       np.arange(len(self.t)))

        # Round the fractional indices to the nearest integer
        time_index = np.round(fractional_indices).astype(int)

        return time_index

    def plot_solution(self, time_fraction: float):
        """
        Plot the solution at a specified fraction of the end time.

        Args:
            time_fraction (float): Fraction of the end time.
        """
        time_index = self.calculate_time_index(time_fraction)

        plt.plot(self.x1, self.c1[:, time_index], 'bo', markersize=2,
                 label='C1')
        plt.plot(self.x2, self.c2[:, time_index], 'ro', markersize=2,
                 label='C2')
        plt.xlabel('x')
        plt.ylabel('c')
        plt.xlim(np.min(self.x1), np.max(self.x2))
        plt.ylim(0, None)
        plt.title('Plot of dimensionless concentration at '
                  f't={self.t[time_index]:.2e}s')
        plt.legend()

    def show(self, time_fraction: float = 1.0):
        """
        Show the plot.

        Args:
            time_fraction (float): Fraction of the end time.
        """
        self.plot_solution(time_fraction)
        plt.show()

    def save(self, filename: str, time_fraction: float = 1.0, dpi: int = 100):
        """
        Save the plot as an image file.

        Args:
            filename (str): Name of the file to save.
            time_fraction (float): Fraction of the end time.
            dpi (int, optional): Resolution of the saved image in dots per
            inch. Defaults to 100.
        """
        self.plot_solution(time_fraction)
        plt.savefig(filename, dpi=dpi)
