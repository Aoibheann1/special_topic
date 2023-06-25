"""docstring."""
from abc import ABC, abstractmethod
import numpy as np


class BasePlot(ABC):
    """Abstract base class for plotting."""

    def __init__(self, x1: np.ndarray, x2: np.ndarray, c1: np.ndarray,
                 c2: np.ndarray, t: np.ndarray):
        """
        Initialize the BasePlot instance.

        Args:
            x1 (numpy.ndarray): Array of x-coordinates for C1.
            x2 (numpy.ndarray): Array of x-coordinates for C2.
            c1 (numpy.ndarray): Array of concentration values for C1.
            c2 (numpy.ndarray): Array of concentration values for C2.
            t (numpy.ndarray): Array of time values.

        Raises:
            ValueError: If any of the arrays are empty or if x1, x2, c1, or c2
                        have different lengths.
        """
        self.x1 = x1
        self.x2 = x2
        self.c1 = c1
        self.c2 = c2
        self.t = t

        self._validate_input()

    def _validate_input(self):
        """
        Validate the input arrays.

        Raises:
            ValueError: If any of the arrays are empty or if x1, x2, c1, or c2
                        have different lengths.
        """
        if (not self.x1.size or not self.x2.size or not self.c1.size
           or not self.c2.size or not self.t.size):
            raise ValueError("Input arrays cannot be empty.")

        lengths = {
            'x1': len(self.x1),
            'x2': len(self.x2),
            'c1': len(self.c1),
            'c2': len(self.c2)
        }

        if len(set(lengths.values())) > 1:
            inconsistent_lengths = [name for name, length in lengths.items()
                                    if length != lengths['x1']]
            if inconsistent_lengths:
                raise ValueError("Inconsistent lengths detected. The "
                                 "following arrays have different lengths "
                                 "compared to x1: "
                                 f"{', '.join(inconsistent_lengths)}.")

    @abstractmethod
    def show(self):
        """
        Show the plot.

        This method displays the plot on the screen.
        """
        pass

    @abstractmethod
    def save(self, filename: str, dpi: int = 100):
        """
        Save the plot as an image file.

        Args:
            filename (str): The filename or path of the output image file.
            dpi (int, optional): The resolution in dots per inch (DPI) for the
            output image. Defaults to 100.

        This method saves the plot as an image file with the specified
        filename and resolution.
        """
        pass
