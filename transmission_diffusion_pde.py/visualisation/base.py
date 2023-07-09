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
            ValueError: If any of the arrays are empty.
            ValueError: If the x2 array or the rows of the c1 or c2 arrays
                have a different length to the x1 array.
            ValueError: If the columns of the c1 or c2 arrays have a different
                length to the t array.
        """
        self.x1 = x1
        self.x2 = x2
        self.c1 = c1
        self.c2 = c2
        self.t = t

        self._validate_input()

    def _validate_input(self):
        """Validate the input arrays."""
        if not (self.x1.size and self.x2.size and self.c1.size and self.c2.size
                and self.t.size):
            raise ValueError("Input arrays cannot be empty.")

        x_lengths = {
            'x1': len(self.x1),
            'x2': len(self.x2),
            'c1_rows': np.shape(self.c1)[0] if len(np.shape(self.c1)) > 1 else len(self.c1),
            'c2_rows': np.shape(self.c2)[0] if len(np.shape(self.c2)) > 1 else len(self.c2)
        }

        inconsistent_x_lengths = [name for name, length in x_lengths.items()
                                  if length != x_lengths['x1']]
        if inconsistent_x_lengths:
            raise ValueError(
                "Inconsistent lengths detected. The following arrays have "
                f"different lengths compared to x1: "
                f"{', '.join(inconsistent_x_lengths)}."
            )

        t_lengths = {
            't': len(self.t),
            'c1_cols': np.shape(self.c1)[1] if len(np.shape(self.c1)) > 1 else 1,
            'c2_cols': np.shape(self.c2)[1] if len(np.shape(self.c2)) > 1 else 1
        }

        inconsistent_t_lengths = [name for name, length in t_lengths.items()
                                  if length != t_lengths['t']]
        if inconsistent_t_lengths:
            raise ValueError(
                "Inconsistent lengths detected. The following arrays have "
                f"different lengths compared to t: "
                f"{', '.join(inconsistent_t_lengths)}."
            )

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
