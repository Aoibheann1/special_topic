from abc import ABC, abstractmethod
import numpy as np

class BasePlot(ABC):
    """Abstract base class for plotting."""

    def __init__(self, x1: np.ndarray, x2: np.ndarray,
                 c1: np.ndarray, c2: np.ndarray, t: np.array):
        """
        Initialize the Plotter instance.

        Args:
            x1 (numpy.ndarray): Array of x-coordinates for C1.
            x2 (numpy.ndarray): Array of x-coordinates for C2.
            c1 (numpy.ndarray): Array of concentration values for C1.
            c2 (numpy.ndarray): Array of concentration values for C2.
        """
        self.x1 = x1
        self.x2 = x2
        self.c1 = c1
        self.c2 = c2
        self.t = t

    @abstractmethod
    def show(self):
        """show the plot."""
        pass
