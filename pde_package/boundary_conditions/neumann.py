import numpy as np
from .base import BoundaryCondition


class NeumannBC(BoundaryCondition):
    """A class representing a Neumann boundary condition."""

    symbol = "dC/dx"

    def apply(self, dc: np.ndarray, c: np.ndarray, dx: float) -> np.ndarray:
        """Apply the Neumann boundary condition operation.

        Parameters
        ----------
        dc : numpy.ndarray
            Array of concentration derivatives.
        c : numpy.ndarray
            Array of concentration values.
        dx : float
            Step size.

        Returns
        -------
        numpy.ndarray
            Updated array of concentration derivatives.
        """
        dc[-self.index] = (2 * c[-3 * self.index + 1]
                           - 2 * c[-self.index]
                           - (-1) ** self.index * dx * self.value
                           ) / dx ** 2
        return dc
