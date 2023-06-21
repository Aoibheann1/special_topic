import numpy as np
from .base import BoundaryCondition


class DirichletBC(BoundaryCondition):
    """A class representing a Dirichlet boundary condition."""

    symbol = "C"

    def apply(self, dc: np.ndarray, c: np.ndarray, dx: float) -> np.ndarray:
        """Apply the Dirichlet boundary condition operation.

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
        dc[-self.index] = 0
        c[-self.index] = self.value
        return dc
