"""Module for implementing boundary conditions.

This module defines the `BoundaryCondition` class, which is an abstract base
class for implementing boundary conditions for transmission diffusion PDE
problems.

Classes:
- BoundaryCondition: An abstract base class for boundary conditions.

Exceptions:
- ValueError: Raised when the input value for the boundary condition is not
numerical.

"""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict
import numpy as np


class BoundaryCondition(ABC):
    """An abstract base class for boundary conditions."""

    def __init__(self, value: float, index: int):
        """Initialise the BoundaryCondition instance.

        Parameters:
        - value (float): The input value of the boundary condition.
        - index (int): The index of the boundary condition.
          0 refers to the left boundary and 1 refers to the right boundary.
        """
        self.value = value
        self.index = index
        self._validate_value()

    def __str__(self) -> str:
        """Return a string representation of the boundary condition."""
        return f"{self.symbol}[{self.index}] = {self.value}"

    def _validate_value(self):
        """Ensure that the value is a number.

        Raises:
        - ValueError: If the input boundary condition is not a number.
        """
        if not isinstance(self.value, Number):
            raise ValueError(
                "The input value for the boundary condition must be a number,"
                f" not a {type(self.value).__name__}."
            )

    @abstractmethod
    def apply(self, d2c_dx2: np.ndarray, c: np.ndarray, h: float,
              parameters: Dict[str, float]):
        """Apply the boundary condition operation.

        Parameters:
        - d2c_dx2 (numpy.ndarray): Array of discretised second derivatives of
                                   the concentration.
        - c (numpy.ndarray): Array of concentration values.
        - h (float): Step size.
        - parameters (Dict[str, Any]): Boundary parameters.

        """
        pass
