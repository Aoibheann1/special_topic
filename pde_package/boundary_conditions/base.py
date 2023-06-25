"""docstring."""
from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict
import numpy as np


class BoundaryCondition(ABC):
    """An abstract base class for boundary conditions."""

    def __init__(self, value: float, index: int):
        """Initialize the BoundaryCondition instance.

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
                "The input boundary condition must be a number, not"
                f"{type(self.value).__name__}."
            )

    @abstractmethod
    def apply(
        self,
        dc: np.ndarray,
        c: np.ndarray,
        dx: float,
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """Apply the boundary condition operation.

        Parameters:
        - dc (numpy.ndarray): Array of concentration derivatives.
        - c (numpy.ndarray): Array of concentration values.
        - dx (float): Step size.
        - parameters (Dict[str, Any]): Boundary parameters.

        Returns:
        - numpy.ndarray: Updated array of concentration derivatives.
        """
        pass
