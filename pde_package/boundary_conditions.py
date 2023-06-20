"""A module implementing the boundary conditions."""

from numbers import Number
import numpy as np


class BoundaryCondition:
    """A base class for boundary conditions.

    Each subclass represents a type of boundary condition.
    """

    def __init__(self, value: float, index: int):
        """Initialise the BoundaryCondition instance.

        Parameters
        ----------
        value : float
            The input value of the boundary condition
        index : int
            The index of the boundary condition, 0 referring to the left
            boundary and 1 referring to the right boundary.
        """
        self.value = value
        self.index = index
        self._validate_value()

    def __str__(self) -> str:
        """Return a string representation of the boundary condition."""
        return f"{self.symbol}[{self.index}] = {self.value}"

    def _validate_value(self):
        """Ensure that the value is a number.

        Raises
        ------
        ValueError
            If the input boundary condition is not a number.
        """
        if not isinstance(self.value, Number):
            raise ValueError("The input boundary condition must be a number,"
                             f"not a {type(self.value).__name__}")


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition."""

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
        dc[- self.index] = (2 * c[- 3 * self.index + 1]
                            - 2 * c[- self.index]
                            - (-1) ** self.index * dx * self.value
                            ) / dx ** 2
        return dc


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition."""

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
        dc[- self.index] = 0
        c[- self.index] = self.value


class BoundaryConditionApplier:
    """A class that applies boundary conditions to input data."""

    def __init__(self, values: list, types: list):
        """Initialize the BoundaryConditionApplier.

        Parameters
        ----------
        values : list
            List of input values for the boundary conditions.
        types : list
            List of types for the boundary conditions.
        """
        self.values = values
        self.types = types

    def _validate_input(self):
        """Validate the values and types for generating boundary conditions.

        Raises
        ------
        ValueError
            If an invalid number of values or types is provided.
        """
        if len(self.values) != 2:
            raise ValueError("Invalid number of values provided")
        if len(self.types) != 2:
            raise ValueError("Invalid number of types provided")

    def generate_boundary_condition_instances(self) -> list:
        """Generate boundary condition instances using input values and types.

        Returns
        -------
        list
            List of boundary condition instances.
        """
        self._validate_input()

        bc_mapping = {
            "neumann": NeumannBC,
            "dirichlet": DirichletBC
        }

        bc = [None, None]

        for i, (value, bc_type) in enumerate(zip(self.values, self.types)):
            if bc_type not in bc_mapping:
                raise ValueError("Invalid boundary condition type")
            bc_class = bc_mapping[bc_type]
            bc[i] = bc_class(value, i)

        return bc
