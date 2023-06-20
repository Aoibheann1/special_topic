"""A module implementing the boundary conditions."""

from numbers import Number


class BoundaryCondition:
    """A base class for boundary conditions.

    Each subclass represents a type of boundary condition.

    Parameters
    ----------
    value: float
        The input value of the boundary condition
    """

    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __str__(self):
        """Return a string of the type and value of the boundary condition."""
        return f"{self.symbol}[{self.index}] = {self.value}"

    def _validate(self):
        """Ensure that the value is a number."""
        if not (isinstance(self.value, Number)):
            raise ValueError("The input boundary condition must be a number"
                             f"not a {type(self.value)}")


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition."""

    symbol = "dC/dx"

    def operation(self, dc, c, dx):
        dc[self.index] = (2 * c[3 * self.index + 1] - 2 * c[self.index]
                          - (-1) ** self.index * dx * self.value) / dx ** 2
        return dc


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition."""

    symbol = "C"

    def operation(self, dc, c, dx):
        dc[self.index] = 0
        c[self.index] = self.value


class BoundaryConditionApplier:
    """A class that applies boundary conditions to input data."""

    def generate_boundary_conditions(self, values, types):
        """Generate boundary condition instances based on input values and types.

        Parameters
        ----------
        values : list
            List of input values for the boundary conditions.
        types : list
            List of types for the boundary conditions.
        Returns
        -------
        list
            List of BoundaryCondition instances representing the boundary conditions.
        """
        if len(values) != 2:
            raise ValueError("Invalid number of values provided")
        if len(types) != 2:
            raise ValueError("Invalid number of types provided")

        bc = [None, None]

        for i in [0, -1]:
            if types[i] == "neumann":
                bc[i] = NeumannBC(values[i], i)
            elif types[i] == "dirichlet":
                bc[i] = DirichletBC(values[i], i)
            else:
                raise ValueError("Invalid boundary condition type")

        return bc
