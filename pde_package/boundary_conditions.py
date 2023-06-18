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

    def __init__(self, value):
        self.value = value

    def __str__(self):
        """Return a string containing the type and value of the boundary condition."""
        return f"The value of the boundary condition is = {self.value}"


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition."""

    def operation(self, dc, c, index, dx):
        dc[index] = (2 * c[3 * index + 1] - 2 * c[index]
                     - (-1) ** index * dx * self.value) / dx ** 2
        return dc


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition."""

    def operation(self, dc, c, index):
        dc[index] = 0
        c[index] = self.value


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
        if len(values) != 2 or len(types) != 2:
            raise ValueError("Invalid number of values or types provided")

        bc = [None, None]

        for i in range(2):
            if types[i] == "neumann":
                bc[i] = NeumannBC(values[i])
            elif types[i] == "dirichlet":
                bc[i] = DirichletBC(values[i])
            else:
                raise ValueError("Invalid boundary condition type")

        return bc

