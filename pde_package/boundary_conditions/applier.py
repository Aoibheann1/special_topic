from .dirichlet import DirichletBC
from .neumann import NeumannBC

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
