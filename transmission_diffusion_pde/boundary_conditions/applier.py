"""Module for applying boundary conditions.

This module provides the `BoundaryConditionApplier` class, which is
responsible for applying boundary conditions to input data.

Classes:
- BoundaryConditionApplier: A class that applies boundary conditions to input
                            data.

Exceptions:
- ValueError: Raised when an invalid number of values or types is provided or
              when an invalid boundary condition type is encountered.

"""

from .dirichlet import DirichletBC
from .neumann import NeumannBC
from .base import BoundaryCondition
from typing import List


class BoundaryConditionApplier:
    """A class that applies boundary conditions to input data."""

    def __init__(self, values: List[float], types: List[str]):
        """Initialise the BoundaryConditionApplier.

        Parameters
        ----------
        values : List[float]
            List of input values for the boundary conditions.
        types : List[str]
            List of types for the boundary conditions.
        """
        self.values = values
        self.types = types
        self._validate_input()

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

    def generate_boundary_condition_instances(self) -> List[BoundaryCondition]:
        """Generate boundary condition instances using input values and types.

        Returns
        -------
        List[BoundaryCondition]
            List of boundary condition instances.
        """
        bc_mapping = {
            "Neumann": NeumannBC,
            "Dirichlet": DirichletBC
        }

        bc_instance_list = []

        for i, (value, bc_type) in enumerate(zip(self.values, self.types)):
            if bc_type not in bc_mapping:
                raise ValueError("Invalid boundary condition type")
            bc_class = bc_mapping[bc_type]
            bc_instance = bc_class(value, i)
            bc_instance_list.append(bc_instance)

        return bc_instance_list
