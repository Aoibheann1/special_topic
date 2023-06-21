from abc import ABC, abstractmethod

class PDE(ABC):
    """Abstract base class for PDEs."""

    @abstractmethod
    def solve_pde_system(self, c_initial, boundary_values, boundary_types):
        """
        Solve the PDE system.

        Args:
            c_initial (numpy.ndarray): Initial concentration array.
            boundary_values (List[float]): List of boundary values.
            boundary_types (List[str]): List of boundary condition types.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Solution concentrations and
            time points.
        """
        pass
