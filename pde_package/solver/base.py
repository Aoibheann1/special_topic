"""Solver base class."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseSolver(ABC):
    """Abstract base class for PDEs."""

    def __init__(
        self,
        diffusion_coefficient1: float,
        diffusion_coefficient2: float,
        len_region1: float,
        len_region2: float,
        a: float,
        combined_n: int,
        t_start: float,
        t_end: float,
        c1_initial: np.ndarray,
        c2_initial: np.ndarray,
        left_bc_value: float,
        right_bc_value: float,
        left_bc_type: str,
        right_bc_type: str
    ) -> None:
        """
        Initialize the BaseSolver.

        Args:
            diffusion_coefficient1 (float): Diffusion coefficient for region 1.
            diffusion_coefficient2 (float): Diffusion coefficient for region 2.
            len_region1 (float): Length of region 1.
            len_region2 (float): Length of region 2.
            a (float): Constant value.
            combined_n (int): Total number of spatial points over both regions.
            t_start (float): Start time.
            t_end (float): End time.
            c1_initial (numpy.ndarray): Initial concentration values for
                                        region 1.
            c2_initial (numpy.ndarray): Initial concentration values for
                                        region 2.
            left_bc_value (float): Value of the left boundary condition.
            right_bc_value (float): Value of the right boundary condition.
            left_bc_type (str): Type of the left boundary condition.
            right_bc_type (str): Type of the right boundary condition.
        """
        parameters = {
            'diffusion_coefficient1': diffusion_coefficient1,
            'diffusion_coefficient2': diffusion_coefficient2,
            'len_region1': len_region1,
            'len_region2': len_region2,
            'a': a,
            'combined_n': combined_n,
            'c_max': np.max(c1_initial)
        }
        self._validate_parameters(parameters, c1_initial, c2_initial, t_start,
                                  t_end)
        self.parameters = parameters
        self.dx = 2.0 / combined_n
        self.n = combined_n // 2
        self.a1 = self._calculate_a1()
        self.a2 = self._calculate_a2()
        self.c_initial = self._calculate_normalized_initial_conditions(
            c1_initial, c2_initial)
        self.values = [left_bc_value, right_bc_value]
        self.types = [left_bc_type, right_bc_type]
        self.T_start = t_start * diffusion_coefficient2 / (len_region2 ** 2)
        self.T_end = t_end * diffusion_coefficient2 / (len_region2 ** 2)

    @abstractmethod
    def solve_pde_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
        """
        Solve the PDE system.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray]: Solution concentrations, spatial points, and time
            points.
        """
        pass

    def _validate_parameters(
        self,
        parameters: dict,
        c1_initial: np.ndarray,
        c2_initial: np.ndarray,
        t_start: float,
        t_end: float
    ) -> None:
        """
        Validate the input parameters.

        Args:
            parameters (dict): Dictionary containing the parameter values.
            c1_initial (numpy.ndarray): Initial concentration values for
                                        region 1.
            c2_initial (numpy.ndarray): Initial concentration values for
                                        region 2.
            t_start (float): Start time.
            t_end (float): End time.

        Raises:
            ValueError: If any parameter values are invalid.
        """
        if any(value <= 0 for value in parameters.values()):
            invalid_params = [param for param, value in parameters.items() if
                              value <= 0]
            raise ValueError("The following parameter(s) must be greater than"
                             f"zero: {', '.join(invalid_params)}")

        if t_start < 0:
            raise ValueError("t_start must be a non-negative value.")

        if t_end < 0:
            raise ValueError("t_end must be a non-negative value.")

        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start.")

        if np.any(c1_initial <= 0):
            raise ValueError("c1_initial must contain positive values.")

        if np.any(c2_initial < 0):
            raise ValueError("c2_initial cannot contain negative values.")

    def _calculate_a1(self) -> float:
        """
        Calculate the value of a1.

        Returns:
            float: The value of a1.
        """
        diffusion_coefficient1 = self.parameters['diffusion_coefficient1']
        diffusion_coefficient2 = self.parameters['diffusion_coefficient2']
        len_region1 = self.parameters['len_region1']
        len_region2 = self.parameters['len_region2']

        return (diffusion_coefficient1 / diffusion_coefficient2
                * (len_region2 / len_region1) ** 2)

    def _calculate_a2(self) -> float:
        """
        Calculate the value of a2.

        Returns:
            float: The value of a2.
        """
        diffusion_coefficient1 = self.parameters['diffusion_coefficient1']
        diffusion_coefficient2 = self.parameters['diffusion_coefficient2']
        len_region1 = self.parameters['len_region1']
        len_region2 = self.parameters['len_region2']
        a = self.parameters['a']

        return (diffusion_coefficient1 / diffusion_coefficient2
                * (len_region2 / len_region1) / a)

    def _calculate_normalized_initial_conditions(
        self,
        c1_initial: np.ndarray,
        c2_initial: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the normalized initial conditions.

        Args:
            c1_initial (numpy.ndarray): Initial concentration values for
                                        region 1.
            c2_initial (numpy.ndarray): Initial concentration values for
                                        region 2.

        Returns:
            numpy.ndarray: The normalized initial conditions.
        """
        c_max = self.parameters['c_max']
        c_initial = np.concatenate((c1_initial, c2_initial /
                                    self.parameters['a']))
        return c_initial / c_max
