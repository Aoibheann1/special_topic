# import pytest


def test_import_check_bcs():
    from diffusion_package.pde import PDESolver # noqa F401


# @pytest.mark.parametrize("bc_left_value, bc_right_value, bc_left_type, bc_right_type, value", [(0,0, "neumann", "neumann", False),
#                                           (2, True),
#                                           (3, True),
#                                           (4, False),
#                                           (47, True),
#                                           (100, False)])
# def test_isprime(value, prime):
#     from diffusion_package.pde import PDESolver 

#     assert PDESolver.check_bcs() == prime, "Incorrect return value from isprime"

import numpy as np
from diffusion_package.pde import PDESolver

def test_check_bcs():
    pde_solver = PDESolver(1, 1, 0.1, 0, 1, 10, 0, 1)
    bc_left_value = 0
    bc_right_value = 1
    bc_left_type = "neumann"
    bc_right_type = "dirichlet"
    bc_left, bc_right = pde_solver.check_bcs(bc_left_value, bc_right_value, bc_left_type, bc_right_type)

    assert isinstance(bc_left, type(pde_solver.bc_left))
    assert isinstance(bc_right, type(pde_solver.bc_right))
    assert bc_left.value == bc_left_value
    assert bc_right.value == bc_right_value
    assert type(bc_left).__name__ == bc_left_type.capitalize() + "BoundaryCondition"
    assert type(bc_right).__name__ == bc_right_type.capitalize() + "BoundaryCondition"