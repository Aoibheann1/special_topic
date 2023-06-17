from functools import singledispatch
import boundary
import numpy as np


class Diffusion:
    def __init__(self, a1, a2, dx):
        self.a1 = a1
        self.a2 = a2
        self.dx = dx

    def apply_boundary_conditions(self, d2C1_dx2, d2C2_dx2, bc_left, bc_right, C1, C2):
        boundarycond(bc_left)(d2C1_dx2, C1)
        boundarycond(bc_right)(d2C2_dx2, C2)

    def discretize(self, bc_left, bc_right):
        def decorator(f):
            def pde_system(x, t, C):
                n = len(C) // 2
                C1 = C[:n]
                C2 = C[n:]

                d2C1_dx2 = np.zeros_like(C1)
                d2C2_dx2 = np.zeros_like(C2)

                # Compute second derivative of C1 and C2 using central difference
                d2C1_dx2[1:-1] = (C1[:-2] - 2 * C1[1:-1] + C1[2:]) / self.dx**2
                d2C2_dx2[1:-1] = (C2[:-2] - 2 * C2[1:-1] + C2[2:]) / self.dx**2

                # Apply boundary conditions
                self.apply_boundary_conditions(d2C1_dx2, d2C2_dx2, bc_left, bc_right, C1, C2)

                # Compute the time derivatives
                dC1_dt = self.a1 * d2C1_dx2
                dC2_dt = self.a2 * d2C2_dx2

                return np.concatenate((dC1_dt, dC2_dt))

            return pde_system

        return decorator


@singledispatch
def boundarycond(bc, *o, **kwargs):
    raise NotImplementedError(
        f"Cannot implement this boundary condition {type(bc).__name__}")


@boundarycond.register(boundary.NeumannBoundaryCondition)
def _(bc, *o, **kwargs):
    return bc.value


@boundarycond.register(boundary.DirichletBoundaryCondition)
def _(bc, *o, **kwargs):
    return bc.value
