import numpy as np


class Diffusion:
    def __init__(self, a1, a2, dx):
        self.a1 = a1
        self.a2 = a2
        self.dx = dx

    def pde_system(self, x, t, C):
        n = len(C) // 2
        C1 = C[:n]
        C2 = C[n:]

        d2C1_dx2 = np.zeros_like(C1)
        d2C2_dx2 = np.zeros_like(C2)

        # Compute second derivative of C1 and C2 using central difference
        d2C1_dx2[1:-1] = (C1[:-2] - 2 * C1[1:-1] + C1[2:]) / self.dx**2
        d2C2_dx2[1:-1] = (C2[:-2] - 2 * C2[1:-1] + C2[2:]) / self.dx**2

        # Neumann Boundary conditions at x = -1
        d2C1_dx2[0] = (2 * C1[1] - 2 * C1[0]) / self.dx**2

        # Neumann Boundary conditions at x = 1
        d2C2_dx2[-1] = (2 * C2[-2] - 2 * C2[-1]) / self.dx**2

        # Boundary conditions at x = 0
        d2C1_dx2[-1] = ((1 + 2 * self.a2) / (1 + 2 * self.a2) * C1[-2]
                        - 2 * C1[-1] + C2[1] / (1 + self.a2)) / self.dx**2
        d2C2_dx2[0] = ((2 + self.a2) / (1 + self.a2) * C2[1] - 2 * C2[0]
                       + self.a2 / (1 + self.a2) * C1[-2]) / self.dx**2

        # Compute the time derivatives
        dC1_dt = self.a1 * d2C1_dx2
        dC2_dt = d2C2_dx2

        return np.concatenate((dC1_dt, dC2_dt))
