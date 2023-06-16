import numpy as np


class Diffusion:
    def __init__(self, D1, D2, dx):
        self.D1 = D1
        self.D2 = D2
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

        # Boundary conditions at x = -1
        d2C1_dx2[0] = (2 * C1[1] - 2 * C1[0]) / self.dx**2

        # Boundary conditions at x = 1
        d2C2_dx2[-1] = (2 * C2[-2] - 2 * C2[-1]) / self.dx**2

        # Boundary conditions at x = 0
        d2C1_dx2[-1] = ((2 * self.D2 + self.D1) / (self.D1 + self.D2) * C1[-2]
                        - 2 * C1[-1] + self.D2 / (self.D1 + self.D2)
                        * C2[1]) / self.dx**2
        d2C2_dx2[0] = ((self.D2 + 2 * self.D1) / (self.D1 + self.D2) * C2[1]
                       - 2 * C2[0] + self.D1 / (self.D1 + self.D2)
                       * C1[-2]) / self.dx**2

        # Compute the time derivatives
        dC1_dt = self.D1 * d2C1_dx2
        dC2_dt = self.D2 * d2C2_dx2

        return np.concatenate((dC1_dt, dC2_dt))
