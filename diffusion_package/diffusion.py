import numpy as np


class Diffusion:
    def __init__(self, a1, a2, dx, bc):
        self.a1 = a1
        self.a2 = a2
        self.dx = dx
        self.bc = bc

    def pde_system(self, x, t, c):
        n = len(c) // 2
        c1 = c[:n]
        c2 = c[n:]

        dc = np.zeros_like(c)
        d2c1_dx2 = dc[:n]
        d2c2_dx2 = dc[n:]

        # Boundary conditions at x = -1
        self.bc[0].operation(dc, c, self.dx)

        # Boundary conditions at x = 1
        self.bc[-1].operation(dc, c, self.dx)

        # Compute second derivative of C1 and C2 using central difference
        d2c1_dx2[1:-1] = (c1[:-2] - 2 * c1[1:-1] + c1[2:]) / self.dx**2
        d2c2_dx2[1:-1] = (c2[:-2] - 2 * c2[1:-1] + c2[2:]) / self.dx**2

        # Boundary conditions at x = 0
        d2c1_dx2[-1] = ((1 + 2 * self.a2) / (1 + 2 * self.a2) * c1[-2]
                        - 2 * c1[-1] + c2[1] / (1 + self.a2)) / self.dx**2
        d2c2_dx2[0] = ((2 + self.a2) / (1 + self.a2) * c2[1] - 2 * c2[0]
                       + self.a2 / (1 + self.a2) * c1[-2]) / self.dx**2

        # Compute the time derivatives
        dc1_dt = self.a1 * d2c1_dx2
        dc2_dt = d2c2_dx2

        return np.concatenate((dc1_dt, dc2_dt))
