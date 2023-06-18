from functools import singledispatchmethod


class BoundaryCondition:
    def __init__(self, value, dx):
        self.value = value
        self.dx = dx

    def apply(self, d2C_dx2, C):
        @singledispatchmethod
        def apply(self, d2C_dx2, C):
            raise NotImplementedError(
                f"Cannot implement this boundary condition {type(self).__name__}")

        @apply.register(NeumannBoundaryCondition)
        def _(self, d2C_dx2, C):
            d2C_dx2[0] = (2 * C[1] - 2 * C[0]) / self.dx**2
            d2C_dx2[-1] = (2 * C[-2] - 2 * C[-1]) / self.dx**2

        @apply.register(DirichletBoundaryCondition)
        def _(self, d2C_dx2, C):
            d2C_dx2[0] = 0
            d2C_dx2[-1] = 0
            C[0] = self.value
            C[-1] = self.value


class NeumannBoundaryCondition(BoundaryCondition):
    def apply(self, d2C_dx2, C):
        d2C_dx2[0] = (2 * C[1] - 2 * C[0]) / self.dx**2
        d2C_dx2[-1] = (2 * C[-2] - 2 * C[-1]) / self.dx**2


class DirichletBoundaryCondition(BoundaryCondition):
    def apply(self, d2C_dx2, C):
        d2C_dx2[0] = 0
        d2C_dx2[-1] = 0
        C[0] = self.value
        C[-1] = self.value
