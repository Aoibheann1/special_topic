import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Visualisation:
    def __init__(self, x, C1_solution, C2_solution, t):
        self.x = x
        self.C1_solution = C1_solution
        self.C2_solution = C2_solution
        self.t = t
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'bo', markersize=2, label='C1')
        self.line2, = self.ax.plot([], [], 'ro', markersize=2, label='C2')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('c')
        self.ax.set_title('Time = ')
        self.ax.set_xlim(np.min(x), np.max(x))
        self.ax.set_ylim(0, 1)
        self.ax.legend()
        self.anim = None

    def update(self, frame):
        self.line1.set_data(self.x[:len(self.x) // 2],
                            self.C1_solution[:, frame])
        self.line2.set_data(self.x[len(self.x) // 2:],
                            self.C2_solution[:, frame])
        self.ax.set_title(f'Time = {self.t[frame]}')

    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update,
                                  frames=len(self.t), interval=200)

    def show(self):
        plt.show()
