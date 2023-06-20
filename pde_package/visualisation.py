import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Visualisation:
    def __init__(self, x1, x2, c1, c2, t):
        self.x1 = x1
        self.x2 = x2
        self.c1 = c1
        self.c2 = c2
        self.t = t
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'bo', markersize=2, label='C1')
        self.line2, = self.ax.plot([], [], 'ro', markersize=2, label='C2')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('c')
        self.ax.set_title('Time = ')
        self.ax.set_xlim(np.min(x1), np.max(x2))
        self.ax.set_ylim(0, 1)
        self.ax.legend()
        self.anim = None

    def update(self, frame):
        self.line1.set_data(self.x1, self.c1[:, frame])
        self.line2.set_data(self.x2, self.c2[:, frame])
        self.ax.set_title(f'Time = {self.t[frame]}')

    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update,
                                  frames=len(self.t), interval=200)

    def show(self):
        plt.show()
