import numpy as np
from pde_package.visualisation import Visualisation


def test_initialise_plot():
    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 10)
    c1 = np.ones((10, 10))
    c2 = np.zeros((10, 10))
    t = np.linspace(0, 1, 10)
    visualisation = Visualisation(x1, x2, c1, c2, t)
    visualisation.initialise_plot()
    assert visualisation.fig is not None
    assert visualisation.ax is not None
    assert visualisation.line1 is not None
    assert visualisation.line2 is not None
    assert visualisation.anim is not None


def test_update_plot():
    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 10)
    c1 = np.ones((10, 10))
    c2 = np.zeros((10, 10))
    t = np.linspace(0, 1, 10)
    visualisation = Visualisation(x1, x2, c1, c2, t)
    visualisation.initialise_plot()
    visualisation.update_plot(0)
    assert len(visualisation.line1.get_xdata()) == 10
    assert len(visualisation.line1.get_ydata()) == 10
    assert len(visualisation.line2.get_xdata()) == 10
    assert len(visualisation.line2.get_ydata()) == 10


def test_animate():
    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 10)
    c1 = np.ones((10, 10))
    c2 = np.zeros((10, 10))
    t = np.linspace(0, 1, 10)
    visualisation = Visualisation(x1, x2, c1, c2, t)
    visualisation.initialise_plot()
    visualisation.animate()
    assert True  # No assertion error means animation was successful


def test_show():
    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 10)
    c1 = np.ones((10, 10))
    c2 = np.zeros((10, 10))
    t = np.linspace(0, 1, 10)
    visualisation = Visualisation(x1, x2, c1, c2, t)
    visualisation.initialise_plot()
    visualisation.show()
    assert True  # No assertion error means plot was successfully displayed