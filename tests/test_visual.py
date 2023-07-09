"""This file contains tests for the classes in the visualisation package.

Modules:
- `numpy` (imported as `np`): Provides numerical computing functionality.
- `matplotlib.pyplot` (imported as `plt`): Provides plotting functionality.
- `pytest`: A testing framework for Python.

Classes:
- `BasePlotTest`: A test class for the `BasePlot` class.

Functions:
- `mock_data()`: A fixture function to generate mock data for testing.

Tests:
- `test_baseplot_initialization()`: Tests the initialization of the `BasePlot`
  class.
- `test_baseplot_valid_input()`: Tests valid input for the `BasePlot` class.
- `test_specifiedtimeplot_calculate_time_index()`: Tests the
  `calculate_time_index` method of the `SpecifiedTimePlot` class.
- `test_specifiedtimeplot_initialization()`: Tests the initialization of the
  `SpecifiedTimePlot` class.
- `test_show()`: Tests the `show` method of the `SpecifiedTimePlot` class.
- `test_save()`: Tests the `save` method of the `SpecifiedTimePlot` class.
- `test_animate_initialization()`: Tests the initialization of the `Animate`
  class.
- `test_show_animate()`: Tests the `show` method of the `Animate` class.
- `test_save_animate()`: Tests the `save` method of the `Animate` class.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from pde_package.visualisation.base import BasePlot
from pde_package.visualisation.plot import SpecifiedTimePlot
from pde_package.visualisation.animate import Animate


class BasePlotTest(BasePlot):
    """Test class for BasePlot."""

    def show(self):
        """Show the plot."""
        pass

    def save(self, filename: str, dpi: int = 100):
        """Save the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            dpi (int, optional): The resolution of the saved plot in dots per
            inch. Defaults to 100.
        """
        pass


@pytest.fixture
def mock_data():
    """Fixture function to generate mock data for testing."""
    t = np.linspace(0, 1, 100)
    x1 = np.linspace(-1, 0, 10)
    c1 = np.random.rand(10, 100)
    x2 = np.linspace(0, 1, 10)
    c2 = np.random.rand(10, 100)
    return x1, x2, c1, c2, t


def test_baseplot_initialization(mock_data):
    """Test initialization of BasePlot."""
    x1, x2, c1, c2, t = mock_data

    # Test valid initialization
    try:
        BasePlotTest(x1, x2, c1, c2, t)
    except Exception as e:
        pytest.fail(f"BasePlot initialization failed: {str(e)}")

    # Test invalid initialization with inconsistent lengths
    with pytest.raises(ValueError):
        BasePlotTest(x1[:-1], x2, c1, c2, t)  # Inconsistent x1 length

    with pytest.raises(ValueError):
        BasePlotTest(x1, x2[:-1], c1, c2, t)  # Inconsistent x2 length

    with pytest.raises(ValueError):
        BasePlotTest(x1, x2, c1[:, :-1], c2, t)  # Inconsistent c1 columns

    with pytest.raises(ValueError):
        BasePlotTest(x1, x2, c1, c2[:, :-1], t)  # Inconsistent c2 columns

    with pytest.raises(ValueError):
        BasePlotTest(x1, x2, c1.T, c2, t)  # Inconsistent c1 rows

    with pytest.raises(ValueError):
        BasePlotTest(x1, x2, c1, c2.T, t)  # Inconsistent c2 rows

    with pytest.raises(ValueError):
        BasePlotTest(x1, x2, c1, c2, t[:-1])  # Inconsistent t length


def test_baseplot_valid_input(mock_data):
    """Test valid input for BasePlot."""
    x1, x2, c1, c2, t = mock_data
    plot = BasePlotTest(x1, x2, c1, c2, t)
    assert plot.x1 is x1
    assert plot.x2 is x2
    assert plot.c1 is c1
    assert plot.c2 is c2
    assert plot.t is t


def test_specifiedtimeplot_calculate_time_index(mock_data):
    """Test calculate_time_index method of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data
    plot = SpecifiedTimePlot(x1, x2, c1, c2, t)

    # Test valid time fractions
    assert plot.calculate_time_index(0) == 0
    assert plot.calculate_time_index(1) == len(t) - 1
    assert plot.calculate_time_index(0.5) == len(t) // 2

    # Test invalid time fractions
    with pytest.raises(ValueError):
        plot.calculate_time_index(-0.1)
    with pytest.raises(ValueError):
        plot.calculate_time_index(1.1)


def test_specifiedtimeplot_initialization(mock_data):
    """Test initialization of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data

    # Test valid initialization
    try:
        SpecifiedTimePlot(x1, x2, c1, c2, t)
    except Exception as e:
        pytest.fail(f"SpecifiedTimePlot initialization failed: {str(e)}")

    # Test invalid initialization by not providing required arguments
    with pytest.raises(TypeError):
        SpecifiedTimePlot(x1, x2, c1, c2)  # Missing t argument


def test_show(mock_data, monkeypatch):
    """Test show method of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data
    plot = SpecifiedTimePlot(x1, x2, c1, c2, t)
    monkeypatch.setattr(plt, "show", lambda: None)

    # Test show method
    plot.plot_solution(0.5)  # No assertion, just ensure no errors occur
    plt.show()


def test_save(mock_data, tmp_path):
    """Test save method of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data
    plot = SpecifiedTimePlot(x1, x2, c1, c2, t)

    # Test save method
    filename = tmp_path / "plot.png"
    plot.save(str(filename), 0.5, dpi=100)

    # Assert the file is saved
    assert filename.is_file()


def test_animate_initialization(mock_data):
    """Test initialization of Animate."""
    x1, x2, c1, c2, t = mock_data

    # Test valid initialization
    try:
        Animate(x1, x2, c1, c2, t)
    except Exception as e:
        pytest.fail(f"Animate initialization failed: {str(e)}")


def test_show_animate(mock_data, monkeypatch):
    """Test show method of Animate."""
    x1, x2, c1, c2, t = mock_data
    anim = Animate(x1, x2, c1, c2, t)
    # Mock plt.show() to prevent the display of the plot during the test
    monkeypatch.setattr(plt, "show", lambda: None)

    # Test show method
    anim.initialize_plot()  # No assertion, just ensure no errors occur
    stored_animation = anim.anim

    # Simulate the animation process without rendering
    stored_animation._init_draw()
    stored_animation._step(0)
    stored_animation.new_frame_seq()

    # Ensure no errors occur during the simulated animation process
    assert True  # Placeholder assertion


def test_save_animate(mock_data, tmp_path):
    """Test save method of Animate."""
    x1, x2, c1, c2, t = mock_data
    anim = Animate(x1, x2, c1, c2, t)

    # Test save method
    filename = tmp_path / "animation.gif"
    anim.save(str(filename), dpi=100)

    # Assert the file is saved
    assert filename.is_file()
