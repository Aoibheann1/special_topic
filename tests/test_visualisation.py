"""
Pytest file for testing the visualization subpackage.

This pytest file contains test cases to ensure the proper functionality of the
visualization subpackage. The subpackage provides classes for visualizing the
solutions of the transmission diffusion PDE problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from transmission_diffusion_pde import (
    BasePlot,
    SpecifiedTimePlot,
    Animate
)


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


def test_baseplot_initialisation(mock_data):
    """Test initialisation of BasePlot."""
    x1, x2, c1, c2, t = mock_data

    try:
        BasePlotTest(x1, x2, c1, c2, t)
    except Exception as e:
        pytest.fail(f"BasePlot initialization failed: {str(e)}")

    # Test invalid initialisation with inconsistent lengths
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


def test_specifiedtimeplot_initialisation(mock_data):
    """Test initialisation of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data

    # Test valid initialisation
    try:
        SpecifiedTimePlot(x1, x2, c1, c2, t)
    except Exception as e:
        pytest.fail(f"SpecifiedTimePlot initialisation failed: {str(e)}")

    # Test invalid initialisation by not providing required arguments
    with pytest.raises(TypeError):
        SpecifiedTimePlot(x1, x2, c1, c2)  # Missing t argument


def test_show(mock_data, monkeypatch):
    """Test show method of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data
    plot = SpecifiedTimePlot(x1, x2, c1, c2, t)
    monkeypatch.setattr(plt, "show", lambda: None)

    plot.plot_solution(0.5)
    plt.show()


def test_save(mock_data, tmp_path):
    """Test save method of SpecifiedTimePlot."""
    x1, x2, c1, c2, t = mock_data
    plot = SpecifiedTimePlot(x1, x2, c1, c2, t)

    filename = tmp_path / "plot.png"
    plot.save(str(filename), 0.5, dpi=100)

    assert filename.is_file()


def test_animate_initialisation(mock_data):
    """Test initialisation of Animate."""
    x1, x2, c1, c2, t = mock_data

    try:
        Animate(x1, x2, c1, c2, t)
    except Exception as e:
        pytest.fail(f"Animate initialisation failed: {str(e)}")


def test_show_animate(mock_data):
    """Test show method of Animate."""
    x1, x2, c1, c2, t = mock_data
    anim = Animate(x1, x2, c1, c2, t)

    anim.animate_solution()
    stored_animation = anim.anim

    # Simulate the animation process without rendering
    stored_animation._init_draw()
    stored_animation._step(0)
    stored_animation.new_frame_seq()


def test_save_animate(mock_data, tmp_path):
    """Test save method of Animate."""
    x1, x2, c1, c2, t = mock_data
    anim = Animate(x1, x2, c1, c2, t)

    filename = tmp_path / "animation.gif"
    anim.save(str(filename), dpi=100)

    assert filename.is_file()
