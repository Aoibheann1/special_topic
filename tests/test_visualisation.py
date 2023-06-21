import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pde_package.visualisation import Visualisation
from pytest_mock import mocker


class TestVisualisation:
    @pytest.fixture
    def visualisation(self):
        x1 = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        x2 = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
        c1 = np.array([[1.0, 0.8, 0.6, 0.4, 0.2], [0.9, 0.7, 0.5, 0.3, 0.1]])
        c2 = np.array([[0.2, 0.4, 0.6, 0.8, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]])
        t = np.array([0.0, 1.0])
        return Visualisation(x1, x2, c1, c2, t)

    def test_visualisation_instance(self, visualisation):
        assert np.array_equal(visualisation.x1, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
        assert np.array_equal(visualisation.x2, np.array([1.0, 1.2, 1.4, 1.6, 1.8]))
        assert np.array_equal(
            visualisation.c1, np.array([[1.0, 0.8, 0.6, 0.4, 0.2], [0.9, 0.7, 0.5, 0.3, 0.1]])
        )
        assert np.array_equal(
            visualisation.c2, np.array([[0.2, 0.4, 0.6, 0.8, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]])
        )
        assert np.array_equal(visualisation.t, np.array([0.0, 1.0]))
        assert visualisation.fig is None
        assert visualisation.ax is None
        assert visualisation.line1 is None
        assert visualisation.line2 is None
        assert visualisation.anim is None

    def test_visualisation_initialise_plot(self, visualisation, mocker):
        mock_fig, mock_ax = mocker.patch.object(plt, "subplots"), mocker.Mock()
        mock_fig.return_value = (mock_fig, mock_ax)
        mock_line1, mock_line2 = mocker.Mock(), mocker.Mock()
        visualisation.ax = mock_ax  # Assigning a mock object to visualisation.ax
        mocker.patch.object(visualisation.ax, "plot", return_value=(mock_line1, mock_line2))
        mocker.patch.object(visualisation.ax, "set_xlabel")
        mocker.patch.object(visualisation.ax, "set_ylabel")
        mocker.patch.object(visualisation.ax, "set_xlim")
        mocker.patch.object(visualisation.ax, "set_ylim")
        mocker.patch.object(visualisation.ax, "set_title")
        mocker.patch.object(FuncAnimation, "__init__", return_value=None)

        visualisation.initialise_plot()

        mock_fig.assert_called_once_with()
        mock_ax.plot.assert_any_call([], [], 'bo', markersize=2, label='C1')
        mock_ax.plot.assert_any_call([], [], 'ro', markersize=2, label='C2')
        visualisation.ax.set_xlabel.assert_called_once_with('x')
        visualisation.ax.set_ylabel.assert_called_once_with('c')
        visualisation.ax.set_xlim.assert_called_once_with(0.0, 1.8)
        visualisation.ax.set_ylim.assert_called_once_with(0, 1)
        visualisation.ax.set_title.assert_called_with('Time = ')
        FuncAnimation.__init__.assert_called_once_with(
            mock_fig, visualisation.update_plot, frames=len(visualisation.t), interval=200
        )


    def test_visualisation_update_plot(self, visualisation):
        visualisation.line1 = mocker.Mock()
        visualisation.line2 = mocker.Mock()
        frame = 1

        visualisation.update_plot(frame)

        visualisation.line1.set_data.assert_called_once_with(
            visualisation.x1, visualisation.c1[:, frame]
        )
        visualisation.line2.set_data.assert_called_once_with(
            visualisation.x2, visualisation.c2[:, frame]
        )

    def test_visualisation_animate(self, visualisation, mocker):
        mocker.patch.object(plt, "show")

        visualisation.animate()

        plt.show.assert_called_once_with()

    def test_visualisation_show(self, visualisation, mocker):
        mocker.patch.object(plt, "isinteractive", return_value=False)
        mocker.patch.object(plt, "show")

        visualisation.show()

        plt.show.assert_called_once_with()
