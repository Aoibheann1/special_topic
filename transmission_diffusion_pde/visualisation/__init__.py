"""Module for visualisation of the diffusion process.

This subpackage provides classes for visualising the diffusion process of a
transmission diffusion PDE problem.

Modules:
- base.py: module defining the abstract base class for visualisation.
- animate.py: module defining the class for animating the diffusion process.
- plot.py: module containing the class for creating static plots of the
           diffusion process.
"""

from .base import BasePlot # noqa F401
from .plot import SpecifiedTimePlot # noqa F401
from .animate import Animate # noqa F401
