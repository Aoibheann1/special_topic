"""Solver subpackage.

This subpackage contains modules related to solving transmission diffusion
PDEs.

Modules:
- base.py: module defining the abstract base class for PDE solvers.
- method_of_lines.py: module implementing the Method of Lines
approach for PDE solving.

"""

from .base import BaseSolver # noqa F401
from .method_of_lines import MethodOfLines # noqa F401
