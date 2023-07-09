# Transmission Diffusion PDE Package

The Transmission Diffusion PDE package is a Python package that provides functionality for solving and visualising transmission diffusion PDE problems. It offers a modular and extensible architecture to handle different boundary conditions and solution methods.

## Installation

To install the package, you can use pip:

```
pip install transmission_diffusion_pde
```

## Features

The Transmission Diffusion PDE package offers the following features:

- Definition and application of various boundary conditions, including Dirichlet and Neumann boundary conditions.
- Solution of transmission diffusion PDE problems using the Method of Lines approach.
- Visualisation of the diffusion process through static plots and animated plots.

## Usage

To use the package, you can import the necessary modules and classes:

```python
from transmission_diffusion_pde import MethodOfLines, SpecifiedTimePlot
```

You can then create a solver instance, and solve the PDE problem:

```python
solver = MethodOfLines(...)
x1, x2, c1, c2, t = solver.solve_pde_system()
```

Finally, you can visualise the results using the SpecifiedTimePlot class:

```python
plot = SpecifiedTimePlot(x1, x2, c1, c2, t)
plot.show()
```

Please refer to the package documentation for detailed usage instructions and examples.
