# sim3D - 3D Reservoir Simulation (Black-Oil Model)

A simple 3D reservoir simulation tool for modeling multi-phase fluid flow in porous media.

## Features

- **Three-Phase Flow Simulation**: Water, oil, and gas phase modeling with relative permeability, gravity, and capillary pressure effects
- **Advanced Numerical Methods**:
  - Explicit and implicit finite difference schemes
  - Adaptive time stepping
- **Complex Boundary Conditions**: Flexible boundary condition system supporting various reservoir geometries
- **Well Modeling**: Injection and production wells with proper phase mobility weighting
- **Visualization**: Comprehensive plotting and analysis tools with plotly

## Project Structure

```
sim3D/
├── sim3D/                  # Main simulation package
│   ├── __init__.py
│   ├── boundaries.py       # Boundary conditions
│   ├── constants.py        # Physical constants
│   ├── dynamic.py          # Dynamic properties
│   ├── factories.py        # Object factories
│   ├── flow.py            # Flow equations and solvers
│   ├── grids.py           # Grid generation and utilities
│   ├── properties.py      # Rock and fluid properties
│   ├── relperm.py         # Relative permeability models
│   ├── static.py          # Static properties
│   ├── types.py           # Type definitions
│   ├── wells.py           # Well modeling
│   └── visualization/     # Plotting and visualization
├── main.py                # Command-line interface
├── main_marimo.py         # Interactive Marimo notebook
└── pyproject.toml         # Project configuration
```

## Setup

### Prerequisites

- Python 3.10 or higher
- uv (recommended) or pip for dependency management

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd sim3D
   ```

2. **Install dependencies using uv (recommended):**

   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install dependencies
   uv sync --extra dev
   ```

   **Or using pip:**

   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -e .
   ```

### Dependencies

The project uses several key dependencies:

- `uv` - Dependency management and virtual environments
- `CoolProp` - Thermophysical properties of fluids
- `numpy` - Numerical computations
- `scipy` - Sparse matrix operations and linear solvers
- `attrs` - Data classes and serialization
- `marimo` - Interactive notebooks
- `plotly` - Visualization

## Usage

### Command Line Interface

Run basic simulations using the command-line interface:

```bash
# Activate virtual environment if using pip
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run simulation
python main.py
```

### Interactive Development with Marimo

For interactive development and experimentation, use the Marimo notebook:

1. **Install Marimo** (if not already installed):

   ```bash
   uv add marimo  # or pip install marimo
   ```

2. **Run in edit mode:**

   ```bash
   # Using uv
   uv run marimo edit main_marimo.py
   
   # Or if using pip (with activated environment)
   marimo edit main_marimo.py
   ```

3. **Open in browser:**
   - Marimo will automatically open your default browser
   - Navigate to the displayed localhost URL (typically `http://localhost:2718`)
   - The notebook provides an interactive environment for:
     - Setting up simulation parameters
     - Running flow calculations
     - Visualizing results
     - Experimenting with different scenarios
  
## Examples

Check `main.py` and `main_marimo.py` for an example simulation and usage pattern.

## Development

### Code Structure

- **Flow Equations** (`flow.py`): Core numerical methods for pressure and saturation evolution
- **Grid Management** (`grids.py`): Grid creation and manipulation utilities
- **Static Models** (`static.py`): Static property definitions
- **Dynamic Simulations** (`dynamic.py`): Run simulations over time
- **Properties** (`properties.py`): Rock and fluid property definitions
- **Factories** (`factories.py`): Model creation factories
- **Constants** (`constants.py`): Physical constants and units
- **Boundaries** (`boundaries.py`): Boundary condition system
- **Wells** (`wells.py`): Injection and production well modeling
- **Relative Permeability** (`relperm.py`): Models for relative permeability and capillary pressure
- **Visualization** (`visualization/`): Plotting and analysis tools

### Numerical Methods

The simulator implements several numerical schemes:

- **Explicit Methods**: Fast, conditionally stable
- **Implicit Methods**: Unconditionally stable, suitable for large time steps
- **Adaptive Methods**: Automatically switch between explicit/implicit based on stability criteria
