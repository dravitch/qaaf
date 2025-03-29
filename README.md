# QAAF - Quantitative Algorithmic Asset Framework

## Overview

QAAF (Quantitative Algorithmic Asset Framework) is an advanced algorithmic framework designed for optimizing asset allocation between pairs of complementary assets, with a particular focus on PAXG/BTC. Unlike approaches that attempt to predict future prices, QAAF identifies optimal rebalancing moments by analyzing the fundamental properties of assets and their intrinsic relationships.

## Key Features

- **Metric-Based Analysis**: Utilizes four fundamental metrics to evaluate asset relationships
- **Adaptive Allocation**: Dynamically adjusts allocations based on market phases
- **Profile-Based Optimization**: Offers multiple optimization profiles for different investment objectives
- **Transaction Fee Optimization**: Minimizes the impact of transaction costs
- **Robust Validation**: Includes out-of-sample validation and stress testing

## Architecture

QAAF is built with a modular architecture:

- `metrics/`: Implementation of the four fundamental metrics
- `market/`: Market phase analysis and detection
- `allocation/`: Adaptive allocation algorithms
- `transaction/`: Transaction fee evaluation and optimization
- `optimization/`: Parameter optimization (grid search and Bayesian)
- `validation/`: Out-of-sample validation and robustness testing
- `data/`: Data management and processing
- `utils/`: Visualization and utility functions
- `core/`: Core orchestration components

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- CuPy (optional, for GPU acceleration)
- scikit-learn (for Bayesian optimization)
- yfinance (for data acquisition)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qaaf-project.git
cd qaaf-project

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
from qaaf.core.qaaf_core import QAAFCore

# Initialize QAAF
qaaf = QAAFCore(
    initial_capital=30000.0,
    trading_costs=0.001,
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Run full analysis with the balanced profile
results = qaaf.run_full_analysis(
    optimize_metrics=True,
    optimize_threshold=True,
    run_validation=True,
    profile='balanced'
)

# Print summary of results
qaaf.print_summary()

# Visualize results
qaaf.visualize_results()
```

## Documentation
Detailed documentation is available in the docs/ directory:

- Architecture overview
- Metrics description
- Optimization profiles
- Validation methodology
- Performance benchmarks

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT

## Citation
If you use QAAF in your research or project, please cite:
@software{qaaf2024,
  author = dravitch, Claude AI
  title = {QAAF: Quantitative Algorithmic Asset Framework},
  year = {2025},
  url = {https://github.com/dravitch/qaaf-project}
}