"""
INFORM selectivity optimization package.

Provides the selectivity metrics and the PSO-based stimulation-protocol
optimization used to evaluate INFORM's inferred organizations.

Metric roles (what the code does):
- selectivity_opt  : maximized during optimization (margin form)
- selectivity_eval : used to evaluate/visualize results (squared-ratio form)
"""

from .selectivity import (
    selectivity_opt,
    selectivity_eval,
    params_to_selectivity_rasp,
    run_pso_selectivity,
    plot_matrix,
    plot_color_section,
    off_diagonal_frobenius_norm,
    DEFAULT_MATRIX_COLORS,
)

__all__ = [
    "selectivity_opt",
    "selectivity_eval",
    "params_to_selectivity_rasp",
    "run_pso_selectivity",
    "plot_matrix",
    "plot_color_section",
    "off_diagonal_frobenius_norm",
    "DEFAULT_MATRIX_COLORS",
]
