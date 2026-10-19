"""Analysis module - Sensitivity analysis and visualization."""

from .sensitivity import (
    run_sensitivity_analysis,
    sensitivity_table_from_counts,
    plot_sensitivity_metric,
)

__all__ = ['run_sensitivity_analysis', 'sensitivity_table_from_counts', 'plot_sensitivity_metric']
