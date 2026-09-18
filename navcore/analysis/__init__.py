"""Offline training analysis: metrics logging and post-hoc visualization.

Deliberately separate from `navcore.visualization`, which renders a *live*
`Environment` tick-by-tick via small `(data, ax)` sub-visualizers. Everything
here instead consumes artifacts produced *after* (or independently of) a
running simulation: a metrics log file, or a trained (eval-mode) policy
queried against a frozen scenario snapshot. Nothing in `navcore.training`
should ever import from here -- see `metrics_logger.py`'s module docstring
for why that boundary is enforced.
"""

from navcore.analysis.metrics_logger import TrainingMetricsLogger
from navcore.analysis.policy_field import PolicyField, PolicyFieldVisualizer
from navcore.analysis.training_curves import TrainingCurvePlotter

__all__ = [
    "PolicyField",
    "PolicyFieldVisualizer",
    "TrainingCurvePlotter",
    "TrainingMetricsLogger",
]
