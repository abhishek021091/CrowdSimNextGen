"""TrainingCurvePlotter: offline rendering of logged training metrics.

Reads a JSONL file written by `TrainingMetricsLogger`. Has no dependency on
any live `Environment`, `Step`, or policy instance -- this is purely "file on
disk in, matplotlib Figure out," so it can be run from a separate script or
notebook, long after (or during, if the log is being tailed) a training run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from navcore.analysis.metrics_logger import TrainingMetricsLogger

#: Metrics plotted by default, one subplot each, in this order, when present.
#: Matches the diagnostic names `CrowdNavPPTrainer.update()` /
#: `collect_rollout()` already compute (see ppo_trainer.py) -- a metric
#: absent from every record (e.g. a non-PPO trainer that only logs
#: mean_episode_reward) is skipped rather than drawn as an empty axes.
_DEFAULT_METRICS: tuple[str, ...] = (
    "mean_episode_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
)


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    """Return a trailing moving average, the same length as `values`.

    Early points average over however many samples exist so far (fewer
    than `window`), rather than being dropped -- a smoothed curve should
    have the same x-extent as the raw one it overlays, not a shorter one
    starting `window` steps late.
    """
    if window <= 1:
        return list(values)
    out: list[float] = []
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += v
        if i >= window:
            running_sum -= values[i - window]
        out.append(running_sum / min(i + 1, window))
    return out


class TrainingCurvePlotter:
    """Renders reward/loss/diagnostic curves from a `TrainingMetricsLogger` file.

    Attributes:
        metrics_path: JSONL file to read.
        smoothing_window: Trailing moving-average window, in units of
            logged *records* -- typically one record per PPO update
            (`total_updates`), not per environment step, so a window of
            10 means "the last 10 updates," whatever `n_steps * n_envs`
            each of those represents.
    """

    def __init__(self, metrics_path: Path, smoothing_window: int = 10) -> None:
        self.metrics_path = Path(metrics_path)
        self.smoothing_window = smoothing_window

    def plot(
        self,
        metric_names: Sequence[str] | None = None,
        save_path: Path | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render one subplot per requested metric that actually appears in the log.

        Args:
            metric_names: Which logged keys to plot, in order. Defaults to
                `_DEFAULT_METRICS`.
            save_path: If given, the figure is also saved here (format
                inferred from the file extension, e.g. `.png`/`.pdf`).
            show: If True, calls `plt.show()`. Leave False for
                script/notebook use where the caller controls display --
                this class never assumes it owns the event loop.

        Returns:
            The created `Figure`, for further customization beyond what
            `save_path` covers.

        Raises:
            ValueError: If the log file has no records, or none of the
                requested metric names appear in any record.
        """
        records = TrainingMetricsLogger.read_all(self.metrics_path)
        if not records:
            raise ValueError(f"No records found in {self.metrics_path}.")
        records.sort(key=lambda r: r.get("step", 0))

        names = metric_names if metric_names is not None else _DEFAULT_METRICS
        present = [name for name in names if any(name in r for r in records)]
        if not present:
            raise ValueError(
                f"None of {list(names)} appear in any record in {self.metrics_path}."
            )

        fig, axes = plt.subplots(
            len(present), 1, figsize=(9, 3 * len(present)), sharex=True
        )
        axes = [axes] if len(present) == 1 else list(axes)

        for ax, name in zip(axes, present):
            xs = [r.get("step", i) for i, r in enumerate(records) if name in r]
            ys = [r[name] for r in records if name in r]
            smoothed = _moving_average(ys, self.smoothing_window)

            ax.plot(xs, ys, color="steelblue", alpha=0.3, linewidth=1, label="raw")
            ax.plot(xs, smoothed, color="steelblue", linewidth=2, label="smoothed")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        axes[-1].set_xlabel("training step")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()

        return fig
