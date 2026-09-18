"""TrainingMetricsLogger: framework-agnostic scalar-metric logging for offline analysis.

Training and visualization must stay decoupled -- the same "rendering must
never influence simulation" principle generalized to "training must never
depend on plotting." This logger's only job is to append one JSON record per
call to a file on disk; it imports nothing from matplotlib, torch, or
numpy, and nothing in `navcore.training` should import matplotlib because of
this file. `TrainingCurvePlotter` (sibling module) is the only piece that
reads this file back and renders it.

JSONL (one JSON object per line), not a single top-level JSON array, so a
training run killed mid-write still leaves every prior record readable -- a
crash-safety property a single array does not have, and appending a line is
O(1) rather than rewriting the whole file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingMetricsLogger:
    """Appends one JSON-lines record per call to `path`.

    Attributes:
        path: Destination file. Parent directories are created on
            construction if missing.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Append one record: `step`, every key in `metrics`, and a wall-clock timestamp.

        Opens the file in append mode for exactly this one write rather
        than holding a handle open for the logger's lifetime -- costs one
        syscall per call (negligible; this is meant to be called once per
        PPO update or once per episode, never once per simulation tick),
        and means a crash mid-training never corrupts or loses records
        already written.

        Args:
            step: Monotonic training-progress counter (e.g. total env
                steps collected so far). Not required to be unique or
                strictly increasing -- `TrainingCurvePlotter` sorts by it
                before plotting, so an out-of-order call degrades
                gracefully rather than corrupting the log.
            metrics: Scalar values only, e.g.
                `{"mean_episode_reward": -3.2, "policy_loss": 0.14}`.
                Passing a non-JSON-serializable value (a tensor, an
                ndarray) raises `TypeError` -- by design; this is not a
                place to smuggle arrays through, cast to `float()` first.
        """
        record: dict[str, float] = {"step": step, "wall_time": time.time(), **metrics}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def read_all(path: Path) -> list[dict[str, float]]:
        """Read every record back, in file order (not necessarily `step` order).

        Returns:
            An empty list if `path` does not exist yet, rather than
            raising -- calling this before a single `log()` call has
            landed (e.g. from a plotting script racing a fresh training
            run) is a normal, not exceptional, case.
        """
        path = Path(path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
