"""Recording helpers layered on top of ``matplotlib.animation``.

Kept separate from ``Visualizer`` so export format concerns (writer
choice, DPI, codec availability) don't clutter the main render loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Recorder:
    def __init__(self, fig: Any) -> None:
        self.fig = fig

    def save_animation(self, anim: Any, path: str | Path, *, fps: int = 20, dpi: int = 150) -> None:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            self._save_gif(anim, path, fps=fps, dpi=dpi)
        elif suffix in (".mp4", ".m4v", ".mov"):
            self._save_mp4(anim, path, fps=fps, dpi=dpi)
        else:
            raise ValueError(f"Unsupported animation export format: {suffix!r}. Use .gif or .mp4.")

    def _save_gif(self, anim: Any, path: Path, *, fps: int, dpi: int) -> None:
        from matplotlib.animation import PillowWriter

        anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)

    def _save_mp4(self, anim: Any, path: Path, *, fps: int, dpi: int) -> None:
        from matplotlib.animation import FFMpegWriter, writers

        if "ffmpeg" not in writers.list():
            raise RuntimeError(
                "MP4 export requires ffmpeg to be installed and on PATH. "
                "Install it, or export as .gif instead."
            )
        anim.save(str(path), writer=FFMpegWriter(fps=fps), dpi=dpi)

    def save_png_sequence(self, render_frame_fn, n_frames: int, out_dir: str | Path, *, dpi: int = 150,
                           name_pattern: str = "frame_{:05d}.png") -> list[Path]:
        """Step through ``n_frames`` frames, saving each as a PNG.

        ``render_frame_fn(i)`` should advance the sim and redraw the
        figure for frame ``i``; this method only handles the file I/O.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for i in range(n_frames):
            render_frame_fn(i)
            out_path = out_dir / name_pattern.format(i)
            self.fig.savefig(out_path, dpi=dpi)
            saved.append(out_path)
        return saved
