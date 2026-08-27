"""Renderer: the pure-observation drawing interface.

A ``Renderer`` consumes ``RenderFrame`` snapshots and draws them --
nothing more. It must never be able to influence simulation state; the
signatures here are deliberately one-way (frame in, nothing out) to
make that boundary mechanically obvious, not just a convention.

``Simulation`` treats its renderer as fully optional: with
``renderer=None`` (the default for training), no frame is ever built
and this protocol is never touched, so headless runs pay zero
rendering cost.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from navcore.rendering.render_frame import RenderFrame


@runtime_checkable
class Renderer(Protocol):
    """Draws simulation state. Never reads simulation state back."""

    def render(self, frame: RenderFrame) -> None:
        """Draw ``frame``. Called at most once per simulation tick."""
        ...

    def reset(self, episode: int) -> None:
        """Handle an episode boundary (e.g. clear trails, reset camera).

        Distinct from :meth:`close`: this fires every episode, close
        fires once, at full teardown.
        """
        ...

    def close(self) -> None:
        """Release any resources held by this renderer (windows, files, ...)."""
        ...
