"""CellVisualizer: renders a DecompositionResult's convex cells.

Sibling to ObstacleVisualizer/CrowdVisualizer/RobotVisualizer (same
constructor shape: ``(data, ax)`` plus a ``draw()`` method), but draws a
planning artifact -- ``boustropheden.decompose()``'s output -- rather
than anything read off the live ``Environment``. Kept as its own class,
not folded into ``ObstacleVisualizer``, since a decomposition is
computed once per episode by a separate module and is not part of
``Environment`` itself; ``Visualizer.visualize()`` only draws it when one
is explicitly passed in, the same way ``RobotVisualizer`` only draws a
mission overlay when a ``mission`` is passed in.
"""

from __future__ import annotations

from matplotlib.patches import Polygon as MplPolygon

from navcore.boustropheden.boustropheden import DecompositionResult


class CellVisualizer:
    """Draws each cell as a translucent, distinctly colored polygon.

    Attributes:
        result: The decomposition to draw.
        ax: The Matplotlib axes to draw onto.
    """

    #: Cycled by ``cell.id % len(_PALETTE)`` so adjacent ids usually (not
    #: guaranteed, for large cell counts) get visually distinct colors.
    #: Colorblind-friendly qualitative palette (ColorBrewer "Set3"-ish).
    _PALETTE = (
        "#a6cee3",
        "#b2df8a",
        "#fb9a99",
        "#fdbf6f",
        "#cab2d6",
        "#ffff99",
        "#8dd3c7",
        "#fccde5",
        "#bebada",
        "#ccebc5",
    )

    def __init__(self, result: DecompositionResult, ax) -> None:
        self.result = result
        self.ax = ax

    def _cell_centroid(
        self, vertices: list[tuple[float, float]]
    ) -> tuple[float, float]:
        n = len(vertices)
        return (
            sum(x for x, _ in vertices) / n,
            sum(y for _, y in vertices) / n,
        )

    def _draw_cells(self) -> None:
        for cell in self.result.cells:
            color = self._PALETTE[cell.id % len(self._PALETTE)]
            vertices = [(v.x, v.y) for v in cell.vertices]

            self.ax.add_patch(
                MplPolygon(
                    vertices,
                    closed=True,
                    fill=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.45,
                    zorder=-2,  # Behind obstacles, crowd, and the robot.
                )
            )

            cx, cy = self._cell_centroid(vertices)
            self.ax.text(
                cx,
                cy,
                f"C{cell.id}",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=-1,
            )

    def _draw_adjacency(self) -> None:
        """Draw a dotted line between adjacent cells' centroids.

        Purely a visual aid for eyeballing the traversal graph's
        connectivity -- not itself part of the decomposition/traversal
        math and safe to skip (``show_adjacency=False``) for a cleaner
        plot.
        """
        centroids = {
            cell.id: self._cell_centroid([(v.x, v.y) for v in cell.vertices])
            for cell in self.result.cells
        }
        for adjacency in self.result.adjacencies:
            if adjacency.cell_a not in centroids or adjacency.cell_b not in centroids:
                continue
            xa, ya = centroids[adjacency.cell_a]
            xb, yb = centroids[adjacency.cell_b]
            self.ax.plot(
                [xa, xb],
                [ya, yb],
                color="dimgray",
                linestyle=":",
                linewidth=1.0,
                zorder=-1,
            )

    def draw(self, show_adjacency: bool = True) -> None:
        """Draw every cell, and optionally the adjacency graph over them."""
        self._draw_cells()
        if show_adjacency:
            self._draw_adjacency()
