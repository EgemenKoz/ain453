"""
astar_grid.py – 8-connected A* on a metric occupancy grid.

The workspace (in metres) is discretised into square cells of side
``resolution_m``. A* searches from the cell containing point ``A`` to the
cell containing point ``B``. The result is a list of (x, y) waypoints in
world coordinates, optionally densified by linear interpolation so the
local planner has dense samples to track.

This module is independent of ROS and can be exercised from the CLI:

    python3 astar_grid.py
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config_loader import Config, default_path, load

Cell = Tuple[int, int]      # (ix, iy) indices into the grid
Point = Tuple[float, float] # (x, y) world coordinates in metres


# ── Grid construction ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Grid:
    """An axis-aligned occupancy grid covering the workspace."""

    res: float
    x_min: float
    y_min: float
    nx: int
    ny: int
    blocked: np.ndarray  # shape (ny, nx); True means cell is non-traversable

    @classmethod
    def empty(cls, cfg: Config) -> "Grid":
        ws = cfg.workspace
        res = cfg.grid.resolution_m
        nx = max(1, int(math.ceil(ws.width / res)))
        ny = max(1, int(math.ceil(ws.height / res)))
        return cls(
            res=res,
            x_min=ws.x_min,
            y_min=ws.y_min,
            nx=nx,
            ny=ny,
            blocked=np.zeros((ny, nx), dtype=bool),
        )

    def world_to_cell(self, x: float, y: float) -> Cell:
        ix = int((x - self.x_min) / self.res)
        iy = int((y - self.y_min) / self.res)
        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        return (ix, iy)

    def cell_to_world(self, c: Cell) -> Point:
        ix, iy = c
        return (
            self.x_min + (ix + 0.5) * self.res,
            self.y_min + (iy + 0.5) * self.res,
        )

    def in_bounds(self, c: Cell) -> bool:
        ix, iy = c
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def is_free(self, c: Cell) -> bool:
        return self.in_bounds(c) and not self.blocked[c[1], c[0]]


# ── Heuristic ───────────────────────────────────────────────────────────────

def _octile(a: Cell, b: Cell) -> float:
    """Octile distance — admissible/consistent for 8-connected grids."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


# ── Heap entry ──────────────────────────────────────────────────────────────

@dataclass(order=True)
class _Entry:
    f: float
    h: float
    cell: Cell = field(compare=False)


# ── A* search ───────────────────────────────────────────────────────────────

def _neighbours(c: Cell, allow_diag: bool) -> List[Tuple[Cell, float]]:
    ix, iy = c
    out: List[Tuple[Cell, float]] = [
        ((ix + 1, iy),     1.0),
        ((ix - 1, iy),     1.0),
        ((ix,     iy + 1), 1.0),
        ((ix,     iy - 1), 1.0),
    ]
    if allow_diag:
        s = math.sqrt(2)
        out.extend([
            ((ix + 1, iy + 1), s),
            ((ix + 1, iy - 1), s),
            ((ix - 1, iy + 1), s),
            ((ix - 1, iy - 1), s),
        ])
    return out


def _reconstruct(parents: dict, end: Cell) -> List[Cell]:
    path = [end]
    while parents.get(path[-1]) is not None:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def search(grid: Grid, start: Cell, goal: Cell, allow_diag: bool) -> List[Cell]:
    """Return the cell sequence from *start* to *goal*. Raises if no path."""
    if not grid.is_free(start):
        raise ValueError(f"start cell {start} is not free")
    if not grid.is_free(goal):
        raise ValueError(f"goal cell {goal} is not free")

    g: dict = {start: 0.0}
    parent: dict = {start: None}
    h0 = _octile(start, goal)
    open_heap: List[_Entry] = [_Entry(f=h0, h=h0, cell=start)]
    closed: set = set()

    while open_heap:
        entry = heapq.heappop(open_heap)
        cell = entry.cell
        if cell in closed:
            continue
        if cell == goal:
            return _reconstruct(parent, goal)
        closed.add(cell)
        for nb, step_cost in _neighbours(cell, allow_diag):
            if not grid.is_free(nb) or nb in closed:
                continue
            tentative = g[cell] + step_cost
            if tentative < g.get(nb, math.inf):
                g[nb] = tentative
                parent[nb] = cell
                h = _octile(nb, goal)
                heapq.heappush(open_heap, _Entry(f=tentative + h, h=h, cell=nb))

    raise RuntimeError(f"A*: no path from {start} to {goal}")


# ── Densifier ───────────────────────────────────────────────────────────────

def densify(waypoints: List[Point], step_m: float) -> List[Point]:
    """Linearly interpolate between waypoints to enforce a max spacing."""
    if step_m <= 0 or len(waypoints) < 2:
        return list(waypoints)
    dense: List[Point] = [waypoints[0]]
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        n = max(1, int(math.ceil(seg / step_m)))
        for k in range(1, n + 1):
            t = k / n
            dense.append((x0 + t * dx, y0 + t * dy))
    return dense


# ── End-to-end planning ─────────────────────────────────────────────────────

def plan(cfg: Config, grid: Optional[Grid] = None) -> List[Point]:
    """
    Run A* using configuration *cfg*. The grid is, by default, empty (the
    assignment specifies the obstacle is *not* part of the global plan), but
    a custom Grid may be supplied for experimentation.

    Returns a list of (x, y) world-coordinate waypoints, densified to
    ``cfg.path.densify_step_m`` spacing.
    """
    grid = grid if grid is not None else Grid.empty(cfg)
    start = grid.world_to_cell(cfg.start.x, cfg.start.y)
    goal = grid.world_to_cell(cfg.goal.x, cfg.goal.y)
    cells = search(grid, start, goal, allow_diag=cfg.grid.allow_diagonal)
    waypoints = [grid.cell_to_world(c) for c in cells]
    return densify(waypoints, cfg.path.densify_step_m)


def path_length(waypoints: List[Point]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )


# ── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load(default_path())
    grid = Grid.empty(cfg)
    waypoints = plan(cfg, grid)
    print(f"grid          : {grid.nx} x {grid.ny} cells (res={grid.res} m)")
    print(f"waypoints     : {len(waypoints)} (first={waypoints[0]}, last={waypoints[-1]})")
    print(f"path length   : {path_length(waypoints):.3f} m")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("matplotlib not available — skipping plot")
    else:
        import matplotlib.pyplot as plt
        xs = [p[0] for p in waypoints]
        ys = [p[1] for p in waypoints]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(cfg.workspace.x_min, cfg.workspace.x_max)
        ax.set_ylim(cfg.workspace.y_min, cfg.workspace.y_max)
        ax.set_aspect("equal")
        ax.set_title("A* global path (empty grid)")
        ax.plot(xs, ys, "-b", lw=2, label="A* path")
        ax.plot(cfg.start.x, cfg.start.y, "go", ms=10, label="A")
        ax.plot(cfg.goal.x, cfg.goal.y, "r*", ms=14, label="B")
        circ = plt.Circle(
            (cfg.obstacle.x, cfg.obstacle.y),
            cfg.obstacle.radius_m,
            fill=False, color="k", label="obstacle (display only)",
        )
        ax.add_patch(circ)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        out = str(Path(__file__).resolve().parents[1] / "output" / "astar.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"plot saved    : {out}")
