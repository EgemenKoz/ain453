"""
viz.py – Rendering primitives for the Task 4 visualisation.

This module exposes a single ``Renderer`` class that draws every required
element on a matplotlib axis:

  * the global path produced by A*
  * the robot pose and its sensing disc
  * the obstacle (filled) and its inflated boundary (dashed)
  * all DWA candidate rollouts, with rejected ones in light grey
  * the chosen rollout, highlighted
  * the trace of executed positions so the map "updates" as the robot moves

The class is framework-agnostic — both the offline simulation driver and
the ROS visualiser node feed it the same ``Snapshot`` dataclass each frame.
The two static elements (workspace, obstacle, A* path) are drawn once and
then only the moving artists are updated, which keeps the animation
responsive on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from config_loader import Config
from costmap import CircleObstacle
from dwa import Rollout

Point = Tuple[float, float]
State = Tuple[float, float, float]


@dataclass
class Snapshot:
    """One frame of state to render."""
    state: State
    rollouts: List[Rollout]
    chosen: Optional[Rollout]
    trace: List[Point]
    step: int = 0
    info: str = ""
    obstacle: Optional["CircleObstacle"] = None  # Task 5: detected at runtime


class Renderer:
    """Draws the Task 4 view and updates incrementally each frame."""

    def __init__(
        self,
        cfg: Config,
        path: Sequence[Point],
        obstacle: Optional[CircleObstacle],
        ax: Optional[Axes] = None,
        show_rejected: Optional[bool] = None,
    ):
        self.cfg = cfg
        self.path = list(path)
        self.obstacle = obstacle
        self.show_rejected = (
            cfg.viz.show_rejected if show_rejected is None else show_rejected
        )

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
        else:
            self.ax = ax
            self.fig = ax.figure

        self._obstacle_patch: Optional[Circle] = None
        self._inflated_patch: Optional[Circle] = None
        self._truth_patch: Optional[Circle] = None

        self._draw_static()
        self._init_dynamic()
        if self.obstacle is not None:
            self.set_obstacle(self.obstacle)

    # ── construction ────────────────────────────────────────────────────────

    def _draw_static(self) -> None:
        ws = self.cfg.workspace
        ax = self.ax
        ax.set_xlim(ws.x_min, ws.x_max)
        ax.set_ylim(ws.y_min, ws.y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # A* path
        xs = [p[0] for p in self.path]
        ys = [p[1] for p in self.path]
        ax.plot(xs, ys, "-", color="#1f77b4", lw=2.0, label="A* path", zorder=2)

        # Start / goal
        ax.plot(self.cfg.start.x, self.cfg.start.y, "o",
                color="#2ca02c", ms=10, label="A (start)", zorder=4)
        ax.plot(self.cfg.goal.x, self.cfg.goal.y, "*",
                color="#d62728", ms=14, label="B (goal)", zorder=4)

        # Goal tolerance
        ax.add_patch(Circle(
            (self.cfg.goal.x, self.cfg.goal.y), self.cfg.goal.tol_m,
            fill=False, color="#d62728", linestyle=":", alpha=0.6, zorder=2,
        ))

    def _init_dynamic(self) -> None:
        ax = self.ax
        # Sensing disc — recoloured per frame
        self._sensing_patch = Circle(
            (0, 0), self.cfg.sensing.radius_m,
            fill=True, color="#2ca02c", alpha=0.08,
            ec="#2ca02c", linestyle="--", label="sensing area", zorder=1,
        )
        ax.add_patch(self._sensing_patch)

        # Robot footprint (radius)
        self._robot_patch = Circle(
            (0, 0), self.cfg.robot.radius_m,
            color="#9467bd", alpha=0.85, label="robot", zorder=5,
        )
        ax.add_patch(self._robot_patch)

        # Heading arrow — drawn as a Line2D for cheap updates
        self._heading_line, = ax.plot(
            [], [], "-", color="#9467bd", lw=2.0, zorder=5,
        )

        # Rollouts — pre-allocate Line2D objects up to v_samples * w_samples
        max_rollouts = self.cfg.dwa.v_samples * self.cfg.dwa.w_samples
        self._rollout_lines: List[Line2D] = []
        for _ in range(max_rollouts):
            ln, = ax.plot([], [], "-", lw=0.6, alpha=0.6, zorder=2)
            self._rollout_lines.append(ln)

        # Chosen rollout
        self._chosen_line, = ax.plot(
            [], [], "-", color="#ff7f0e", lw=2.5,
            label="chosen rollout", zorder=6,
        )

        # Executed trajectory
        self._trace_line, = ax.plot(
            [], [], "-", color="#2ca02c", lw=1.6, alpha=0.9,
            label="executed", zorder=4,
        )

        # Info text
        self._info_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # ── obstacle (re)attachment ────────────────────────────────────────────

    def set_obstacle(self, obstacle: CircleObstacle) -> None:
        """Add or replace the obstacle drawing.

        Used in Task 5 bonus, where the obstacle only appears after the ToF
        sensor detects it.
        """
        self.obstacle = obstacle
        if self._obstacle_patch is not None:
            self._obstacle_patch.remove()
        if self._inflated_patch is not None:
            self._inflated_patch.remove()
        self._obstacle_patch = Circle(
            (obstacle.cx, obstacle.cy), obstacle.radius,
            color="black", alpha=0.85, label="obstacle", zorder=3,
        )
        self._inflated_patch = Circle(
            (obstacle.cx, obstacle.cy), obstacle.inflated_radius,
            fill=False, color="black", linestyle="--",
            label="inflated boundary", zorder=3,
        )
        self.ax.add_patch(self._obstacle_patch)
        self.ax.add_patch(self._inflated_patch)
        self.ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def set_truth_obstacle(self, obstacle: CircleObstacle) -> None:
        """Always-visible reference circle for the (test scenario) truth.

        Drawn as a thin grey dashed outline so the user can see where the
        obstacle is even before the ToF detection has fired.
        """
        if self._truth_patch is not None:
            self._truth_patch.remove()
        self._truth_patch = Circle(
            (obstacle.cx, obstacle.cy), obstacle.radius,
            fill=False, color="#7f7f7f", linestyle=":", lw=1.2, alpha=0.8,
            label="obstacle (truth)", zorder=2,
        )
        self.ax.add_patch(self._truth_patch)
        self.ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # ── per-frame update ───────────────────────────────────────────────────

    def update(self, snap: Snapshot) -> None:
        # Attach the obstacle the first time a snapshot carries one
        # (Task 5: shows the obstacle "appearing" at detection time).
        if snap.obstacle is not None and self._obstacle_patch is None:
            self.set_obstacle(snap.obstacle)
        x, y, th = snap.state

        # Robot + sensing
        self._sensing_patch.center = (x, y)
        self._robot_patch.center = (x, y)
        # Heading marker: short line from centre toward heading direction
        r = max(self.cfg.robot.radius_m * 1.6, 0.04)
        import math
        self._heading_line.set_data([x, x + r * math.cos(th)],
                                    [y, y + r * math.sin(th)])

        # Rollouts (rejected → light grey, accepted → soft blue)
        for ln, ro in zip(self._rollout_lines, snap.rollouts):
            if ro.rejected and not self.show_rejected:
                ln.set_data([], [])
                continue
            colour = "#cccccc" if ro.rejected else "#aec7e8"
            ln.set_color(colour)
            ln.set_data(ro.xs, ro.ys)
        # Hide unused line objects (in case we have extras)
        for ln in self._rollout_lines[len(snap.rollouts):]:
            ln.set_data([], [])

        # Chosen
        if snap.chosen is not None:
            self._chosen_line.set_data(snap.chosen.xs, snap.chosen.ys)
        else:
            self._chosen_line.set_data([], [])

        # Trace
        if snap.trace:
            tx = [p[0] for p in snap.trace]
            ty = [p[1] for p in snap.trace]
            self._trace_line.set_data(tx, ty)

        # Info
        v = snap.chosen.v if snap.chosen else 0.0
        w = snap.chosen.w if snap.chosen else 0.0
        self._info_text.set_text(
            f"step {snap.step}\n"
            f"pose ({x:.2f}, {y:.2f}, {math.degrees(th):.0f}°)\n"
            f"cmd  v={v:.2f} m/s, ω={w:+.2f} rad/s\n"
            f"{snap.info}"
        )

    # ── convenience ─────────────────────────────────────────────────────────

    def save(self, path: str, dpi: int = 120) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ── Standalone demo: closed-loop sim animated frame-by-frame ────────────────

def _run_demo(out_path: Optional[str] = None,
              max_steps: int = 200) -> None:
    if out_path is None:
        from pathlib import Path
        out_path = str(Path(__file__).resolve().parents[1] / "output" / "run.gif")
    """Run the full sim with the renderer and save an animated GIF."""
    import math
    import matplotlib.animation as animation

    from astar_grid import plan as astar_plan
    from config_loader import default_path, load
    from dwa import DWAPlanner

    cfg = load(default_path())
    waypoints = astar_plan(cfg)
    obstacle = CircleObstacle.from_config(cfg)
    planner = DWAPlanner(cfg, obstacle, waypoints)

    state: State = (cfg.start.x, cfg.start.y, cfg.start.theta)
    trace: List[Point] = [(state[0], state[1])]
    snapshots: List[Snapshot] = []

    for step in range(max_steps):
        result = planner.step(state)
        snapshots.append(Snapshot(
            state=state,
            rollouts=list(result.rollouts),
            chosen=result.chosen,
            trace=list(trace),
            step=step,
            info="",
        ))
        if math.hypot(state[0] - cfg.goal.x, state[1] - cfg.goal.y) <= cfg.goal.tol_m:
            snapshots[-1].info = "*** goal reached ***"
            break
        if result.chosen is None or result.chosen.rejected:
            snapshots[-1].info = "no feasible move"
            break
        v, w = result.chosen.v, result.chosen.w
        dt = cfg.dwa.dt
        th = state[2] + w * dt
        x = state[0] + v * math.cos(th) * dt
        y = state[1] + v * math.sin(th) * dt
        state = (x, y, th)
        trace.append((x, y))

    print(f"snapshots: {len(snapshots)}")

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle("Assignment 4 – A* + DWA local planner", fontsize=11)
    renderer = Renderer(cfg, waypoints, obstacle, ax=ax)

    def update(i: int):
        renderer.update(snapshots[i])
        return ()

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=80, blit=False,
    )
    try:
        anim.save(out_path, writer="pillow", fps=12)
        print(f"animation saved: {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"animation save failed ({exc}); saving final frame instead")
        renderer.update(snapshots[-1])
        png = out_path.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"png saved: {png}")


if __name__ == "__main__":
    _run_demo()
