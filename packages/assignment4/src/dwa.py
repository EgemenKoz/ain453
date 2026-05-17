"""
dwa.py – Dynamic Window Approach local planner (simplified).

Pipeline (one control step):

  1. v is fixed at ``v_max`` (constant velocity – DWA samples ω only).
  2. Build the ω window as ``[last_w − band, last_w + band]`` clamped to the
     static ω limits, where ``band = alpha_max · window_dt``. This rate-limits
     ω between control steps so the noisy real robot doesn't jerk.
  3. For each ω sample, simulate a unicycle rollout for ``horizon_s`` seconds
     in steps of ``dt``.
  4. Score each rollout with: distance to A* path + distance to goal +
     obstacle clearance penalty. Rollouts that enter the inflated obstacle
     are hard-rejected.
  5. Return the lowest-cost rollout.

Cost terms are kept to the minimum required by the assignment (path, goal,
obstacle). Heading, velocity-reward, wall, and out-of-bounds terms were
removed because pose drift on the real robot makes them unreliable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config_loader import Config
from costmap import CircleObstacle

Point = Tuple[float, float]
State = Tuple[float, float, float]  # (x, y, theta)


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Rollout:
    v: float
    w: float
    xs: np.ndarray         # shape (H+1,)
    ys: np.ndarray         # shape (H+1,)
    cost: float = math.inf
    rejected: bool = False
    reason: str = ""       # short tag for debugging / display


@dataclass
class DWAResult:
    chosen: Optional[Rollout]
    rollouts: List[Rollout] = field(default_factory=list)
    window: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # v_lo,v_hi,w_lo,w_hi


# ── Rollout simulation ──────────────────────────────────────────────────────

def _rollout(state: State, v: float, w: float, horizon_s: float, dt: float
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-Euler unicycle rollout. Returns (xs, ys) including the start."""
    n = max(1, int(round(horizon_s / dt)))
    x, y, th = state
    xs = np.empty(n + 1)
    ys = np.empty(n + 1)
    xs[0] = x
    ys[0] = y
    for k in range(1, n + 1):
        th += w * dt
        x  += v * math.cos(th) * dt
        y  += v * math.sin(th) * dt
        xs[k] = x
        ys[k] = y
    return xs, ys


# ── Cost terms ──────────────────────────────────────────────────────────────

def _path_distance_term(xs: np.ndarray, ys: np.ndarray,
                        path_xy: np.ndarray) -> float:
    """Mean nearest-waypoint distance from rollout points to the A* path."""
    pts = np.stack([xs, ys], axis=1)[:, None, :]
    diff = pts - path_xy[None, :, :]
    sq = np.sum(diff * diff, axis=2)               # (H+1, P)
    min_sq = sq.min(axis=1)                        # (H+1,)
    return float(np.sqrt(min_sq).mean())


def _goal_term(xs: np.ndarray, ys: np.ndarray, target: Point) -> float:
    """Euclidean distance from rollout *endpoint* to *target*."""
    return math.hypot(xs[-1] - target[0], ys[-1] - target[1])


# ── DWA core ────────────────────────────────────────────────────────────────

class DWAPlanner:
    """Stateful DWA planner. Holds the last commanded ω for rate-limiting.

    The obstacle may be ``None`` (Task 5 bonus: unknown obstacle), in which
    case the obstacle-related cost terms and collision rejection are skipped
    until ``set_obstacle()`` is called at runtime.
    """

    def __init__(self, cfg: Config, obstacle: Optional[CircleObstacle],
                 path: Sequence[Point]):
        self.cfg = cfg
        self.obstacle = obstacle
        self.path_xy = np.asarray(path, dtype=float)
        if self.path_xy.ndim != 2 or self.path_xy.shape[1] != 2:
            raise ValueError("path must be a sequence of (x, y) tuples")
        self._last_w: float = 0.0

    def set_obstacle(self, obstacle: Optional[CircleObstacle]) -> None:
        """Inject (or replace) the obstacle used by the planner."""
        self.obstacle = obstacle

    # ── public API ──────────────────────────────────────────────────────────

    def step(self, state: State) -> DWAResult:
        """Compute one DWA step from the current robot state."""
        d = self.cfg.dwa
        v = d.v_max  # constant velocity

        # ω rate-limit window around the last commanded ω.
        w_band = d.alpha_max * d.window_dt
        w_lo = max(d.w_min, self._last_w - w_band)
        w_hi = min(d.w_max, self._last_w + w_band)
        if w_hi < w_lo:
            w_hi = w_lo

        w_grid = (np.linspace(w_lo, w_hi, d.w_samples)
                  if d.w_samples > 1 else np.array([0.0]))

        rollouts: List[Rollout] = []
        for w in w_grid:
            rollouts.append(self._evaluate(state, v, float(w)))

        feasible = [r for r in rollouts if not r.rejected]
        chosen: Optional[Rollout]
        if feasible:
            chosen = min(feasible, key=lambda r: r.cost)
            self._last_w = chosen.w
        else:
            chosen = min(rollouts, key=lambda r: r.cost) if rollouts else None

        return DWAResult(chosen=chosen, rollouts=rollouts,
                         window=(v, v, w_lo, w_hi))

    # ── internals ───────────────────────────────────────────────────────────

    def _evaluate(self, state: State, v: float, w: float) -> Rollout:
        d = self.cfg.dwa
        xs, ys = _rollout(state, v, w, d.horizon_s, d.dt)

        # Hard reject if any rollout point leaves the workspace box.
        # Keeps the robot inside the 1.5 × 1.5 m map (A4.md §2). The robot
        # radius is subtracted from the bounds so the *footprint* stays in
        # the map, not just the centre.
        ws = self.cfg.workspace
        r = self.cfg.robot.radius_m
        if (xs.min() < ws.x_min + r or xs.max() > ws.x_max - r or
                ys.min() < ws.y_min + r or ys.max() > ws.y_max - r):
            return Rollout(v=v, w=w, xs=xs, ys=ys,
                           cost=1e6, rejected=True, reason="out_of_bounds")

        # Hard reject if any rollout point enters the inflated obstacle.
        if (self.obstacle is not None and d.reject_in_inflated
                and self.obstacle.any_in_collision(zip(xs, ys))):
            return Rollout(v=v, w=w, xs=xs, ys=ys,
                           cost=1e6, rejected=True, reason="obstacle")

        path_d = _path_distance_term(xs, ys, self.path_xy)
        goal_d = _goal_term(xs, ys, (self.cfg.goal.x, self.cfg.goal.y))

        obs_cost = 0.0
        if self.obstacle is not None:
            # Soft obstacle term: only when the robot is within sensing range.
            # Ramps from 0 (clearance ≥ safe) to 1 (touching the inflated edge).
            rx, ry, _ = state
            sense_dist = math.hypot(rx - self.obstacle.cx, ry - self.obstacle.cy)
            if sense_dist <= self.cfg.sensing.radius_m + self.obstacle.inflated_radius:
                min_clear = self.obstacle.min_clearance(zip(xs, ys))
                safe = max(1e-3, d.safe_clear_m)
                obs_cost = max(0.0, (safe - min_clear) / safe)

        wts = d.weights
        cost = (wts.path     * path_d +
                wts.goal     * goal_d +
                wts.obstacle * obs_cost)
        return Rollout(v=v, w=w, xs=xs, ys=ys, cost=cost,
                       rejected=False, reason="")


# ── Standalone simulation test ──────────────────────────────────────────────

def _simulate_run(cfg: Config, max_steps: int = 400) -> dict:
    """Run a closed-loop simulation of a unicycle following A*+DWA."""
    from astar_grid import plan as astar_plan

    waypoints = astar_plan(cfg)
    obstacle = CircleObstacle.from_config(cfg)
    planner = DWAPlanner(cfg, obstacle, waypoints)

    x, y, th = cfg.start.x, cfg.start.y, cfg.start.theta
    trace_x, trace_y = [x], [y]
    last_result: Optional[DWAResult] = None
    reached = False
    contact = False
    for step in range(max_steps):
        if math.hypot(x - cfg.goal.x, y - cfg.goal.y) <= cfg.goal.tol_m:
            reached = True
            break
        result = planner.step((x, y, th))
        last_result = result
        if result.chosen is None or result.chosen.rejected:
            break
        v, w = result.chosen.v, result.chosen.w
        dt = cfg.dwa.dt
        th += w * dt
        x  += v * math.cos(th) * dt
        y  += v * math.sin(th) * dt
        trace_x.append(x)
        trace_y.append(y)
        if obstacle.distance_to_centre(x, y) < obstacle.radius:
            contact = True
            break

    return {
        "reached": reached,
        "contact": contact,
        "steps": len(trace_x) - 1,
        "trace_x": trace_x,
        "trace_y": trace_y,
        "waypoints": waypoints,
        "obstacle": obstacle,
        "last_result": last_result,
        "final_state": (x, y, th),
    }


if __name__ == "__main__":
    from config_loader import default_path, load

    cfg = load(default_path())
    sim = _simulate_run(cfg)
    fx, fy, _ = sim["final_state"]
    goal_dist = math.hypot(fx - cfg.goal.x, fy - cfg.goal.y)
    print(f"steps        : {sim['steps']}")
    print(f"reached goal : {sim['reached']}  (final dist={goal_dist:.3f} m)")
    print(f"obstacle hit : {sim['contact']}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
    else:
        obstacle = sim["obstacle"]
        wp = sim["waypoints"]
        result = sim["last_result"]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(cfg.workspace.x_min, cfg.workspace.x_max)
        ax.set_ylim(cfg.workspace.y_min, cfg.workspace.y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        ax.plot([p[0] for p in wp], [p[1] for p in wp], "-",
                color="#1f77b4", lw=2, label="A* path")
        ax.plot(sim["trace_x"], sim["trace_y"], "-", color="#2ca02c",
                lw=2, label="executed trajectory")
        ax.plot(cfg.start.x, cfg.start.y, "go", ms=10)
        ax.plot(cfg.goal.x, cfg.goal.y, "r*", ms=14)

        ax.add_patch(plt.Circle((obstacle.cx, obstacle.cy), obstacle.radius,
                                color="k", alpha=0.7, label="obstacle"))
        ax.add_patch(plt.Circle((obstacle.cx, obstacle.cy), obstacle.inflated_radius,
                                fill=False, color="k", linestyle="--",
                                label="inflated boundary"))

        if result is not None:
            for r in result.rollouts:
                col = "lightgray" if r.rejected else "#aec7e8"
                ax.plot(r.xs, r.ys, "-", color=col, lw=0.6, alpha=0.7)
            if result.chosen is not None:
                ax.plot(result.chosen.xs, result.chosen.ys, "-",
                        color="#ff7f0e", lw=2.0, label="chosen rollout")

        ax.legend(loc="lower right", fontsize=9)
        from pathlib import Path
        out = str(Path(__file__).resolve().parents[1] / "output" / "dwa.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"plot saved   : {out}")
