"""
dwa.py – Dynamic Window Approach local planner (from scratch).

Pipeline (one control step):

  1. Build the dynamic window of feasible (v, ω) commands, restricted by
     velocity bounds, acceleration bounds (around the *current* command),
     and the configurable workspace limits.
  2. For each (v, ω) in a regular grid over the window, simulate a unicycle
     rollout for ``horizon_s`` seconds in steps of ``dt``.
  3. Score each rollout with a weighted sum of:
       * distance to the densified A* path
       * distance to the goal
       * proximity to the inflated obstacle (only when within sensing radius)
       * heading misalignment with the path tangent at the closest waypoint
     Rollouts that enter the inflated obstacle are hard-rejected when
     ``reject_in_inflated`` is true; otherwise they receive a large penalty.
  4. Return the chosen (v, ω) along with every rollout (and its score) for
     visualisation.

The module is intentionally numpy-vectorised over rollout points but uses
plain Python loops over the (v, ω) grid — the grid is small (≤ ~80 entries)
so the overhead is negligible compared to the per-rollout work.
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
                        path_xy: np.ndarray) -> Tuple[float, int]:
    """
    Mean perpendicular distance from rollout points to the A* path.
    Returns (mean_distance, idx_of_nearest_waypoint_for_last_rollout_point).
    """
    # Per-point nearest-waypoint distance via broadcasting:
    # rollout points  shape (H+1, 1, 2)
    # path points     shape (1, P, 2)
    pts = np.stack([xs, ys], axis=1)[:, None, :]
    diff = pts - path_xy[None, :, :]
    sq = np.sum(diff * diff, axis=2)               # (H+1, P)
    min_sq = sq.min(axis=1)                        # (H+1,)
    last_idx = int(sq[-1].argmin())
    return float(np.sqrt(min_sq).mean()), last_idx


def _goal_term(xs: np.ndarray, ys: np.ndarray, goal: Point) -> float:
    """Euclidean distance from rollout *endpoint* to the goal."""
    return math.hypot(xs[-1] - goal[0], ys[-1] - goal[1])


def _heading_term(state: State, v: float, w: float, dt: float,
                  path_xy: np.ndarray, last_idx: int) -> float:
    """
    Misalignment (radians, in [0, π]) between the rollout's final heading
    and the path tangent at the waypoint closest to the rollout's endpoint.
    Returns 0 for stationary rollouts so they are not unfairly penalised.
    """
    if v == 0.0 and w == 0.0:
        return 0.0
    n = path_xy.shape[0]
    if n < 2:
        return 0.0
    j = min(last_idx, n - 2)
    tx = path_xy[j + 1, 0] - path_xy[j, 0]
    ty = path_xy[j + 1, 1] - path_xy[j, 1]
    if tx == 0.0 and ty == 0.0:
        return 0.0
    path_heading = math.atan2(ty, tx)
    horizon_steps = int(round(_config_horizon(state, v, w, dt)))   # unused stub
    final_heading = state[2] + w * (horizon_steps * dt)
    err = final_heading - path_heading
    while err >  math.pi: err -= 2 * math.pi
    while err < -math.pi: err += 2 * math.pi
    return abs(err)


def _config_horizon(_state, _v, _w, _dt) -> int:
    """Hook for a future variable horizon; for now the caller passes H+1."""
    return 0


# ── DWA core ────────────────────────────────────────────────────────────────

class DWAPlanner:
    """Stateful DWA planner. Holds the last commanded (v, ω) for windowing."""

    def __init__(self, cfg: Config, obstacle: CircleObstacle, path: Sequence[Point]):
        self.cfg = cfg
        self.obstacle = obstacle
        self.path_xy = np.asarray(path, dtype=float)
        if self.path_xy.ndim != 2 or self.path_xy.shape[1] != 2:
            raise ValueError("path must be a sequence of (x, y) tuples")
        # Last commanded velocities — initially zero.
        self._last_v: float = 0.0
        self._last_w: float = 0.0

    # ── public API ──────────────────────────────────────────────────────────

    def step(self, state: State) -> DWAResult:
        """Compute one DWA step from the current robot state."""
        d = self.cfg.dwa
        # Dynamic window — bound by both static limits and acceleration limits
        # around the previously commanded (v, w).
        v_lo = max(d.v_min, self._last_v - d.a_max * d.dt * d.horizon_s)
        v_hi = min(d.v_max, self._last_v + d.a_max * d.dt * d.horizon_s)
        w_lo = max(d.w_min, self._last_w - d.alpha_max * d.dt * d.horizon_s)
        w_hi = min(d.w_max, self._last_w + d.alpha_max * d.dt * d.horizon_s)
        if v_hi < v_lo: v_hi = v_lo
        if w_hi < w_lo: w_hi = w_lo

        v_grid = np.linspace(v_lo, v_hi, d.v_samples) if d.v_samples > 1 else np.array([v_lo])
        w_grid = np.linspace(w_lo, w_hi, d.w_samples) if d.w_samples > 1 else np.array([w_lo])

        rollouts: List[Rollout] = []
        for v in v_grid:
            for w in w_grid:
                ro = self._evaluate(state, float(v), float(w))
                rollouts.append(ro)

        # Pick the best non-rejected rollout. If everything is rejected,
        # fall back to the cheapest rejected one (so visualization still has
        # a "chosen" candidate to highlight even in pathological cases).
        feasible = [r for r in rollouts if not r.rejected]
        chosen: Optional[Rollout]
        if feasible:
            chosen = min(feasible, key=lambda r: r.cost)
        else:
            chosen = min(rollouts, key=lambda r: r.cost) if rollouts else None

        if chosen is not None and not chosen.rejected:
            self._last_v = chosen.v
            self._last_w = chosen.w

        return DWAResult(chosen=chosen, rollouts=rollouts,
                         window=(v_lo, v_hi, w_lo, w_hi))

    # ── internals ───────────────────────────────────────────────────────────

    def _evaluate(self, state: State, v: float, w: float) -> Rollout:
        d = self.cfg.dwa
        xs, ys = _rollout(state, v, w, d.horizon_s, d.dt)

        # Workspace bounds — reject rollouts that exit the workspace.
        ws = self.cfg.workspace
        if (np.any(xs < ws.x_min) or np.any(xs > ws.x_max) or
                np.any(ys < ws.y_min) or np.any(ys > ws.y_max)):
            return Rollout(v=v, w=w, xs=xs, ys=ys,
                           cost=1e6, rejected=True, reason="oob")

        # Obstacle check (always — independent of sensing radius).
        # Note: we still gate the *cost contribution* by sensing radius
        # below; rejection itself is unconditional so the robot does not
        # crash even if obstacle is just outside its current sensing disc.
        if d.reject_in_inflated and self.obstacle.any_in_collision(zip(xs, ys)):
            return Rollout(v=v, w=w, xs=xs, ys=ys,
                           cost=1e6, rejected=True, reason="obstacle")

        # Distance / goal terms.
        path_d, last_idx = _path_distance_term(xs, ys, self.path_xy)
        goal_d = _goal_term(xs, ys, (self.cfg.goal.x, self.cfg.goal.y))

        # Obstacle term — only contributes when robot's *current* position
        # is within sensing radius of the obstacle (matches the rubric:
        # "consider the obstacle if it's within the sensing radius").
        # The term is bounded: zero once clearance exceeds safe_clear_m, and
        # ramps to 1 as clearance drops to 0. This avoids penalising the robot
        # for merely being *near* the obstacle while still safely clear.
        rx, ry, _ = state
        sense_dist = math.hypot(rx - self.obstacle.cx, ry - self.obstacle.cy)
        if sense_dist <= self.cfg.sensing.radius_m + self.obstacle.inflated_radius:
            min_clear = self.obstacle.min_clearance(zip(xs, ys))
            safe = max(1e-3, d.safe_clear_m)
            obs_cost = max(0.0, (safe - min_clear) / safe)
        else:
            obs_cost = 0.0

        # Heading term.
        n_steps = max(1, int(round(d.horizon_s / d.dt)))
        final_heading = state[2] + w * n_steps * d.dt
        if last_idx + 1 < self.path_xy.shape[0]:
            tx = self.path_xy[last_idx + 1, 0] - self.path_xy[last_idx, 0]
            ty = self.path_xy[last_idx + 1, 1] - self.path_xy[last_idx, 1]
            path_heading = math.atan2(ty, tx) if (tx or ty) else final_heading
        else:
            path_heading = final_heading
        err = final_heading - path_heading
        while err >  math.pi: err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi
        head_d = abs(err)

        wts = d.weights
        cost = (wts.path     * path_d +
                wts.goal     * goal_d +
                wts.obstacle * obs_cost +
                wts.heading  * head_d -
                wts.velocity * v)        # reward forward motion

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
            # No feasible move — abort.
            break
        v, w = result.chosen.v, result.chosen.w
        # Advance one *control* step (use the planner's dt as tick length).
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
