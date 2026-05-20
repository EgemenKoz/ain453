"""
run_offline.py – Tek komutla tüm offline pipeline'ı çalıştır.

ROS'a ihtiyaç yok. A* yolunu hesaplar, DWA ile A→B kapalı döngü
simülasyonu yapar ve Task-4 görselleştirmesini animasyonlu GIF olarak kaydeder.
Çıktılar packages/assignment4/output/ altına yazılır.

Kullanım:
    python3 run_offline.py                # default config (params.yaml)
    python3 run_offline.py --config OTHER.yaml
    python3 run_offline.py --no-gif       # sadece son kareyi PNG olarak kaydet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from astar_grid import Grid, path_length, plan as astar_plan
from config_loader import Config, default_path, load
from costmap import CircleObstacle
from dwa import DWAPlanner, DWAResult
from viz import Renderer, Snapshot

Point = Tuple[float, float]
State = Tuple[float, float, float]


# ── Steps ───────────────────────────────────────────────────────────────────

def step_astar(cfg: Config, out_dir: Path) -> List[Point]:
    grid = Grid.empty(cfg)
    waypoints = astar_plan(cfg, grid)
    print(f"[A*]  grid {grid.nx}x{grid.ny} cells (res={grid.res} m)")
    print(f"[A*]  {len(waypoints)} dense waypoints, length {path_length(waypoints):.3f} m")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(cfg.workspace.x_min, cfg.workspace.x_max)
    ax.set_ylim(cfg.workspace.y_min, cfg.workspace.y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.plot([p[0] for p in waypoints], [p[1] for p in waypoints],
            "-b", lw=2, label="A* path")
    ax.plot(cfg.start.x, cfg.start.y, "go", ms=10, label="A")
    ax.plot(cfg.goal.x, cfg.goal.y, "r*", ms=14, label="B")
    ax.add_patch(plt.Circle((cfg.obstacle.x, cfg.obstacle.y),
                            cfg.obstacle.radius_m, fill=False, color="k",
                            label="obstacle (display only)"))
    ax.set_title("Task 1 — A* global path (empty grid)")
    ax.legend(loc="lower right", fontsize=9)
    out = out_dir / "astar.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[A*]  saved {out}")
    return waypoints


def step_simulate(cfg: Config, waypoints: List[Point],
                  obstacle: CircleObstacle,
                  max_steps: int = 400,
                  bonus: bool = False
                  ) -> Tuple[List[Snapshot], dict]:
    """Closed-loop A*+DWA simulation.

    ``obstacle`` is the *true* obstacle in the world. In bonus mode the
    planner starts blind to it; the obstacle is "detected" and handed to
    the planner once the simulated forward ToF beam intersects it within
    ``cfg.bonus.detect_distance_m``.
    """
    # Static obstacle is always active. In bonus mode a second obstacle is
    # appended once the simulated ToF beam hits the true circle.
    planner = DWAPlanner(cfg, obstacle, waypoints)
    state: State = (cfg.start.x, cfg.start.y, cfg.start.theta)
    trace: List[Point] = [(state[0], state[1])]
    snapshots: List[Snapshot] = []
    reached = False
    contact = False
    detected: bool = False  # ToF detection (bonus only)
    detected_obstacle: CircleObstacle | None = None

    for step in range(max_steps):
        if bonus and not detected:
            det = _simulate_tof_detection(cfg, state, obstacle)
            if det is not None:
                planner.add_obstacle(det)  # static obstacle stays; ToF obs added
                detected_obstacle = det
                detected = True

        result: DWAResult = planner.step(state)
        snapshots.append(Snapshot(
            state=state,
            rollouts=list(result.rollouts),
            chosen=result.chosen,
            trace=list(trace),
            step=step,
            info="(unknown obstacle)" if bonus and not detected else "",
            obstacle=detected_obstacle,
        ))
        d_goal = math.hypot(state[0] - cfg.goal.x, state[1] - cfg.goal.y)
        if d_goal <= cfg.goal.tol_m:
            snapshots[-1].info = "*** goal reached ***"
            reached = True
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
        if obstacle.distance_to_centre(x, y) < obstacle.radius:
            contact = True
            break

    summary = {
        "steps": len(snapshots),
        "reached": reached,
        "contact": contact,
        "final_state": state,
        "final_d_goal": math.hypot(state[0] - cfg.goal.x, state[1] - cfg.goal.y),
    }
    print(f"[Sim] steps={summary['steps']}  reached={reached}  "
          f"contact={contact}  final d_goal={summary['final_d_goal']:.3f} m")
    if bonus:
        if detected_obstacle is not None:
            print(f"[Sim] bonus: obstacle detected at "
                  f"({detected_obstacle.cx:.2f}, {detected_obstacle.cy:.2f})  "
                  f"r={detected_obstacle.radius:.3f} m  "
                  f"(true centre ({obstacle.cx:.2f}, {obstacle.cy:.2f}))")
        else:
            print("[Sim] bonus: obstacle was never detected by simulated ToF")
    summary["detected_obstacle"] = detected_obstacle
    return snapshots, summary


def _simulate_tof_detection(cfg: Config, state: State,
                            truth: CircleObstacle) -> CircleObstacle | None:
    """Simulate a forward ToF beam against the *true* obstacle.

    Returns a detected obstacle (with bonus-radius, world-frame centre
    estimated from the hit point) if the beam intersects the true circle
    within ``cfg.bonus.detect_distance_m``. Otherwise None.
    """
    x, y, th = state
    # ToF mounting point.
    ox = x + cfg.bonus.tof_offset_m * math.cos(th)
    oy = y + cfg.bonus.tof_offset_m * math.sin(th)
    # Closest distance from truth centre to the beam line (perp distance).
    dx = truth.cx - ox
    dy = truth.cy - oy
    proj = dx * math.cos(th) + dy * math.sin(th)  # along beam
    perp_sq = dx * dx + dy * dy - proj * proj
    if perp_sq > truth.radius * truth.radius or proj <= 0.0:
        return None
    # Ray-circle entry distance along the beam.
    entry = proj - math.sqrt(max(0.0, truth.radius * truth.radius - perp_sq))
    if entry <= 0.0 or entry > cfg.bonus.detect_distance_m:
        return None
    # Project the hit point and shift forward by detected_radius so the
    # circle straddles the true obstacle instead of clipping its near face.
    radius = cfg.bonus.detected_radius_m
    d = cfg.bonus.tof_offset_m + entry + radius
    cx = x + d * math.cos(th)
    cy = y + d * math.sin(th)
    return CircleObstacle(
        cx=cx, cy=cy, radius=radius, inflation=cfg.robot.inflated_radius_m,
    )


def step_render(cfg: Config, waypoints: List[Point],
                obstacle: CircleObstacle,
                snapshots: List[Snapshot], out_dir: Path,
                make_gif: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle("Assignment 4 — A* + DWA local planner", fontsize=11)
    renderer = Renderer(cfg, waypoints, obstacle, ax=ax)

    # Always save the final frame as a still PNG.
    renderer.update(snapshots[-1])
    final_png = out_dir / "final_frame.png"
    fig.savefig(final_png, dpi=120, bbox_inches="tight")
    print(f"[Viz] saved {final_png}")

    if not make_gif:
        plt.close(fig)
        return

    # Then re-run as animation.
    def update(i: int):
        renderer.update(snapshots[i])
        return ()

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=80, blit=False,
    )
    gif = out_dir / "run.gif"
    try:
        anim.save(gif, writer="pillow", fps=12)
        print(f"[Viz] saved {gif}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Viz] GIF save failed ({exc}); final frame still written")
    plt.close(fig)


# ── Entry ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Assignment 4 offline pipeline")
    ap.add_argument("--config", type=Path, default=default_path(),
                    help="path to params.yaml")
    ap.add_argument("--no-gif", action="store_true",
                    help="skip animated GIF (still saves PNGs)")
    ap.add_argument("--max-steps", type=int, default=400,
                    help="simulation step cap")
    ap.add_argument("--bonus", action="store_true",
                    help="Task 5 bonus: obstacle is unknown until simulated "
                         "ToF detects it; overrides bonus.enabled in YAML")
    args = ap.parse_args()

    cfg = load(args.config)
    bonus = args.bonus or cfg.bonus.enabled
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config : {args.config}")
    print(f"output : {out_dir}")
    print(f"bonus  : {bonus}")
    print()
    print("─── Task 1: A* global path ─────────────────────────────────")
    waypoints = step_astar(cfg, out_dir)
    print()
    section = ("Task 5 bonus: unknown obstacle + DWA"
               if bonus else "Tasks 2+3: DWA + obstacle avoidance")
    print(f"─── {section} ────")
    obstacle = CircleObstacle.from_config(cfg)
    snapshots, summary = step_simulate(cfg, waypoints, obstacle,
                                       max_steps=args.max_steps,
                                       bonus=bonus)
    print()
    print("─── Task 4: visualisation ──────────────────────────────────")
    # Static obstacle is always visible from frame 0. In bonus mode the
    # ToF-detected second obstacle appears via snap.obstacle in the animation.
    step_render(cfg, waypoints, obstacle, snapshots, out_dir,
                make_gif=not args.no_gif)
    print()
    print("done.")


if __name__ == "__main__":
    main()
