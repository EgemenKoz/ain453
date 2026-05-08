"""
config_loader.py – Typed view of params.yaml.

Loads the YAML file once and exposes nested dataclasses so the rest of the
package can read parameters with attribute access (cfg.dwa.v_max) instead of
dict lookups. Failing fast at load time also catches typos in YAML keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Workspace:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class Grid:
    resolution_m: float
    allow_diagonal: bool


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float = 0.0


@dataclass(frozen=True)
class Goal:
    x: float
    y: float
    tol_m: float


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius_m: float


@dataclass(frozen=True)
class Robot:
    radius_m: float
    safety_margin_m: float
    wheel_base_m: float
    wheel_radius_m: float
    ticks_per_rev: int

    @property
    def inflated_radius_m(self) -> float:
        return self.radius_m + self.safety_margin_m


@dataclass(frozen=True)
class Sensing:
    radius_m: float


@dataclass(frozen=True)
class DWAWeights:
    path: float
    goal: float
    obstacle: float
    heading: float
    velocity: float


@dataclass(frozen=True)
class DWA:
    v_min: float
    v_max: float
    w_min: float
    w_max: float
    v_samples: int
    w_samples: int
    a_max: float
    alpha_max: float
    horizon_s: float
    dt: float
    weights: DWAWeights
    safe_clear_m: float
    reject_in_inflated: bool


@dataclass(frozen=True)
class Control:
    rate_hz: float
    linear_max: float
    angular_max: float


@dataclass(frozen=True)
class Viz:
    rate_hz: float
    show_rejected: bool


@dataclass(frozen=True)
class PathCfg:
    densify_step_m: float


@dataclass(frozen=True)
class Config:
    workspace: Workspace
    grid: Grid
    start: Pose
    goal: Goal
    obstacle: Obstacle
    robot: Robot
    sensing: Sensing
    dwa: DWA
    control: Control
    viz: Viz
    path: PathCfg


def _build(raw: Dict[str, Any]) -> Config:
    dwa_raw = raw["dwa"]
    return Config(
        workspace=Workspace(**raw["workspace"]),
        grid=Grid(**raw["grid"]),
        start=Pose(**raw["start"]),
        goal=Goal(**raw["goal"]),
        obstacle=Obstacle(**raw["obstacle"]),
        robot=Robot(**raw["robot"]),
        sensing=Sensing(**raw["sensing"]),
        dwa=DWA(
            v_min=dwa_raw["v_min"],
            v_max=dwa_raw["v_max"],
            w_min=dwa_raw["w_min"],
            w_max=dwa_raw["w_max"],
            v_samples=dwa_raw["v_samples"],
            w_samples=dwa_raw["w_samples"],
            a_max=dwa_raw["a_max"],
            alpha_max=dwa_raw["alpha_max"],
            horizon_s=dwa_raw["horizon_s"],
            dt=dwa_raw["dt"],
            weights=DWAWeights(**dwa_raw["weights"]),
            safe_clear_m=dwa_raw["safe_clear_m"],
            reject_in_inflated=dwa_raw["reject_in_inflated"],
        ),
        control=Control(**raw["control"]),
        viz=Viz(**raw["viz"]),
        path=PathCfg(**raw["path"]),
    )


def load(path: str | Path) -> Config:
    """Load a Config from *path*. Raises if the file is missing or malformed."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return _build(raw)


def default_path() -> Path:
    """Path to the in-package params.yaml (works for both rosrun and CLI use)."""
    return Path(__file__).resolve().parents[1] / "config" / "params.yaml"


if __name__ == "__main__":
    cfg = load(default_path())
    print(cfg)
