"""
costmap.py – Static circular obstacle representation used by the DWA planner.

The map is intentionally trivial: a single circular obstacle at a known
position, inflated by ``robot.radius_m + robot.safety_margin_m`` for
collision checking. The same object also exposes the *distance* from any
query point to the obstacle surface, which the DWA cost function uses to
penalise rollouts that approach the obstacle without yet entering the
inflated zone.

The class is deliberately framework-free so the DWA test bench and the ROS
nodes can use the same object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

from config_loader import Config

Point = Tuple[float, float]


@dataclass(frozen=True)
class CircleObstacle:
    """Single circular obstacle with an inflated safety zone."""

    cx: float
    cy: float
    radius: float          # physical radius (metres)
    inflation: float       # robot_radius + safety_margin (metres)

    @property
    def inflated_radius(self) -> float:
        return self.radius + self.inflation

    def distance_to_centre(self, x: float, y: float) -> float:
        return math.hypot(x - self.cx, y - self.cy)

    def signed_clearance(self, x: float, y: float) -> float:
        """
        Distance from (x, y) to the *inflated* boundary.
        Positive outside the inflated zone, zero on the boundary,
        negative inside (collision).
        """
        return self.distance_to_centre(x, y) - self.inflated_radius

    def in_collision(self, x: float, y: float) -> bool:
        return self.signed_clearance(x, y) <= 0.0

    def any_in_collision(self, points: Iterable[Point]) -> bool:
        return any(self.in_collision(x, y) for x, y in points)

    def min_clearance(self, points: Iterable[Point]) -> float:
        """Return the minimum signed clearance over a sequence of points."""
        return min(self.signed_clearance(x, y) for x, y in points)

    @classmethod
    def from_config(cls, cfg: Config) -> "CircleObstacle":
        return cls(
            cx=cfg.obstacle.x,
            cy=cfg.obstacle.y,
            radius=cfg.obstacle.radius_m,
            inflation=cfg.robot.inflated_radius_m,
        )


# ── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config_loader import default_path, load

    cfg = load(default_path())
    obs = CircleObstacle.from_config(cfg)
    print(f"obstacle centre   : ({obs.cx}, {obs.cy})")
    print(f"physical radius   : {obs.radius:.3f} m")
    print(f"inflation         : {obs.inflation:.3f} m")
    print(f"inflated radius   : {obs.inflated_radius:.3f} m")
    for x, y in [(cfg.start.x, cfg.start.y),
                 (obs.cx, obs.cy),
                 (obs.cx + obs.inflated_radius + 0.01, obs.cy)]:
        print(
            f"  ({x:.2f}, {y:.2f}): clearance={obs.signed_clearance(x, y):+.3f}  "
            f"collision={obs.in_collision(x, y)}"
        )
