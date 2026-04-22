"""
visualizer.py – OpenCV-based grid map visualizer for A* navigation.

Publishes a sensor_msgs/Image showing:
  - 4×4 grid nodes and all edges (walls shown as crossed lines)
  - A* path highlighted in orange
  - Robot's estimated position and heading
  - Currently detected ARTag IDs
  - State machine status and target node
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import EDGES, GOAL_NODE, NODE_COORDS, START_NODE

# ── Layout constants ──────────────────────────────────────────────────────────
_IMG_SIZE   = 620          # square canvas (pixels)
_MARGIN     = 70           # border around grid (pixels)
_GRID_PX    = _IMG_SIZE - 2 * _MARGIN   # usable grid area

# Derived: pixel step between adjacent nodes
_STEP = _GRID_PX // 3      # 3 intervals for 4 nodes

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
_C_BG           = (245, 245, 245)
_C_EDGE         = (190, 190, 190)
_C_PATH_EDGE    = (0,   140, 255)   # orange-blue (A* path)
_C_NODE_DEFAULT = (180, 180, 180)
_C_NODE_START   = (80,  200,  80)   # green
_C_NODE_GOAL    = (50,   50, 230)   # red
_C_NODE_PATH    = (0,   170, 255)   # gold/orange
_C_NODE_VISITED = (160, 160, 160)
_C_NODE_CURRENT = (0,    80, 230)   # red (robot here)
_C_NODE_TARGET  = (0,   200, 200)   # yellow (next waypoint)
_C_NODE_SEEN    = (200, 100,   0)   # teal (tag detected)
_C_ROBOT_ARROW  = (0,    50, 200)
_C_TEXT         = (30,   30,  30)
_C_WALL         = (60,   60, 200)   # dark-red (wall indicator)


def _node_px(node_id: int) -> Tuple[int, int]:
    """Return pixel (x, y) of a node (y-up grid → y-down image)."""
    gx, gy = NODE_COORDS[node_id]
    px = _MARGIN + gx * _STEP
    py = _MARGIN + (3 - gy) * _STEP   # flip y
    return (px, py)


def _build_edge_set() -> Set[Tuple[int, int]]:
    s = set()
    for u, v, _ in EDGES:
        s.add((min(u, v), max(u, v)))
    return s


_VALID_EDGES = _build_edge_set()


def _path_edge_set(path: List[int]) -> Set[Tuple[int, int]]:
    s = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        s.add((min(u, v), max(u, v)))
    return s


class GridVisualizer:
    """
    Stateless renderer.  Call render() each control cycle to get a fresh image.
    """

    def __init__(self, path: List[int]):
        self._path      = path
        self._path_set  = _path_edge_set(path)
        self._path_nodes= set(path)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        current_node: int,
        next_node: int,
        heading_rad: float,
        detected_ids: Set[int],
        visited_nodes: Set[int],
        state: str,
        step_idx: int,
    ) -> np.ndarray:
        """
        Render the full visualization frame.

        Parameters
        ----------
        current_node  : robot's last confirmed node
        next_node     : node the robot is currently heading toward
        heading_rad   : robot's estimated heading (radians, grid frame)
        detected_ids  : set of ARTag IDs visible in the current camera frame
        visited_nodes : nodes already reached (including current)
        state         : state machine string (SEARCHING / APPROACHING / …)
        step_idx      : index into path (for progress display)
        """
        img = np.full((_IMG_SIZE, _IMG_SIZE, 3), _C_BG, dtype=np.uint8)

        self._draw_walls(img)
        self._draw_edges(img)
        self._draw_nodes(img, current_node, next_node, detected_ids, visited_nodes)
        self._draw_robot(img, current_node, heading_rad)
        self._draw_legend(img, current_node, next_node, state, step_idx, detected_ids)

        return img

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_walls(self, img: np.ndarray) -> None:
        """Draw a small X on grid positions that are NOT connected (walls)."""
        for n in NODE_COORDS:
            gx, gy = NODE_COORDS[n]
            # Check all 4 cardinal neighbours
            for dx, dy in ((1, 0), (0, 1)):
                nx_coord = gx + dx
                ny_coord = gy + dy
                # Find neighbour node with those coords
                nb = next(
                    (k for k, (cx, cy) in NODE_COORDS.items() if cx == nx_coord and cy == ny_coord),
                    None,
                )
                if nb is None:
                    continue
                edge_key = (min(n, nb), max(n, nb))
                if edge_key not in _VALID_EDGES:
                    # Draw a wall tick between the two nodes
                    p1 = _node_px(n)
                    p2 = _node_px(nb)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    hs = 8
                    cv2.line(img, (mid[0]-hs, mid[1]-hs), (mid[0]+hs, mid[1]+hs), _C_WALL, 3)
                    cv2.line(img, (mid[0]+hs, mid[1]-hs), (mid[0]-hs, mid[1]+hs), _C_WALL, 3)

    def _draw_edges(self, img: np.ndarray) -> None:
        for u, v, cost in EDGES:
            p1 = _node_px(u)
            p2 = _node_px(v)
            key = (min(u, v), max(u, v))
            if key in self._path_set:
                cv2.line(img, p1, p2, _C_PATH_EDGE, 4)
                # Edge cost label
                mid = ((p1[0]+p2[0])//2 + 6, (p1[1]+p2[1])//2 - 6)
                cv2.putText(img, f"{cost:.1f}", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C_PATH_EDGE, 1, cv2.LINE_AA)
            else:
                cv2.line(img, p1, p2, _C_EDGE, 2)

    def _draw_nodes(
        self,
        img: np.ndarray,
        current_node: int,
        next_node: int,
        detected_ids: Set[int],
        visited_nodes: Set[int],
    ) -> None:
        for nid in NODE_COORDS:
            px = _node_px(nid)
            r = 20

            # Pick fill colour (priority order)
            if nid == current_node:
                fill = _C_NODE_CURRENT
            elif nid == next_node:
                fill = _C_NODE_TARGET
            elif nid in detected_ids:
                fill = _C_NODE_SEEN
            elif nid == GOAL_NODE:
                fill = _C_NODE_GOAL
            elif nid == START_NODE:
                fill = _C_NODE_START
            elif nid in visited_nodes:
                fill = _C_NODE_VISITED
            elif nid in self._path_nodes:
                fill = _C_NODE_PATH
            else:
                fill = _C_NODE_DEFAULT

            cv2.circle(img, px, r, fill, -1)
            cv2.circle(img, px, r, (80, 80, 80), 1)

            # Node label
            label = f"N{nid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, label,
                        (px[0] - tw//2, px[1] + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_robot(self, img: np.ndarray, current_node: int, heading_rad: float) -> None:
        """Draw an arrow on the current node indicating heading."""
        px = _node_px(current_node)
        arrow_len = 32
        # heading_rad: 0 = +x (right), π/2 = +y (up in grid = down in image)
        # Image y is flipped vs grid y
        dx = int(arrow_len * math.cos(heading_rad))
        dy = int(-arrow_len * math.sin(heading_rad))   # flip y for image coords
        tip = (px[0] + dx, px[1] + dy)
        cv2.arrowedLine(img, px, tip, _C_ROBOT_ARROW, 3, tipLength=0.4)

    def _draw_legend(
        self,
        img: np.ndarray,
        current_node: int,
        next_node: int,
        state: str,
        step_idx: int,
        detected_ids: Set[int],
    ) -> None:
        total = len(self._path) - 1
        progress = f"Step {step_idx}/{total}"
        path_str = " > ".join(f"N{n}" for n in self._path)

        lines = [
            f"State  : {state}",
            f"At     : N{current_node}",
            f"Target : N{next_node}",
            f"Seen   : {sorted(detected_ids) if detected_ids else '-'}",
            f"Progress: {progress}",
            f"Path: {path_str}",
        ]

        y = _IMG_SIZE - 10 - len(lines) * 18
        cv2.rectangle(img, (0, y - 8), (_IMG_SIZE, _IMG_SIZE), (230, 230, 230), -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (8, y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_TEXT, 1, cv2.LINE_AA)

        # Axis labels (x/y direction indicators)
        ax_y = _MARGIN - 30
        cv2.putText(img, "+y (N4/N8/N12 dir)", (8, ax_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        cv2.putText(img, "+x →", (_IMG_SIZE - 60, _MARGIN + 3 * _STEP + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
