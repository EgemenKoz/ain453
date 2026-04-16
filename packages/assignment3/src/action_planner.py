"""
action_planner.py – Converts an A* node-path into a sequence of discrete
                    TURN_LEFT / TURN_RIGHT / FORWARD actions.

Heading convention
------------------
  EAST  (0) : +x direction  (x increases right in NODE_COORDS)
  NORTH (1) : +y direction  (y increases up)
  WEST  (2) : -x direction
  SOUTH (3) : -y direction

Turn directions (when viewed from above)
-----------------------------------------
  TURN_LEFT  – 90° counter-clockwise (CCW)
  TURN_RIGHT – 90° clockwise         (CW)

Examples
--------
  Facing EAST, need to go NORTH → TURN_LEFT then FORWARD
  Facing EAST, need to go SOUTH → TURN_RIGHT then FORWARD
  Facing EAST, need to go WEST  → TURN_LEFT + TURN_LEFT then FORWARD
  Facing EAST, need to go EAST  → FORWARD (already aligned)
"""

from typing import List, Tuple

from config import INITIAL_HEADING, NODE_COORDS

# ── Action constants ──────────────────────────────────────────────────────────

FORWARD    = "FORWARD"
TURN_LEFT  = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"

# ── Heading helpers ───────────────────────────────────────────────────────────

# Maps (dx, dy) unit step → heading int
_DELTA_TO_HEADING = {
    ( 1,  0): 0,   # EAST
    ( 0,  1): 1,   # NORTH
    (-1,  0): 2,   # WEST
    ( 0, -1): 3,   # SOUTH
}

HEADING_NAMES = {0: "EAST", 1: "NORTH", 2: "WEST", 3: "SOUTH"}


def required_heading(from_node: int, to_node: int) -> int:
    """
    Return the heading (0–3) needed to step from from_node to to_node.

    Raises ValueError if the two nodes are not adjacent by a unit grid step.
    """
    x1, y1 = NODE_COORDS[from_node]
    x2, y2 = NODE_COORDS[to_node]
    dx = int(x2 - x1)
    dy = int(y2 - y1)
    # Normalise to unit step
    if dx != 0:
        dx = dx // abs(dx)
    if dy != 0:
        dy = dy // abs(dy)
    key = (dx, dy)
    if key not in _DELTA_TO_HEADING:
        raise ValueError(
            f"Non-adjacent or diagonal step N{from_node}→N{to_node}: "
            f"delta=({x2-x1},{y2-y1})"
        )
    return _DELTA_TO_HEADING[key]


# ── Main planner ──────────────────────────────────────────────────────────────

def compute_actions(path: List[int]) -> List[Tuple[str, int]]:
    """
    Convert a node path (from A*) into an ordered list of discrete actions.

    Parameters
    ----------
    path : list of node IDs from START_NODE to GOAL_NODE (inclusive)

    Returns
    -------
    List of (action_type, path_step_idx) tuples where:
      action_type   ∈ {FORWARD, TURN_LEFT, TURN_RIGHT}
      path_step_idx = index in `path` of the *destination* node for this
                      group of actions.

    Multiple actions can share the same path_step_idx when a turn precedes
    the forward move toward that node.  Execution order is preserved.

    Example
    -------
    path = [0, 1, 5, 9]  (N0→N1 east, N1→N5 north, N5→N9 north)
    INITIAL_HEADING = 0  (EAST)

    → [("FORWARD", 1),                 # N0→N1, already facing EAST
       ("TURN_LEFT", 2), ("FORWARD", 2),  # turn then go N1→N5
       ("FORWARD", 3)]                 # still facing NORTH: N5→N9
    """
    actions: List[Tuple[str, int]] = []
    heading = INITIAL_HEADING

    for i in range(1, len(path)):
        req = required_heading(path[i - 1], path[i])
        diff = (req - heading) % 4   # how many CCW quarter-turns needed

        if diff == 1:
            # One left turn
            actions.append((TURN_LEFT, i))
        elif diff == 2:
            # 180°: two left turns  (could also use two rights — same cost)
            actions.append((TURN_LEFT, i))
            actions.append((TURN_LEFT, i))
        elif diff == 3:
            # One right turn  (= three lefts, but one right is shorter)
            actions.append((TURN_RIGHT, i))
        # diff == 0: already aligned, no turn needed

        actions.append((FORWARD, i))
        heading = req

    return actions


def format_actions(path: List[int], actions: List[Tuple[str, int]]) -> str:
    """Return a human-readable summary of the action plan."""
    parts = []
    for action_type, step_idx in actions:
        node = path[step_idx]
        parts.append(f"{action_type}→N{node}")
    return ", ".join(parts)
