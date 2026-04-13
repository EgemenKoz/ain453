"""
astar.py – A* pathfinding on the 4×4 grid maze.

Implements A* entirely from scratch:
  - OPEN list backed by a min-heap (priority queue)
  - CLOSED set to skip already-expanded nodes
  - Per-node g(n), h(n), f(n) = g(n) + h(n) values
  - Tie-breaking rule: when f values are equal, prefer lower h(n)
  - Path reconstruction via parent tracking
  - Configurable heuristic: "euclidean" or "manhattan" (set in config.py)
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import EDGES, GOAL_NODE, HEURISTIC, NODE_COORDS, START_NODE


# ── Adjacency list ────────────────────────────────────────────────────────────

def _build_adjacency() -> Dict[int, List[Tuple[int, float]]]:
    """Build a symmetric adjacency list from the EDGES definition."""
    adj: Dict[int, List[Tuple[int, float]]] = {n: [] for n in NODE_COORDS}
    for u, v, cost in EDGES:
        adj[u].append((v, cost))
        adj[v].append((u, cost))
    return adj


_ADJ = _build_adjacency()


# ── Heuristic functions ───────────────────────────────────────────────────────

def _euclidean(node: int, goal: int) -> float:
    """Straight-line (Euclidean) distance between node coordinates."""
    x1, y1 = NODE_COORDS[node]
    x2, y2 = NODE_COORDS[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _manhattan(node: int, goal: int) -> float:
    """Sum of absolute coordinate differences (Manhattan distance)."""
    x1, y1 = NODE_COORDS[node]
    x2, y2 = NODE_COORDS[goal]
    return float(abs(x2 - x1) + abs(y2 - y1))


def _heuristic(node: int, goal: int) -> float:
    """Dispatch to the heuristic selected in config.py."""
    if HEURISTIC == "manhattan":
        return _manhattan(node, goal)
    return _euclidean(node, goal)   # default


# ── Heap entry ────────────────────────────────────────────────────────────────

@dataclass(order=True)
class _Entry:
    """
    Min-heap entry.  Comparison is lexicographic on (f, h), which implements
    the required tie-breaking rule: equal f → prefer lower h.
    The node field is excluded from comparison to avoid tie-break ambiguity.
    """
    f: float
    h: float
    node: int = field(compare=False)


# ── A* search ────────────────────────────────────────────────────────────────

def run(
    start: int = START_NODE,
    goal: int = GOAL_NODE,
) -> Tuple[List[int], float]:
    """
    Run A* from *start* to *goal*.

    Parameters
    ----------
    start : int  – source node ID
    goal  : int  – target node ID

    Returns
    -------
    path : list[int]  – node sequence from start to goal (inclusive)
    cost : float      – total accumulated edge cost of the path

    Raises
    ------
    RuntimeError  – if no path exists between start and goal
    """
    # g[n] = best known cost to reach n from start
    g: Dict[int, float] = {start: 0.0}
    # parent[n] = predecessor of n on the best path found so far
    parent: Dict[int, Optional[int]] = {start: None}

    h_start = _heuristic(start, goal)
    open_heap: List[_Entry] = [_Entry(f=h_start, h=h_start, node=start)]
    closed: set = set()

    while open_heap:
        entry = heapq.heappop(open_heap)
        current = entry.node

        # A node may appear in the heap multiple times; skip stale entries.
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return _reconstruct_path(parent, goal), g[goal]

        for neighbour, edge_cost in _ADJ[current]:
            if neighbour in closed:
                continue

            tentative_g = g[current] + edge_cost
            if tentative_g < g.get(neighbour, math.inf):
                g[neighbour] = tentative_g
                parent[neighbour] = current
                h_n = _heuristic(neighbour, goal)
                heapq.heappush(
                    open_heap,
                    _Entry(f=tentative_g + h_n, h=h_n, node=neighbour),
                )

    raise RuntimeError(
        f"A*: no path found from N{start} to N{goal} "
        f"(heuristic={HEURISTIC!r})"
    )


def _reconstruct_path(
    parent: Dict[int, Optional[int]],
    goal: int,
) -> List[int]:
    """Walk the parent map backwards from goal to start."""
    path: List[int] = []
    node: Optional[int] = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


# ── Output helpers ────────────────────────────────────────────────────────────

def format_path(path: List[int]) -> str:
    """Return the path as a human-readable arrow sequence, e.g. N0 → N1 → …"""
    return " \u2192 ".join(f"N{n}" for n in path)


def print_result(path: List[int], cost: float) -> None:
    """Print the path sequence and total cost to stdout."""
    print(f"Heuristic : {HEURISTIC}")
    print(f"Path      : {format_path(path)}")
    print(f"Total cost: {cost:.4f}")
