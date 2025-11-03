from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List, Set
from collections import deque
import heapq
import time

from puzzle import State, GOAL, neighbors, is_goal

@dataclass(order=True)
class PrioritizedNode:
    f: int
    g: int = field(compare=False)
    state: State = field(compare=False)
    parent: Optional['PrioritizedNode'] = field(compare=False, default=None)
    action: Optional[str] = field(compare=False, default=None)

@dataclass
class SearchResult:
    path: List[str]
    cost: int
    expanded: int
    max_depth: int
    elapsed: float

# ---------- Heurísticas ----------
def h_misplaced(state: State) -> int:
    return sum(1 for i, v in enumerate(state) if v != 0 and v != i+1)

def h_manhattan(state: State) -> int:
    dist = 0
    for i, v in enumerate(state):
        if v == 0: 
            continue
        goal_i = v - 1
        r, c = divmod(i, 3)
        gr, gc = divmod(goal_i, 3)
        dist += abs(r - gr) + abs(c - gc)
    return dist

# ---------- BFS (não informada) ----------
def bfs(start: State) -> SearchResult:
    t0 = time.time()
    if is_goal(start):
        return SearchResult([], 0, 0, 0, 0.0)
    frontier = deque([start])
    parents: Dict[State, Tuple[Optional[State], Optional[str]]] = {start: (None, None)}
    expanded = 0
    max_depth = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        # Reconstruir profundidade
        depth = 0
        s = state
        while True:
            p, _ = parents[s]
            if p is None: break
            depth += 1
            s = p
        max_depth = max(max_depth, depth)
        for a, nb in neighbors(state):
            if nb not in parents:
                parents[nb] = (state, a)
                if is_goal(nb):
                    path = _reconstruct_path(parents, nb)
                    dt = time.time() - t0
                    return SearchResult(path, len(path), expanded, max_depth, dt)
                frontier.append(nb)
    dt = time.time() - t0
    return SearchResult([], -1, expanded, max_depth, dt)  # não encontrado

def _reconstruct_path(parents: Dict[State, Tuple[Optional[State], Optional[str]]], goal_state: State) -> List[str]:
    actions: List[str] = []
    s = goal_state
    while True:
        p, a = parents[s]
        if p is None:
            break
        actions.append(a)
        s = p
    actions.reverse()
    return actions

# ---------- A* (informada) ----------
def astar(start: State, heuristic: str = "manhattan") -> SearchResult:
    hfun = h_manhattan if heuristic.lower().startswith('manh') else h_misplaced
    t0 = time.time()
    if is_goal(start):
        return SearchResult([], 0, 0, 0, 0.0)
    open_heap: List[PrioritizedNode] = []
    g: Dict[State, int] = {start: 0}
    start_node = PrioritizedNode(f=hfun(start), g=0, state=start, parent=None, action=None)
    heapq.heappush(open_heap, start_node)
    closed: Set[State] = set()
    parent_map: Dict[State, Tuple[Optional[State], Optional[str]]] = {start: (None, None)}
    expanded = 0
    max_depth = 0

    while open_heap:
        node = heapq.heappop(open_heap)
        if node.state in closed:
            continue
        closed.add(node.state)
        expanded += 1

        if node.g > max_depth:
            max_depth = node.g

        if is_goal(node.state):
            path = _reconstruct_path(parent_map, node.state)
            dt = time.time() - t0
            return SearchResult(path, len(path), expanded, max_depth, dt)

        for a, nb in neighbors(node.state):
            tentative_g = node.g + 1
            if nb in closed and tentative_g >= g.get(nb, float('inf')):
                continue
            if tentative_g < g.get(nb, float('inf')):
                g[nb] = tentative_g
                parent_map[nb] = (node.state, a)
                f = tentative_g + hfun(nb)
                heapq.heappush(open_heap, PrioritizedNode(f=f, g=tentative_g, state=nb))
    dt = time.time() - t0
    return SearchResult([], -1, expanded, max_depth, dt)
