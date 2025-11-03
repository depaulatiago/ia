from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import random

# Representação de estado como tupla de 9 inteiros (0 = branco)
State = Tuple[int, ...]
GOAL: State = (1,2,3,4,5,6,7,8,0)

MOVES = {
    'U': -3,  # up
    'D':  3,  # down
    'L': -1,  # left
    'R':  1,  # right
}

def index_to_rc(i: int) -> Tuple[int, int]:
    return divmod(i, 3)  # (row, col)

def rc_to_index(r: int, c: int) -> int:
    return r * 3 + c

def can_move(blank_idx: int, move: str) -> bool:
    r, c = index_to_rc(blank_idx)
    if move == 'U': return r > 0
    if move == 'D': return r < 2
    if move == 'L': return c > 0
    if move == 'R': return c < 2
    return False

def apply_move(state: State, move: str) -> State:
    """Move o branco (0) na direção dada e retorna novo estado."""
    blank = state.index(0)
    if not can_move(blank, move):
        return state
    r, c = index_to_rc(blank)
    if move == 'U': r -= 1
    elif move == 'D': r += 1
    elif move == 'L': c -= 1
    elif move == 'R': c += 1
    swap_idx = rc_to_index(r, c)
    s = list(state)
    s[blank], s[swap_idx] = s[swap_idx], s[blank]
    return tuple(s)

def neighbors(state: State) -> Iterable[Tuple[str, State]]:
    blank = state.index(0)
    for m in ('U','D','L','R'):
        if can_move(blank, m):
            yield m, apply_move(state, m)

def is_goal(state: State) -> bool:
    return state == GOAL

def inversions_count(state: State) -> int:
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def is_solvable(state: State) -> bool:
    # Para 8-puzzle (tabuleiro 3x3), número de inversões deve ser PAR
    return inversions_count(state) % 2 == 0

def shuffle_from_goal(steps: int = 50) -> State:
    """Embaralha aplicando 'steps' movimentos válidos a partir do objetivo (mantém solução garantida)."""
    state = GOAL
    last_move = None
    opposite = {'U':'D','D':'U','L':'R','R':'L'}
    for _ in range(steps):
        possible = [m for m,_ in neighbors(state) if m != opposite.get(last_move)]
        m = random.choice(possible)
        state = apply_move(state, m)
        last_move = m
    return state

def pretty(state: State) -> str:
    s = []
    for i, v in enumerate(state):
        s.append(' ' if v == 0 else str(v))
        if i % 3 == 2:
            s.append('\n')
        else:
            s.append(' ')
    return ''.join(s)
