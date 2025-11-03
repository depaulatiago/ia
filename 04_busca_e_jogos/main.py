import argparse
from typing import List, Tuple
from puzzle import GOAL, shuffle_from_goal, is_solvable, pretty, State
from search import bfs, astar

def parse_state(s: str) -> State:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 9 or set(parts) != set(range(9)):
        raise ValueError("Estado deve conter os números 0..8 exatamente uma vez (0 = vazio).")
    return tuple(parts)  # type: ignore

def main():
    ap = argparse.ArgumentParser(description="Jogo dos Oito – GCC128 (BFS e A*)")
    ap.add_argument("--mode", choices=["bfs", "astar"], default="astar")
    ap.add_argument("--heuristic", choices=["misplaced", "manhattan"], default="manhattan")
    ap.add_argument("--start", type=str, help="Lista de 9 números separados por vírgula (0=branco)")
    ap.add_argument("--shuffle", type=int, default=0, help="Embaralhar N passos a partir do objetivo")
    args = ap.parse_args()

    if args.start:
        start = parse_state(args.start)
    elif args.shuffle > 0:
        start = shuffle_from_goal(args.shuffle)
    else:
        start = (1,2,3,4,5,6,0,7,8)  # exemplo simples

    print("Estado inicial:")
    print(pretty(start))

    if not is_solvable(start):
        print(">> Este estado não é solucionável. Tente outro (--shuffle N).")
        return

    if args.mode == "bfs":
        res = bfs(start)
    else:
        res = astar(start, heuristic=args.heuristic)

    if res.cost == -1:
        print("Solução não encontrada.")
        return

    print(f"\nCaminho de ações: {''.join(res.path)}")
    print(f"Custo: {res.cost}")
    print(f"Nós expandidos: {res.expanded}")
    print(f"Profundidade máx.: {res.max_depth}")
    print(f"Tempo: {res.elapsed:.3f}s")

if __name__ == "__main__":
    main()
