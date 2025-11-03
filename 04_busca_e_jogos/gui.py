import tkinter as tk
from tkinter import messagebox
from typing import List, Optional
import time

from puzzle import State, GOAL, shuffle_from_goal, apply_move, is_solvable, pretty
from search import bfs, astar

TILE_FONT = ("Segoe UI", 20, "bold")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Jogo dos Oito – GCC128")
        self.resizable(False, False)

        self.state: State = GOAL
        self.solution: List[str] = []
        self.step_index = 0
        self.is_auto_playing = False

        self.build_ui()
        self.update_board()

    def build_ui(self):
        self.board_frame = tk.Frame(self, padx=10, pady=10)
        self.board_frame.grid(row=0, column=0, columnspan=4)

        self.tiles = []
        for r in range(3):
            row = []
            for c in range(3):
                b = tk.Button(self.board_frame, text="", width=4, height=2, font=TILE_FONT, 
                              command=lambda rr=r, cc=c: self.click_tile(rr, cc))
                b.grid(row=r, column=c, padx=4, pady=4)
                row.append(b)
            self.tiles.append(row)

        self.btn_shuffle = tk.Button(self, text="Embaralhar", command=self.shuffle)
        self.btn_bfs = tk.Button(self, text="Resolver (BFS)", command=self.solve_bfs)
        self.btn_a1 = tk.Button(self, text="Resolver (A* Misplaced)", command=lambda: self.solve_astar('misplaced'))
        self.btn_a2 = tk.Button(self, text="Resolver (A* Manhattan)", command=lambda: self.solve_astar('manhattan'))
        self.btn_autoplay = tk.Button(self, text="Auto‑Play", command=self.autoplay)
        self.btn_step = tk.Button(self, text="Próximo Passo", command=self.next_step)

        self.btn_shuffle.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        self.btn_bfs.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        self.btn_a1.grid(row=1, column=2, sticky="ew", padx=6, pady=4)
        self.btn_a2.grid(row=1, column=3, sticky="ew", padx=6, pady=4)
        self.btn_autoplay.grid(row=2, column=2, sticky="ew", padx=6, pady=4)
        self.btn_step.grid(row=2, column=3, sticky="ew", padx=6, pady=4)

        self.metrics = tk.StringVar(value="Pronto.")
        self.lbl_metrics = tk.Label(self, textvariable=self.metrics, justify="left", anchor="w")
        self.lbl_metrics.grid(row=2, column=0, columnspan=2, sticky="w", padx=10)

        # Bindings para jogadas manuais
        self.bind("<Up>", lambda e: self.try_move('U'))
        self.bind("<Down>", lambda e: self.try_move('D'))
        self.bind("<Left>", lambda e: self.try_move('L'))
        self.bind("<Right>", lambda e: self.try_move('R'))
        self.bind("w", lambda e: self.try_move('U'))
        self.bind("s", lambda e: self.try_move('D'))
        self.bind("a", lambda e: self.try_move('L'))
        self.bind("d", lambda e: self.try_move('R'))

    def update_board(self):
        for i, v in enumerate(self.state):
            r, c = divmod(i, 3)
            btn = self.tiles[r][c]
            if v == 0:
                btn.config(text="", state="disabled", bg="#eee")
            else:
                btn.config(text=str(v), state="normal", bg="#f5f5f5")

        if self.state == GOAL:
            self.metrics.set("Objetivo alcançado!")

    def click_tile(self, rr: int, cc: int):
        # Permite clicar em um vizinho do branco para mover
        blank = self.state.index(0)
        br, bc = divmod(blank, 3)
        if (abs(br - rr) + abs(bc - cc)) == 1:
            # Determinar direção relativa
            if rr == br and cc == bc - 1: move = 'L'
            elif rr == br and cc == bc + 1: move = 'R'
            elif cc == bc and rr == br - 1: move = 'U'
            elif cc == bc and rr == br + 1: move = 'D'
            else: return
            self.state = apply_move(self.state, move)
            self.update_board()

    def shuffle(self):
        self.state = shuffle_from_goal(steps=60)
        self.solution = []
        self.step_index = 0
        self.is_auto_playing = False
        self.metrics.set("Estado embaralhado.")
        self.update_board()

    def solve_bfs(self):
        self.solve_generic('bfs')

    def solve_astar(self, heuristic: str):
        self.solve_generic('astar', heuristic)

    def solve_generic(self, mode: str, heuristic: Optional[str] = None):
        if not is_solvable(self.state):
            messagebox.showwarning("Não solucionável", "Esse estado não é solucionável. Embaralhe novamente.")
            return
        self.metrics.set("Resolvendo...")
        self.update_idletasks()
        if mode == 'bfs':
            res = bfs(self.state)
        else:
            res = astar(self.state, heuristic=heuristic or 'manhattan')
        if res.cost == -1:
            self.metrics.set("Solução não encontrada.")
            self.solution = []
            self.step_index = 0
            return
        self.metrics.set(f"Algoritmo: {mode.upper() if mode=='bfs' else 'A* ' + (heuristic or '')} | "
                         f"Custo: {res.cost} | Expandidos: {res.expanded} | "
                         f"Profundidade máx.: {res.max_depth} | Tempo: {res.elapsed:.3f}s")
        # Aplicar solução ao estado meta passo a passo
        self.solution = res.path
        self.step_index = 0

    def next_step(self):
        if self.step_index < len(self.solution):
            m = self.solution[self.step_index]
            self.state = apply_move(self.state, m)
            self.step_index += 1
            self.update_board()
        else:
            self.metrics.set("Fim da sequência/objetivo atingido.")

    def autoplay(self):
        if not self.solution:
            self.metrics.set("Não há sequência calculada. Use 'Resolver' primeiro.")
            return
        self.is_auto_playing = True
        self._autoplay_tick()

    def _autoplay_tick(self):
        if not self.is_auto_playing:
            return
        if self.step_index < len(self.solution):
            self.next_step()
            # Ajuste de velocidade aqui (em ms)
            self.after(250, self._autoplay_tick)
        else:
            self.is_auto_playing = False
            self.metrics.set("Animação concluída.")

    def try_move(self, move: str):
        # Jogadas manuais
        new_state = apply_move(self.state, move)
        if new_state != self.state:
            self.state = new_state
            self.update_board()

if __name__ == "__main__":
    app = App()
    app.mainloop()
