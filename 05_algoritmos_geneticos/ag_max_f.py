"""
GCC128 - Inteligência Artificial
Trabalho Prático 05 - Algoritmos Genéticos

Objetivo:
    Maximizar f(x) = x^2 - 3x + 4 no intervalo [-10, 10]

Data: 02/12/2025
"""

import random
from typing import List, Tuple

# Parâmetros ajustáveis
POPULATION_SIZE = 4
NUM_GENERATIONS = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 2
NUM_BITS = 16

X_MIN = -10.0
X_MAX = 10.0


def objective_function(x: float) -> float:
    return x ** 2 - 3 * x + 4


def random_chromosome(num_bits: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(num_bits)]


def decode(chromosome: List[int], x_min: float, x_max: float) -> float:
    bit_string = ''.join(str(bit) for bit in chromosome)
    integer_value = int(bit_string, 2)
    max_int = 2 ** len(chromosome) - 1
    return x_min + (x_max - x_min) * (integer_value / max_int)


def evaluate_population(population: List[List[int]]) -> Tuple[List[float], float, List[int], float]:
    fitness_values = []
    best_fitness = float('-inf')
    best_chromosome = None
    best_x = None

    for chrom in population:
        x = decode(chrom, X_MIN, X_MAX)
        fitness = objective_function(x)
        fitness_values.append(fitness)

        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = chrom
            best_x = x

    return fitness_values, best_fitness, best_chromosome, best_x


def tournament_selection(population: List[List[int]], fitness_values: List[float], size: int) -> List[int]:
    best_i = random.randrange(len(population))
    for _ in range(size - 1):
        i = random.randrange(len(population))
        if fitness_values[i] > fitness_values[best_i]:
            best_i = i
    return population[best_i][:]


def single_point_crossover(p1: List[int], p2: List[int], rate: float):
    if random.random() > rate:
        return p1[:], p2[:]

    point = random.randint(1, len(p1) - 1)
    return (
        p1[:point] + p2[point:],
        p2[:point] + p1[point:]
    )


def mutate(chrom: List[int], rate: float) -> None:
    for i in range(len(chrom)):
        if random.random() < rate:
            chrom[i] = 1 - chrom[i]


def genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, num_bits, tournament_size):
    population = [random_chromosome(num_bits) for _ in range(pop_size)]

    print("==============================")
    print("Iniciando Algoritmo Genético")
    print(f"População: {pop_size}")
    print(f"Gerações: {generations}")
    print("==============================\n")

    fitness_values, best_fitness, best_chrom, best_x = evaluate_population(population)
    print(f"Geração 0: x = {best_x:.5f}, f(x) = {best_fitness:.5f}")

    for gen in range(1, generations + 1):
        new_pop = [best_chrom[:]]

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness_values, tournament_size)
            p2 = tournament_selection(population, fitness_values, tournament_size)

            c1, c2 = single_point_crossover(p1, p2, crossover_rate)

            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitness_values, curr_best_fitness, curr_best_chrom, curr_best_x = evaluate_population(population)

        if curr_best_fitness > best_fitness:
            best_fitness = curr_best_fitness
            best_chrom = curr_best_chrom
            best_x = curr_best_x

        print(f"Geração {gen}: x = {best_x:.5f}, f(x) = {best_fitness:.5f}")

    print("\n==============================")
    print("Fim da execução")
    print(f"Melhor x encontrado: {best_x:.5f}")
    print(f"f(x) máximo aproximado: {best_fitness:.5f}")
    print("==============================")


if __name__ == "__main__":
    genetic_algorithm(
        pop_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        num_bits=NUM_BITS,
        tournament_size=TOURNAMENT_SIZE,
    )