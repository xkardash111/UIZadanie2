import random
from main import mutate, generate_cities, generate_population, route_length, distance_matrix
import math

T0 = 300
T_MIN = 0.1
ALPHA = 0.95
STEPS_PER_T = 100



def simulated_annealing(dist:list):
    current = generate_population(len(dist), 1)[0]
    current_length = route_length(current, dist)
    best, best_length = current, current_length
    T = T0

    while T > T_MIN:
        for _ in range(STEPS_PER_T):
            neighbor = mutate(current)
            neighbor_length = route_length(neighbor, dist)
            delta = neighbor_length - current_length

            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_length = neighbor, neighbor_length
                if current_length < best_length:
                    best, best_length = current, current_length
        T *= ALPHA

    return best, best_length




def main():
    cities = generate_cities(20)
    dist = distance_matrix(cities)
    best, length = simulated_annealing(dist)
    print("Route:", best)
    print("Length:", round(length, 2))


if __name__ == '__main__':
    main()
