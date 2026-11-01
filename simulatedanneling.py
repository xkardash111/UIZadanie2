import random
from main import mutate, generate_cities, generate_population, route_length, distance_matrix
import math

T0 = 300
T_MIN = 0.1
ALPHA = 0.95
STEPS_PER_T = 100



def simulated_annealing(dist:list, t0:float = T0, alpha:float = ALPHA, steps_per_t:int = STEPS_PER_T, t_min:float = T_MIN):
    current = generate_population(len(dist), 1)[0]
    current_length = route_length(current, dist)
    best, best_length = current, current_length
    T = t0
    history = {"T": [], "current": [], "best": [], "acceptance": []}

    while T > t_min:
        accepted = 0
        for _ in range(steps_per_t):
            neighbor = mutate(current)
            neighbor_length = route_length(neighbor, dist)
            delta = neighbor_length - current_length

            if delta < 0 or random.random() < math.exp(-delta / T):
                current, current_length = neighbor, neighbor_length
                accepted += 1
                if current_length < best_length:
                    best, best_length = current, current_length

        history["T"].append(T)
        history["current"].append(current_length)
        history["best"].append(best_length)
        history["acceptance"].append(accepted / steps_per_t)
        T *= alpha
    return best, best_length, history




def main():
    cities = generate_cities(20)
    dist = distance_matrix(cities)
    best, length, history = simulated_annealing(dist)
    print("Temperature levels:", len(history["T"]))
    print("Route:", best)
    print("Length:", round(length, 2))


if __name__ == '__main__':
    main()
