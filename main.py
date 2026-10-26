from math import sqrt
import random

MAP_SIZE = 200
SEED = 42
random.seed(SEED)

POPULATION_SIZE = 100
MAX_GENERATIONS = 1000
PATIENCE = 200
MUTATION_RATE = 0.3
ELITE_COUNT = 2


def euclidean(x:tuple, y:tuple):
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def mutate(ind:list, count:int):
    individual = list(ind)
    l, r = sorted(random.sample(range(len(individual)), 2))
    opa = []
    for i in range(l, r+1):
        opa.append(individual[i])
    opa.reverse()
    for i in range(len(opa)):
        individual[l+i] = opa[i]
    return individual


def crossover(ind1:list, ind2:list, count:int):
    individual = list(ind1)
    individual1 = list(ind2)
    l, r = sorted(random.sample(range(len(individual)), 2))
    new_individual = [0] * len(individual)
    s = set()
    for i in range(l, r):
        new_individual[i] = individual[i]
        s.add(individual[i])
    insert_in = 0
    for i in range(count):
        if individual1[i] not in s:
            if insert_in >= l and insert_in <= r:
                insert_in = r
            new_individual[insert_in] = individual1[i]
            s.add(individual1[i])
            insert_in += 1
    return new_individual






def generate_population(count:int, size:int = 100):
    population = []
    for i in range(size):
        individual = list(range(count))
        random.shuffle(individual)
        population.append(individual)
    return population



def route_length(individual:list, dist:list):
    n = len(individual)
    segments = [dist[individual[i]][individual[(i+1) % n]] for i in range(n)]
    return sum(segments)


def fitness(individual:list, dist:list):
    return 1/route_length(individual, dist)


def tournament(population:list, fitnesses:list, k:int = 3):
    candidates = random.sample(range(len(population)), k)
    best = candidates[0]
    for candidate in candidates[1:]:
        if fitnesses[candidate] > fitnesses[best]:
            best = candidate
    return population[best]


def genetic_algorithm(dist:list, selection = tournament):
    count = len(dist)
    population = generate_population(count, POPULATION_SIZE)
    best_individual = min(population, key=lambda ind: route_length(ind, dist))
    best_length = route_length(best_individual, dist)
    best_history = []
    avg_history = []
    no_improve = 0

    for generation in range(MAX_GENERATIONS):
        population.sort(key=lambda ind: route_length(ind, dist))
        new_population = population[:ELITE_COUNT]
        fitnesses = [fitness(ind, dist) for ind in population]

        while len(new_population) < POPULATION_SIZE:
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            child = crossover(p1, p2, count)
            if random.random() < MUTATION_RATE:
                child = mutate(child, count)
            new_population.append(child)

        population = new_population

        lengths = [route_length(ind, dist) for ind in population]
        generation_best = min(lengths)
        best_history.append(generation_best)
        avg_history.append(sum(lengths) / len(lengths))

        if generation_best < best_length:
            best_length = generation_best
            best_individual = population[lengths.index(generation_best)]
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    return best_individual, best_length, best_history, avg_history


def roulette(population:list, fitnesses:list):
    total = sum(fitnesses)
    pick = random.uniform(0, total)
    sum_fitness = 0
    for individual, fit in zip(population, fitnesses):
        sum_fitness += fit
        if sum_fitness >= pick:
            return individual
    return population[-1]


def distance_matrix(cities:list):
    count = len(cities)
    dist = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            d = euclidean(cities[i], cities[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def generate_cities(count:int):
    if count < 20 or count > 40:
        raise ValueError("count must be between 20 and 40")
    cities = []
    for i in range(count):
        x = random.randrange(0, MAP_SIZE)
        y = random.randrange(0, MAP_SIZE)
        while (x,y) in cities:
            x = random.randrange(0, MAP_SIZE)
            y = random.randrange(0, MAP_SIZE)
        cities.append((x,y))
    return cities


def main():
    cities = generate_cities(20)
    dist = distance_matrix(cities)
    print(cities)

    best, length, best_history, avg_history = genetic_algorithm(dist, tournament)
    print("Generations:", len(best_history))
    print("Route:", best)
    print("Length:", round(length, 2))
    return

if __name__ == '__main__':
    main()
