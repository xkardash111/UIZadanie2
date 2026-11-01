MAP_SEED = 42
RUNS = 5

# parameter sweeps are run on this map, one parameter changes, the rest stays BASELINE
SWEEP_CITIES = 30
# map used for the tournament vs roulette plots (routes, convergence)
SELECTION_CITIES = 30

BASELINE = {
    "population_size": 100,
    "mutation_rate": 0.3,
    "elite_count": 2,
    "tournament_k": 3,
}

GA_EXPERIMENTS = {
    "cities": [20, 25, 30, 35, 40],
    "tournament_k": [2, 3, 5, 10, 20],
    "population_size": [20, 50, 100, 200],
    "mutation_rate": [0.05, 0.1, 0.3, 0.5, 0.8],
    "elite_count": [0, 1, 2, 5, 10],
}

# selection pressure: how often each rank of the initial population is picked as a parent
PRESSURE_CITIES = 30
PRESSURE_POPULATION = 100
PRESSURE_K = [2, 3, 5, 10, 20]
PRESSURE_SAMPLES = 100_000

SA_CITY_COUNTS = [20, 30, 40]
# map used for the schedule plot (temperature, accepted moves, one run)
SA_SCHEDULE_CITIES = 30

SA_BASELINE = {
    "t0": 300,
    "alpha": 0.95,
    "steps_per_t": 100,
    "t_min": 0.1,
}

SA_EXPERIMENTS = {
    "alpha": [0.8, 0.9, 0.95, 0.99, 0.995],
    "t0": [10, 50, 300, 1000, 5000],
    "steps_per_t": [10, 50, 100, 500, 1000],
}
