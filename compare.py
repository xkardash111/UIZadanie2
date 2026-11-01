import csv
import os
import random
import statistics
import time
from functools import partial
from math import comb

import matplotlib.pyplot as plt

from main import (generate_cities, distance_matrix, generate_population, route_length, fitness,
                  genetic_algorithm, tournament, roulette, MAX_GENERATIONS)
from configs import (MAP_SEED, RUNS, SWEEP_CITIES, SELECTION_CITIES, BASELINE, GA_EXPERIMENTS,
                     PRESSURE_CITIES, PRESSURE_POPULATION, PRESSURE_K, PRESSURE_SAMPLES)

PLOTS_DIR = "plots"                              # tournament vs roulette + selection pressure plots
GA_PLOTS_DIR = os.path.join(PLOTS_DIR, "ga")     # parameter sweep plots
RESULTS_DIR = "results"                          # csv tables

# every selection method keeps the same color on all plots
SELECTION_COLORS = {"tournament": "#2a78d6", "roulette": "#eb6834"}
# ordered blue ramp for ordered values (k = 2 ... 20, small -> large population, ...)
VALUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MUTED = "#6b6a63"  # gray for helper lines (border between the better and the worse half)

_maps = {}   # city count -> (cities, distance matrix)
_cache = {}  # cache_key -> results of RUNS runs


# same seed -> the same map for a given city count; cached so every experiment uses identical cities
def make_map(count:int):
    if count not in _maps:
        random.seed(MAP_SEED)
        cities = generate_cities(count)
        _maps[count] = (cities, distance_matrix(cities))
    return _maps[count]


# picks evenly spaced shades from VALUE_RAMP, lightest for the smallest value
def value_colors(values:list):
    if len(values) == 1:
        return [VALUE_RAMP[2]]
    step = (len(VALUE_RAMP) - 1) / (len(values) - 1)
    return [VALUE_RAMP[round(i * step)] for i in range(len(values))]


def pad(history:list, length:int):
    # run stopped early -> its best stays the same until the end (elitism)
    return history + [history[-1]] * (length - len(history))


# arithmetic mean of a list
def mean(values:list):
    return sum(values) / len(values)


# averages the per-generation history of several runs into one curve
def mean_curve(results:list, key:str):
    return [mean(values) for values in zip(*(r[key] for r in results))]


# identifies one configuration; equal keys -> the runs are reused instead of repeated
def cache_key(count:int, selection_name:str, config:dict):
    config = dict(config)
    if selection_name != "tournament":
        config.pop("tournament_k")  # roulette has no k -> same runs for every k
    return count, selection_name, tuple(sorted(config.items()))


# runs the GA RUNS times (seeds 0..RUNS-1) with one configuration and collects stats of every run
def run_experiment(count:int, selection_name:str, config:dict):
    key = cache_key(count, selection_name, config)
    if key in _cache:
        return _cache[key]

    _, dist = make_map(count)
    if selection_name == "tournament":
        selection = partial(tournament, k=config["tournament_k"])
    else:
        selection = roulette

    results = []
    for seed in range(RUNS):
        random.seed(seed)
        start = time.perf_counter()
        best, length, best_history, avg_history = genetic_algorithm(
            dist, selection, config["population_size"], config["mutation_rate"], config["elite_count"])
        elapsed = time.perf_counter() - start
        results.append({
            "cities": count,
            "selection": selection_name,
            "config": dict(config),
            "seed": seed,
            "route": best,
            "length": length,
            "gen_of_best": best_history.index(min(best_history)) + 1,
            "generations": len(best_history),
            "time": elapsed,
            "best_history": pad(best_history, MAX_GENERATIONS),
            "avg_history": pad(avg_history, MAX_GENERATIONS),
        })
    _cache[key] = results
    return results


# which selection methods take part in a sweep (k exists only for the tournament)
def selections_for(param:str):
    return ["tournament"] if param == "tournament_k" else ["tournament", "roulette"]


# changes one parameter through its values, everything else stays BASELINE
def run_sweep(param:str, values:list):
    sweep = {}
    for value in values:
        for selection_name in selections_for(param):
            config = dict(BASELINE)
            count = SWEEP_CITIES
            if param == "cities":
                count = value
            else:
                config[param] = value
            print(f"{param}={value}, {selection_name}, {count} cities...")
            sweep[(selection_name, value)] = run_experiment(count, selection_name, config)
    return sweep


# shortest route found on this map over all runs of all experiments
def best_known(count:int):
    return min(r["length"] for results in _cache.values() for r in results if r["cities"] == count)


# one table row from several runs: length stats, gap to best known, generation of best, time, hits
def summarize(results:list):
    lengths = [r["length"] for r in results]
    known = best_known(results[0]["cities"])
    hits = sum(1 for l in lengths if l - known < 0.01)
    return {
        "avg": mean(lengths),
        "std": statistics.stdev(lengths),
        "best": min(lengths),
        "worst": max(lengths),
        "gap_%": (mean(lengths) / known - 1) * 100,
        "avg_gen": mean([r["gen_of_best"] for r in results]),
        "avg_time": mean([r["time"] for r in results]),
        "hit_best": f"{hits}/{len(results)}",
    }


# ---------- tables ----------

# writes results/ga_<param>.csv and prints the same table to the console
def write_sweep_table(param:str, sweep:dict):
    path = os.path.join(RESULTS_DIR, f"ga_{param}.csv")
    columns = [param, "selection", "cities", "avg", "std", "best", "worst", "gap_%", "avg_gen", "avg_time", "hit_best"]
    print(f"\n=== {param} ===")
    print(f"{param:<16}{'selection':<12}{'avg':>9}{'std':>8}{'best':>9}{'worst':>9}{'gap %':>7}"
          f"{'avg gen':>9}{'time':>8}{'hit':>7}")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for (selection_name, value), results in sweep.items():
            s = summarize(results)
            writer.writerow([value, selection_name, results[0]["cities"],
                             round(s["avg"], 2), round(s["std"], 2), round(s["best"], 2), round(s["worst"], 2),
                             round(s["gap_%"], 2), round(s["avg_gen"], 1), round(s["avg_time"], 3), s["hit_best"]])
            print(f"{str(value):<16}{selection_name:<12}{s['avg']:>9.2f}{s['std']:>8.2f}{s['best']:>9.2f}"
                  f"{s['worst']:>9.2f}{s['gap_%']:>7.2f}{s['avg_gen']:>9.1f}{s['avg_time']:>7.2f}s{s['hit_best']:>7}")


# writes results/ga_runs.csv, one row per GA run
def write_runs_table():
    path = os.path.join(RESULTS_DIR, "ga_runs.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cities", "selection", "population_size", "mutation_rate", "elite_count", "tournament_k",
                         "seed", "length", "gen_of_best", "generations", "time", "route"])
        for results in _cache.values():
            for r in results:
                c = r["config"]
                k = c["tournament_k"] if r["selection"] == "tournament" else ""
                writer.writerow([r["cities"], r["selection"], c["population_size"], c["mutation_rate"],
                                 c["elite_count"], k, r["seed"], round(r["length"], 2), r["gen_of_best"],
                                 r["generations"], round(r["time"], 3), " ".join(map(str, r["route"]))])


# ---------- plots ----------

# common axis look: labels, light grid, no top/right frame
def style(ax, xlabel:str, ylabel:str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# mean final length +- std for every value of the parameter, one line per selection
def draw_sweep_mean(ax, param:str, values:list, sweep:dict):
    for selection_name in selections_for(param):
        summaries = [summarize(sweep[(selection_name, v)]) for v in values]
        ax.errorbar(range(len(values)), [s["avg"] for s in summaries], yerr=[s["std"] for s in summaries],
                    marker="o", markersize=7, linewidth=2, capsize=4,
                    color=SELECTION_COLORS[selection_name], label=selection_name)
    # evenly spaced positions labelled with the real values -> no 2.5, 7.5 ... ticks
    ax.set_xticks(range(len(values)), [str(v) for v in values])
    ax.set_title(f"Final route length vs {param} ({SWEEP_CITIES} cities, mean ± std of {RUNS} runs)")
    style(ax, param, "route length")
    ax.legend()


# tournament size: final quality (left) and how fast it converges (right)
def plot_tournament_k(values:list, sweep:dict):
    fig, (ax_mean, ax_conv) = plt.subplots(1, 2, figsize=(14, 5))
    draw_sweep_mean(ax_mean, "tournament_k", values, sweep)
    for value, color in zip(values, value_colors(values)):
        ax_conv.plot(mean_curve(sweep[("tournament", value)], "best_history"),
                     color=color, linewidth=2, label=f"k={value}")
    ax_conv.set_title(f"Best length per generation (mean of {RUNS} runs)")
    style(ax_conv, "generation", "route length")
    ax_conv.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GA_PLOTS_DIR, "tournament_k.png"), dpi=150)
    plt.close(fig)


# elitism: roulette depends on it, tournament barely
def plot_elite_count(values:list, sweep:dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_sweep_mean(ax, "elite_count", values, sweep)
    fig.tight_layout()
    fig.savefig(os.path.join(GA_PLOTS_DIR, "elite_count_mean.png"), dpi=150)
    plt.close(fig)


# ---------- tournament vs roulette (baseline) ----------

# best and average population length per generation, tournament vs roulette
def plot_convergence(count:int, all_results:dict):
    fig, (ax_best, ax_avg) = plt.subplots(1, 2, figsize=(13, 5))
    for name, results in all_results.items():
        ax_best.plot(mean_curve(results, "best_history"), label=name, color=SELECTION_COLORS[name], linewidth=2)
        ax_avg.plot(mean_curve(results, "avg_history"), label=name, color=SELECTION_COLORS[name], linewidth=2)
    ax_best.set_title(f"Best length per generation ({count} cities, mean of {RUNS} runs)")
    ax_avg.set_title(f"Average population length ({count} cities, mean of {RUNS} runs)")
    for ax in (ax_best, ax_avg):
        style(ax, "generation", "route length")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"convergence_{count}.png"), dpi=150)
    plt.close(fig)


# best route found by each selection drawn on the map
def plot_routes(count:int, cities:list, all_results:dict):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 6))
    for ax, (name, results) in zip(axes, all_results.items()):
        best = min(results, key=lambda r: r["length"])
        route = best["route"] + [best["route"][0]]
        ax.plot([cities[c][0] for c in route], [cities[c][1] for c in route], "-", color="gray")
        ax.scatter([x for x, y in cities], [y for x, y in cities], color="red", zorder=3)
        for i, (x, y) in enumerate(cities):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(f"{name}: {best['length']:.2f}")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"routes_{count}.png"), dpi=150)
    plt.close(fig)


# baseline tournament vs roulette on SELECTION_CITIES (runs come from the cache)
def compare_selections():
    cities, _ = make_map(SELECTION_CITIES)
    all_results = {name: run_experiment(SELECTION_CITIES, name, BASELINE) for name in ("tournament", "roulette")}
    plot_convergence(SELECTION_CITIES, all_results)
    plot_routes(SELECTION_CITIES, cities, all_results)


# ---------- selection pressure ----------

# picks parents from one sorted initial population many times and counts how often each rank wins
def selection_pressure():
    _, dist = make_map(PRESSURE_CITIES)
    random.seed(MAP_SEED)
    population = generate_population(PRESSURE_CITIES, PRESSURE_POPULATION)
    population.sort(key=lambda ind: route_length(ind, dist))  # index = rank, 0 is the best
    fitnesses = [fitness(ind, dist) for ind in population]
    rank_of = {id(ind): rank for rank, ind in enumerate(population)}
    n = len(population)

    methods = {f"tournament k={k}": partial(tournament, k=k) for k in PRESSURE_K}
    methods["roulette"] = roulette

    # rank r wins a tournament (sampled without replacement) <=> it is in the sample and all others are worse
    theory = {f"tournament k={k}": [comb(n - 1 - r, k - 1) / comb(n, k) for r in range(n)] for k in PRESSURE_K}
    theory["roulette"] = [f / sum(fitnesses) for f in fitnesses]

    measured = {}
    for name, selection in methods.items():
        print(f"selection pressure: {name}...")
        counts = [0] * n
        for _ in range(PRESSURE_SAMPLES):
            counts[rank_of[id(selection(population, fitnesses))]] += 1
        measured[name] = [c / PRESSURE_SAMPLES for c in counts]

    with open(os.path.join(RESULTS_DIR, "selection_pressure_rank.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank"] + [f"{name} {kind}" for name in methods for kind in ("measured", "theory")])
        for r in range(n):
            writer.writerow([r + 1] + [round(v, 6) for name in methods for v in (measured[name][r], theory[name][r])])

    weak = n // 2
    weak_share = {name: (sum(measured[name][weak:]), sum(theory[name][weak:])) for name in methods}
    print(f"\n=== parents picked from the worse half (ranks {weak + 1}-{n}) ===")
    with open(os.path.join(RESULTS_DIR, "selection_pressure_weak_half.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "measured_%", "theory_%"])
        for name, (m, t) in weak_share.items():
            writer.writerow([name, round(m * 100, 3), round(t * 100, 3)])
            print(f"{name:<18}{m * 100:>8.3f}%  (theory {t * 100:.3f}%)")

    plot_pressure_rank(n, measured, theory)
    plot_pressure_weak_half(n, weak_share)


# tournaments get blue shades by k, roulette keeps its orange
def pressure_colors(names:list):
    tournament_names = [name for name in names if name.startswith("tournament")]
    colors = dict(zip(tournament_names, value_colors(tournament_names)))
    colors["roulette"] = SELECTION_COLORS["roulette"]
    return colors


# probability of being picked by rank, linear and log scale, measured vs theory
def plot_pressure_rank(n:int, measured:dict, theory:dict):
    colors = pressure_colors(list(measured))
    ranks = range(1, n + 1)
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in (ax_lin, ax_log):
        for name in measured:
            # never picked -> no point on the log scale instead of a drop to zero
            values = [p if p > 0 else float("nan") for p in measured[name]] if ax is ax_log else measured[name]
            ax.plot(ranks, values, color=colors[name], linewidth=2, label=name)
            ax.plot(ranks, theory[name], color=colors[name], linewidth=1, linestyle="--")
        ax.axvline(n / 2 + 0.5, color=MUTED, linewidth=1, linestyle=":")
        style(ax, "rank in population (1 = best route)", "probability of being picked as parent")
    ax_log.set_yscale("log")
    ax_log.set_ylim(bottom=0.1 / PRESSURE_SAMPLES)  # below one pick per sample run nothing can be measured
    ax_lin.set_title(f"Parent selection probability by rank (population {n})")
    ax_log.set_title("Same, log scale (solid = measured, dashed = theory)")
    ax_lin.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "selection_pressure_rank.png"), dpi=150)
    plt.close(fig)


# share of parents coming from the worse half of the population, measured vs theory
def plot_pressure_weak_half(n:int, weak_share:dict):
    colors = pressure_colors(list(weak_share))
    names = list(weak_share)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, [weak_share[name][0] * 100 for name in names],
                  color=[colors[name] for name in names], width=0.6)
    ax.scatter(names, [weak_share[name][1] * 100 for name in names], color="black", marker="_", s=400,
               zorder=3, label="theory")
    for bar, name in zip(bars, names):
        ax.annotate(f"{weak_share[name][0] * 100:.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_title(f"Parents picked from the worse half of the population (ranks {n // 2 + 1}-{n})")
    style(ax, "selection method", "share of parents [%]")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "selection_pressure_weak_half.png"), dpi=150)
    plt.close(fig)


# selection pressure -> parameter sweeps -> tournament vs roulette -> tables and plots
def main():
    os.makedirs(GA_PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    selection_pressure()

    sweeps = {param: run_sweep(param, values) for param, values in GA_EXPERIMENTS.items()}
    compare_selections()

    # tables after all runs, so "best known" uses every run on that map
    for param in GA_EXPERIMENTS:
        write_sweep_table(param, sweeps[param])
    plot_tournament_k(GA_EXPERIMENTS["tournament_k"], sweeps["tournament_k"])
    plot_elite_count(GA_EXPERIMENTS["elite_count"], sweeps["elite_count"])
    write_runs_table()
    print(f"\nPlots saved to '{PLOTS_DIR}/', tables to '{RESULTS_DIR}/'")


if __name__ == '__main__':
    main()
