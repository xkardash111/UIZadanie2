import os
import random
import time

import matplotlib.pyplot as plt

from main import generate_cities, distance_matrix, genetic_algorithm, tournament, roulette, MAX_GENERATIONS

MAP_SEED = 42
CITY_COUNTS = [20, 30]
RUNS = 10
METHODS = {"tournament": tournament, "roulette": roulette}
PLOTS_DIR = "plots"


def make_map(count:int):
    random.seed(MAP_SEED)
    cities = generate_cities(count)
    return cities, distance_matrix(cities)


def pad(history:list, length:int):
    # run stopped early -> its best stays the same until the end (elitism)
    return history + [history[-1]] * (length - len(history))


def run_experiment(dist:list, selection):
    results = []
    for seed in range(RUNS):
        random.seed(seed)
        start = time.perf_counter()
        best, length, best_history, avg_history = genetic_algorithm(dist, selection)
        elapsed = time.perf_counter() - start
        results.append({
            "route": best,
            "length": length,
            "gen_of_best": best_history.index(min(best_history)) + 1,
            "time": elapsed,
            "best_history": pad(best_history, MAX_GENERATIONS),
            "avg_history": pad(avg_history, MAX_GENERATIONS),
        })
    return results


def mean(values:list):
    return sum(values) / len(values)


def print_table(count:int, all_results:dict):
    best_known = min(r["length"] for results in all_results.values() for r in results)
    print(f"\n=== {count} cities (best known length: {best_known:.2f}) ===")
    print(f"{'method':<12}{'avg':>10}{'best':>10}{'worst':>10}{'avg gen':>10}{'avg time':>10}{'hit best':>10}")
    for name, results in all_results.items():
        lengths = [r["length"] for r in results]
        hits = sum(1 for l in lengths if l - best_known < 0.01)
        print(f"{name:<12}{mean(lengths):>10.2f}{min(lengths):>10.2f}{max(lengths):>10.2f}"
              f"{mean([r['gen_of_best'] for r in results]):>10.1f}"
              f"{mean([r['time'] for r in results]):>9.2f}s"
              f"{hits:>6}/{len(results)}")


def mean_curve(results:list, key:str):
    return [mean(values) for values in zip(*(r[key] for r in results))]


def plot_convergence(count:int, all_results:dict):
    fig, (ax_best, ax_avg) = plt.subplots(1, 2, figsize=(13, 5))
    for name, results in all_results.items():
        ax_best.plot(mean_curve(results, "best_history"), label=name)
        ax_avg.plot(mean_curve(results, "avg_history"), label=name)
    ax_best.set_title(f"Best length per generation ({count} cities, mean of {RUNS} runs)")
    ax_avg.set_title(f"Average population length ({count} cities, mean of {RUNS} runs)")
    for ax in (ax_best, ax_avg):
        ax.set_xlabel("generation")
        ax.set_ylabel("route length")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"convergence_{count}.png"), dpi=150)
    plt.close(fig)


def plot_boxplot(count:int, all_results:dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([[r["length"] for r in results] for results in all_results.values()])
    ax.set_xticks(range(1, len(all_results) + 1), list(all_results.keys()))
    ax.set_title(f"Final length over {RUNS} runs ({count} cities)")
    ax.set_ylabel("route length")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"boxplot_{count}.png"), dpi=150)
    plt.close(fig)


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


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for count in CITY_COUNTS:
        cities, dist = make_map(count)
        all_results = {}
        for name, selection in METHODS.items():
            print(f"{count} cities, {name}...")
            all_results[name] = run_experiment(dist, selection)
        print_table(count, all_results)
        plot_convergence(count, all_results)
        plot_boxplot(count, all_results)
        plot_routes(count, cities, all_results)
    print(f"\nPlots saved to '{PLOTS_DIR}/'")


if __name__ == '__main__':
    main()
