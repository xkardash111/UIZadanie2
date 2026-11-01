import csv
import os
import random
import statistics
import time

import matplotlib.pyplot as plt

from simulatedanneling import simulated_annealing
from compare import make_map, mean, mean_curve, style, value_colors, PLOTS_DIR, RESULTS_DIR
from configs import RUNS, SA_CITY_COUNTS, SA_SCHEDULE_CITIES, SA_BASELINE, SA_EXPERIMENTS

SA_PLOTS_DIR = os.path.join(PLOTS_DIR, "sa")     # all SA plots
SWEEP_COLORS = {"alpha": "#2a78d6", "t0": "#eb6834", "steps_per_t": "#1baf7a"}  # one color per swept parameter
LABEL_OFFSETS = {"alpha": (5, 6), "t0": (-12, 8), "steps_per_t": (5, -13)}  # value labels of each sweep don't overlap
CURRENT_COLOR = "#86b6ef"  # light blue: route SA is standing on
BEST_COLOR = "#104281"     # dark blue: best route found so far

_cache = {}  # (cities, config) -> results of RUNS runs


# runs SA RUNS times (seeds 0..RUNS-1) with one configuration and collects stats of every run
def run_experiment(count:int, config:dict):
    key = (count, tuple(sorted(config.items())))
    if key in _cache:
        return _cache[key]

    _, dist = make_map(count)
    results = []
    for seed in range(RUNS):
        random.seed(seed)
        start = time.perf_counter()
        best, length, history = simulated_annealing(dist, config["t0"], config["alpha"],
                                                    config["steps_per_t"], config["t_min"])
        elapsed = time.perf_counter() - start
        results.append({
            "cities": count,
            "config": dict(config),
            "seed": seed,
            "route": best,
            "length": length,
            "levels": len(history["T"]),
            "iterations": len(history["T"]) * config["steps_per_t"],
            "time": elapsed,
            "history": history,
        })
    _cache[key] = results
    return results


# changes one parameter through its values on every map, everything else stays SA_BASELINE
def run_sweep(param:str, values:list):
    sweep = {}
    for count in SA_CITY_COUNTS:
        for value in values:
            config = dict(SA_BASELINE)
            config[param] = value
            print(f"{param}={value}, {count} cities...")
            sweep[(count, value)] = run_experiment(count, config)
    return sweep


# shortest route SA found on this map over all runs of all experiments
def best_known(count:int):
    return min(r["length"] for results in _cache.values() for r in results if r["cities"] == count)


# one table row from several runs: length stats, gap to best known, schedule length, time, hits
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
        "iterations": results[0]["iterations"],  # the schedule is deterministic -> same for every seed
        "avg_time": mean([r["time"] for r in results]),
        "hit_best": f"{hits}/{len(results)}",
    }


# ---------- tables ----------

# writes results/sa_<param>.csv and prints the same table to the console
def write_sweep_table(param:str, sweep:dict):
    path = os.path.join(RESULTS_DIR, f"sa_{param}.csv")
    print(f"\n=== SA {param} ===")
    print(f"{param:<13}{'cities':>7}{'avg':>9}{'std':>8}{'best':>9}{'worst':>9}{'gap %':>7}"
          f"{'iterations':>12}{'time':>8}{'hit':>6}")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([param, "cities", "avg", "std", "best", "worst", "gap_%", "iterations", "avg_time", "hit_best"])
        for (count, value), results in sweep.items():
            s = summarize(results)
            writer.writerow([value, count, round(s["avg"], 2), round(s["std"], 2), round(s["best"], 2),
                             round(s["worst"], 2), round(s["gap_%"], 2), s["iterations"],
                             round(s["avg_time"], 3), s["hit_best"]])
            print(f"{str(value):<13}{count:>7}{s['avg']:>9.2f}{s['std']:>8.2f}{s['best']:>9.2f}{s['worst']:>9.2f}"
                  f"{s['gap_%']:>7.2f}{s['iterations']:>12}{s['avg_time']:>7.2f}s{s['hit_best']:>6}")


# writes results/sa_runs.csv, one row per SA run
def write_runs_table():
    path = os.path.join(RESULTS_DIR, "sa_runs.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cities", "t0", "alpha", "steps_per_t", "t_min", "seed", "length",
                         "levels", "iterations", "time", "route"])
        for results in _cache.values():
            for r in results:
                c = r["config"]
                writer.writerow([r["cities"], c["t0"], c["alpha"], c["steps_per_t"], c["t_min"], r["seed"],
                                 round(r["length"], 2), r["levels"], r["iterations"], round(r["time"], 3),
                                 " ".join(map(str, r["route"]))])


# ---------- schedule: quality vs cost, and what happens during cooling ----------

# every configuration of every sweep as a point: schedule length vs final length
def plot_tradeoff(sweeps:dict):
    fig, axes = plt.subplots(1, len(SA_CITY_COUNTS), figsize=(7 * len(SA_CITY_COUNTS), 5.5))
    for ax, count in zip(axes, SA_CITY_COUNTS):
        for param, sweep in sweeps.items():
            values = SA_EXPERIMENTS[param]
            summaries = [summarize(sweep[(count, v)]) for v in values]
            xs = [s["iterations"] for s in summaries]
            ys = [s["avg"] for s in summaries]
            ax.plot(xs, ys, marker="o", markersize=7, linewidth=2, color=SWEEP_COLORS[param], label=param)
            for x, y, value in zip(xs, ys, values):
                if value == SA_BASELINE[param]:
                    continue  # the baseline is the same point in every sweep, labelled once below
                ax.annotate(str(value), (x, y), textcoords="offset points", xytext=LABEL_OFFSETS[param],
                            fontsize=8)
        baseline = summarize(run_experiment(count, SA_BASELINE))
        ax.annotate("baseline", (baseline["iterations"], baseline["avg"]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold")
        ax.set_xscale("log")
        ax.set_title(f"Schedule length vs result ({count} cities, mean of {RUNS} runs)")
        style(ax, "iterations of the schedule (log scale)", "mean final route length")
        ax.legend(title="changed parameter")
    fig.tight_layout()
    fig.savefig(os.path.join(SA_PLOTS_DIR, "tradeoff.png"), dpi=150)
    plt.close(fig)


# temperature and acceptance rate for every alpha, and one baseline run (current vs best route)
def plot_schedule(count:int, alpha_sweep:dict):
    values = SA_EXPERIMENTS["alpha"]
    fig, (ax_t, ax_acc, ax_run) = plt.subplots(1, 3, figsize=(19, 5))
    for value, color in zip(values, value_colors(values)):
        results = alpha_sweep[(count, value)]
        steps = results[0]["config"]["steps_per_t"]
        temperatures = results[0]["history"]["T"]
        acceptance = mean_curve([r["history"] for r in results], "acceptance")
        xs = [(i + 1) * steps for i in range(len(temperatures))]
        ax_t.plot(xs, temperatures, color=color, linewidth=2, label=f"alpha={value}")
        ax_acc.plot(xs, [a * 100 for a in acceptance], color=color, linewidth=2, label=f"alpha={value}")

    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_title("Temperature schedule")
    style(ax_t, "iteration (log scale)", "temperature (log scale)")
    ax_t.legend()

    ax_acc.set_xscale("log")
    ax_acc.set_title(f"Accepted moves per temperature level (mean of {RUNS} runs)")
    style(ax_acc, "iteration (log scale)", "accepted moves [%]")
    ax_acc.legend()

    run = run_experiment(count, SA_BASELINE)[0]
    steps = SA_BASELINE["steps_per_t"]
    xs = [(i + 1) * steps for i in range(run["levels"])]
    ax_run.plot(xs, run["history"]["current"], color=CURRENT_COLOR, linewidth=2, label="current route")
    ax_run.plot(xs, run["history"]["best"], color=BEST_COLOR, linewidth=2, label="best so far")
    ax_run.set_title(f"One baseline run (alpha={SA_BASELINE['alpha']}, seed 0)")
    style(ax_run, "iteration", "route length")
    ax_run.legend()

    fig.suptitle(f"Simulated annealing schedule, {count} cities")
    fig.tight_layout()
    fig.savefig(os.path.join(SA_PLOTS_DIR, f"schedule_{count}.png"), dpi=150)
    plt.close(fig)


# best route of the baseline drawn on the map, one subplot per map
def plot_routes():
    fig, axes = plt.subplots(1, len(SA_CITY_COUNTS), figsize=(6 * len(SA_CITY_COUNTS), 6))
    for ax, count in zip(axes, SA_CITY_COUNTS):
        cities, _ = make_map(count)
        best = min(run_experiment(count, SA_BASELINE), key=lambda r: r["length"])
        route = best["route"] + [best["route"][0]]
        ax.plot([cities[c][0] for c in route], [cities[c][1] for c in route], "-", color="gray")
        ax.scatter([x for x, y in cities], [y for x, y in cities], color="red", zorder=3)
        for i, (x, y) in enumerate(cities):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(f"SA baseline, {count} cities: {best['length']:.2f}")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(SA_PLOTS_DIR, "routes.png"), dpi=150)
    plt.close(fig)


# parameter sweeps -> tables -> tradeoff, schedule and route plots
def main():
    os.makedirs(SA_PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sweeps = {param: run_sweep(param, values) for param, values in SA_EXPERIMENTS.items()}

    # tables after all runs, so "best known" uses every run on that map
    for param in SA_EXPERIMENTS:
        write_sweep_table(param, sweeps[param])
    write_runs_table()

    plot_tradeoff(sweeps)
    plot_schedule(SA_SCHEDULE_CITIES, sweeps["alpha"])
    plot_routes()
    print(f"\nPlots saved to '{SA_PLOTS_DIR}/', tables to '{RESULTS_DIR}/'")


if __name__ == '__main__':
    main()
