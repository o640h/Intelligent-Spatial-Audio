import argparse
import random
from pathlib import Path

import pandas as pd
from deap import base, creator, tools, algorithms


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def target_from_features(row):
    """
    Feature-informed 'ideal' tendencies for each stem.
    These are not final outputs, just soft targets for optimisation.
    """
    file_name = row["file"].lower()
    low_freq_ratio = row["low_freq_ratio"]
    centroid = row["centroid_mean"]
    zcr = row["zcr_mean"]

    if "bass" in file_name or low_freq_ratio > 0.60:
        return 0.00, 0.10, 0.35

    elif "vocals" in file_name:
        return 0.00, 0.20, 0.22

    elif "drums" in file_name:
        width = 0.60 if centroid > 6000 else 0.50
        return 0.00, width, 0.28

    else:
        width = 0.75 if centroid > 3000 else 0.60
        depth = 0.40 if zcr < 0.08 else 0.32
        return 0.00, width, depth


def make_fitness_function(row, start_pan, start_width, start_depth):
    target_pan, target_width, target_depth = target_from_features(row)

    def fitness(individual):
        pan, width, depth = individual

        # Stay close to feature-informed targets
        target_penalty = (
            abs(pan - target_pan) * 1.4
            + abs(width - target_width) * 1.0
            + abs(depth - target_depth) * 0.8
        )

        # Stay somewhat close to the current rule/ML output
        stability_penalty = (
            abs(pan - start_pan) * 1.2
            + abs(width - start_width) * 1.0
            + abs(depth - start_depth) * 0.8
        )

        # Harder penalty for bad bass placement
        bass_penalty = 0.0
        if row["low_freq_ratio"] > 0.60:
            bass_penalty += abs(pan) * 3.0
            bass_penalty += abs(width - 0.10) * 2.0

        # Reward brightness / percussion being wider
        width_reward = 0.0
        if row["centroid_mean"] > 5000:
            width_reward += width * 0.5
        if row["zcr_mean"] > 0.08:
            width_reward += width * 0.4

        # Final objective: lower penalty is better, but DEAP maximises
        score = -(target_penalty + stability_penalty + bass_penalty) + width_reward
        return (score,)

    return fitness


def optimise_stem(row, start_pan, start_width, start_depth, ngen=30, pop_size=30):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_pan():
        return clamp(random.uniform(start_pan - 0.08, start_pan + 0.08), -1.0, 1.0)

    def init_width():
        return clamp(random.uniform(start_width - 0.08, start_width + 0.08), 0.0, 1.0)

    def init_depth():
        return clamp(random.uniform(start_depth - 0.08, start_depth + 0.08), 0.0, 1.0)

    toolbox.register("attr_pan", init_pan)
    toolbox.register("attr_width", init_width)
    toolbox.register("attr_depth", init_depth)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_pan, toolbox.attr_width, toolbox.attr_depth),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    fitness_fn = make_fitness_function(row, start_pan, start_width, start_depth)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.03, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    hall_of_fame = tools.HallOfFame(1)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.6,
        mutpb=0.3,
        ngen=ngen,
        halloffame=hall_of_fame,
        verbose=False,
    )

    best = hall_of_fame[0]
    pan = clamp(best[0], -1.0, 1.0)
    width = clamp(best[1], 0.0, 1.0)
    depth = clamp(best[2], 0.0, 1.0)

    return pan, width, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features CSV")
    parser.add_argument("--spatial", required=True, help="Path to current spatial CSV")
    parser.add_argument("--outfile", default="outputs/spatial/test_spatial_deap.csv", help="Output CSV")
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    spatial_df = pd.read_csv(args.spatial)

    merged = pd.merge(features_df, spatial_df, on="file", how="inner")
    rows = []

    for _, row in merged.iterrows():
        start_pan = float(row["pan"])
        start_width = float(row["width"])
        start_depth = float(row["depth"])

        opt_pan, opt_width, opt_depth = optimise_stem(
            row, start_pan, start_width, start_depth
        )

        rows.append({
            "file": row["file"],
            "start_pan": start_pan,
            "start_width": start_width,
            "start_depth": start_depth,
            "opt_pan": opt_pan,
            "opt_width": opt_width,
            "opt_depth": opt_depth,
        })

    out_df = pd.DataFrame(rows)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, index=False)

    print(out_df)
    print(f"\nSaved DEAP-optimised spatial parameters to: {outfile}")


if __name__ == "__main__":
    main()