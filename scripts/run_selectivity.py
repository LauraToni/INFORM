"""
Run INFORM selectivity optimization on one nerve (circular or median).

Cleaned, reproducible counterpart of the CIRCLE/MEDIAN selectivity notebooks.
For each population it loads the localization results (which contain the
inferred ``pred_experiment``), optimizes a selective stimulation protocol per
cluster via PSO on BOTH the inferred and the true experiment, and saves the
comparison. Diagnostic plotting lives in the example notebooks, not here, to
stay consistent with ``run_localization.py``.

The PSO objective is ``params_to_selectivity_rasp`` (the margin metric, i.e.
``selectivity_opt``). Cross-evaluation of the inferred-optimized protocol on the
true nerve is summarized with ``selectivity_eval`` (the squared-ratio metric).

Usage
-----
python scripts/run_selectivity.py \
    --project-folder "/path/to/Models" \
    --nerve-folder nerve_1 \
    --trial cross \
    --population-start 1 --population-stop 51 \
    --active-sites 3
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from nerve_model.experiment import Experiment  # noqa: F401  (used via pickles)
from selectivity_optimization.selectivity import (
    params_to_selectivity_rasp,
    selectivity_eval,
    off_diagonal_frobenius_norm,
)

# -----------------------------------------------------------------------------
# Legacy pickle compatibility (same shim as the localization scripts)
# -----------------------------------------------------------------------------
import nerve_model.experiment as experiment
import nerve_model.fiber_population as fiber_population
import nerve_model.nerve_section as nerve_section
import nerve_model.histological_nerve_section as histological_nerve_section
import nerve_model.recruitment_curves as recruitment_curves
import nerve_model.implant as implant

sys.modules["experiment"] = experiment
sys.modules["fiber_population"] = fiber_population
sys.modules["nerve_section"] = nerve_section
sys.modules["recruitment_curves"] = recruitment_curves
sys.modules["implant"] = implant
# Legacy median pickles saved the polygonal topography as 'nerve_section_2'.
sys.modules["nerve_section_2"] = histological_nerve_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run INFORM selectivity optimization.")

    parser.add_argument("--project-folder", type=Path, required=True,
                        help="Root folder containing experiments/ and surrogate_experiments/.")
    parser.add_argument("--median", action="store_true",
                        help="Use median-nerve folder naming convention.")
    parser.add_argument("--nerve-folder", type=str, default="nerve_1",
                        help="Circular-nerve folder name (ignored when --median).")
    parser.add_argument("--block-id", type=int, default=10)
    parser.add_argument("--segment-id", type=int, default=1)
    parser.add_argument("--mask-id", type=int, default=1)
    parser.add_argument("--shift", type=str, default="")
    parser.add_argument("--trial", type=str, default="cross")

    parser.add_argument("--population-start", type=int, default=1)
    parser.add_argument("--population-stop", type=int, default=2)
    parser.add_argument("--active-sites", type=int, default=3,
                        help="Number of simultaneously active electrodes.")

    # PSO settings (defaults match the notebooks).
    parser.add_argument("--max-stim", type=float, default=1.0)
    parser.add_argument("--n-particles", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--ftol-iter", type=int, default=50)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--c2", type=float, default=1.5)
    parser.add_argument("--w", type=float, default=0.9)

    return parser.parse_args()


def get_nerve_folder(args: argparse.Namespace) -> str:
    if args.median:
        base = f"block_{args.block_id}_segment_{args.segment_id}_nerve_morphology_{args.mask_id}"
        return base
    return args.nerve_folder


def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def optimize_one_experiment(target_experiment, args, n_clusters):
    """Optimize a protocol for every cluster of one experiment via PSO.

    Returns lists (per cluster) of the PSO result tuples and the derived
    (protocol, recruitment, selectivity) obtained by re-evaluating the best
    particle. Mirrors the notebook logic; uses params_to_selectivity_rasp.
    """
    from pyswarms.single import GlobalBestPSO

    n_sites = target_experiment.implant.n_sites
    max_c = args.max_stim
    options = {"c1": args.c1, "c2": args.c2, "w": args.w}
    bounds = (
        np.repeat(np.array([[-max_c], [0]]), n_sites),
        np.repeat(np.array([[max_c], [1]]), n_sites),
    )

    stats = [None] * n_clusters
    for cluster_idx in range(n_clusters):
        optimizer = GlobalBestPSO(
            n_particles=args.n_particles,
            dimensions=n_sites * 2,
            options=options,
            bounds=bounds,
            ftol_iter=args.ftol_iter,
            ftol=args.ftol,
        )
        # pyswarms passes the whole swarm (n_particles, dims) to the objective;
        # params_to_selectivity_rasp handles batch_size = n_particles.
        stats[cluster_idx] = optimizer.optimize(
            params_to_selectivity_rasp,
            iters=args.iters,
            n_sites=n_sites,
            n_active_sites=args.active_sites,
            true_population=target_experiment.fiber_population,
            experiment=target_experiment,
            musc_selective=cluster_idx,
            batch_size=args.n_particles,
        )

    # Re-evaluate the best particle of each cluster (batch_size = 1).
    protocols = [None] * n_clusters
    recruitments = [None] * n_clusters
    selectivities = [None] * n_clusters
    for cluster_idx in range(n_clusters):
        best_params = stats[cluster_idx][1]
        sp, rc, sel = params_to_selectivity_rasp(
            best_params, n_sites, args.active_sites,
            target_experiment.fiber_population, target_experiment,
            musc_selective=cluster_idx, batch_size=1, return_recruitment=True,
        )
        protocols[cluster_idx] = sp
        recruitments[cluster_idx] = rc
        selectivities[cluster_idx] = sel

    return stats, protocols, recruitments, selectivities


def run_population(population_id, paths, args):
    """Optimize and cross-evaluate selectivity for one population."""
    true_experiment = load_pickle(paths["population_file"])

    # Localization result carries the inferred (pred) experiment. The generic
    # run script saves a dict (key "pred_experiment"); the median script saves a
    # list with the inferred experiment third from the end. Support both.
    loc = load_pickle(paths["localization_result"])
    if isinstance(loc, dict):
        pred_experiment = loc["pred_experiment"]
    else:
        pred_experiment = loc[-3]

    n_clusters = true_experiment.fiber_population.n_groups

    # Optimize on the inferred organization.
    stats_pred, sp_pred, rc_pred, sel_pred = optimize_one_experiment(
        pred_experiment, args, n_clusters
    )
    rc_pred_array = np.array(rc_pred).squeeze()

    # Cross-evaluation: apply the inferred-optimized protocols to the TRUE nerve.
    rc_true_array = np.zeros((n_clusters, n_clusters))
    for cluster_idx in range(n_clusters):
        rc_true_array[cluster_idx, :] = true_experiment.compute_recruitment_patterns(
            stimulation_protocols=sp_pred[cluster_idx], method="from_self"
        )
    selectivity_crossed = selectivity_eval(rc_true_array, normalize=True, squared=True)

    sel_pred_array = np.array(sel_pred).squeeze()
    _, fro_ratio_pred = off_diagonal_frobenius_norm(rc_pred_array)
    _, fro_ratio_true = off_diagonal_frobenius_norm(rc_true_array)

    results = {
        "n_active_sites": args.active_sites,
        "stats_pred": stats_pred,
        "protocols_pred": sp_pred,
        "recruitment_pred": rc_pred,
        "selectivity_pred": sel_pred,
        "recruitment_true_crossed": rc_true_array,
        "selectivity_crossed": selectivity_crossed,
        "fro_ratio_pred": fro_ratio_pred,
        "fro_ratio_true": fro_ratio_true,
    }

    out = paths["results_folder"] / (
        f"resultsPRED_{args.trial}_pop_{population_id}_activesites_{args.active_sites}.pkl"
    )
    save_pickle(out, results)
    return out


def build_paths(population_id, args):
    nerve_folder = get_nerve_folder(args)
    surrogate_nerve_folder = f"{nerve_folder}{args.shift}"

    exp_base = args.project_folder / "Median nerve" / "experiments" / nerve_folder / args.trial
    surr_base = (
        args.project_folder / "Median nerve" / "surrogate_experiments"
        / surrogate_nerve_folder / args.trial
    )
    results_folder = surr_base / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    if args.median:
        population_file = exp_base / f"experiment_pop{population_id}trial{args.trial}.pkl"
    else:
        population_file = exp_base / f"experiment_pop_{population_id}trial_{args.trial}.pkl"

    localization_result = results_folder / (
        f"results_{args.trial}_pop_{population_id}_with_topography.pkl"
    )

    return {
        "population_file": population_file,
        "localization_result": localization_result,
        "results_folder": results_folder,
    }


def main() -> None:
    args = parse_args()
    np.set_printoptions(suppress=True)

    start = time.time()
    for population_id in tqdm(range(args.population_start, args.population_stop),
                              desc="Selectivity over populations"):
        paths = build_paths(population_id, args)
        out = run_population(population_id, paths, args)
        print("Saved:", out)
    print(f"Elapsed time: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
