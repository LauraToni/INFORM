"""
Run INFORM localization for the realistic histological median-nerve case.

This script follows the validated MEDIAN localization notebook logic and keeps
the original batch-based INFORM localization routine.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from nerve_model.experiment import Experiment
from nerve_model.fiber_population import MotorFiberPopulation
from localization.candidate_generation import create_loc_candidates
from localization.bayesian_localization import performLocalizationCluster
from localization.visualization import (
    plot_recruitment_extended,
    plot_recruitment_superposed,
)

# Legacy pickle compatibility.
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


MEDIAN_COLORS = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run INFORM histological median localization.")

    parser.add_argument("--project-folder", type=Path, required=True)
    parser.add_argument("--block-id", type=int, default=10)
    parser.add_argument("--segment-id", type=int, default=1)
    parser.add_argument("--mask-id", type=int, default=1)
    parser.add_argument("--shift", type=str, default="")
    parser.add_argument("--trial", type=str, default="cross")

    parser.add_argument("--n-stims-per-site", type=int, default=20)
    parser.add_argument("--min-stim", type=float, default=0.0)
    parser.add_argument("--max-stim", type=float, default=1000.0)

    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--tradeoff", type=float, default=0.1)

    parser.add_argument("--std-min", type=float, default=0.05)
    parser.add_argument("--std-max", type=float, default=0.2)
    parser.add_argument("--num-fibers", type=int, default=200)
    parser.add_argument("--n-location-x", type=int, default=20)
    parser.add_argument("--n-location-y", type=int, default=20)
    parser.add_argument("--n-std", type=int, default=20)
    parser.add_argument("--n-num", type=int, default=20)
    parser.add_argument("--nerve-radius", type=float, default=3.0)

    parser.add_argument("--n-clusters-to-localize", type=int, default=2)
    parser.add_argument("--save-full-reference-objects", action="store_true")
    parser.add_argument("--save-plots", action="store_true")

    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def main() -> None:
    args = parse_args()
    np.set_printoptions(suppress=True)

    nerve_folder = f"block_{args.block_id}_segment_{args.segment_id}_nerve_morphology_{args.mask_id}"
    inferred_nerve_folder = f"{nerve_folder}{args.shift}"

    inferred_base_path = (
        args.project_folder
        / "Median nerve"
        / "surrogate_experiments"
        / inferred_nerve_folder
        / args.trial
    )

    true_base_path = (
        args.project_folder
        / "Median nerve"
        / "experiments"
        / nerve_folder
        / args.trial
    )

    inferred_experiment_path = inferred_base_path / f"experiment_trial_{args.trial}.pkl"
    true_experiment_path = true_base_path / f"experiment_trial_{args.trial}.pkl"
    results_folder = inferred_base_path / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    full_inferred_experiment = load_pickle(inferred_experiment_path)
    true_experiment = load_pickle(true_experiment_path)

    fiber_population_ref = MotorFiberPopulation(
        n_internodes=true_experiment.fiber_population.n_internodes,
        diameters=true_experiment.fiber_population.diameters,
        locs=true_experiment.fiber_population.locs,
        central_node_z="random",
        cluster_ids=true_experiment.fiber_population.cluster_ids,
        nerve_topography=true_experiment.nerve_topography,
        cc=True,
    )

    n_clusters_total = fiber_population_ref.n_groups
    n_clusters = min(args.n_clusters_to_localize, n_clusters_total)

    mean_locs_x = fiber_population_ref.cluster_locs[:, 0]
    mean_locs_y = fiber_population_ref.cluster_locs[:, 1]
    new_std = fiber_population_ref.cluster_std

    amp_lims = [args.min_stim, args.max_stim]
    true_recruitment_curves_obj = true_experiment.generate_recruitment_curves(
        amp_lims=amp_lims,
        n_steps=args.n_stims_per_site,
        method="from_self",
    )

    true_recruitment_curves = true_recruitment_curves_obj.recruitment_values
    amplitudes = true_recruitment_curves_obj.amplitudes

    candidates_grid = create_loc_candidates(
        nerve_radius=args.nerve_radius,
        limCandidateStd=(args.std_min, args.std_max),
        limCandidateNum=(args.num_fibers, args.num_fibers),
        nTriesLocs=(args.n_location_x, args.n_location_y),
        nTriesStd=args.n_std,
        nTriesNum=args.n_num,
    )

    scaler = StandardScaler()
    standardized_candidates_grid = scaler.fit_transform(candidates_grid)

    experiment_info = {
        "full_experiment": full_inferred_experiment,
        "full_lfm": full_inferred_experiment.lead_field_matrix,
        "amp_lims": np.array(amp_lims),
        "n_stims_per_site": args.n_stims_per_site,
    }

    kernel = Matern(
        length_scale=[1.0, 1.0, 1.0, 1.0],
        nu=2.5,
        length_scale_bounds=(1e-5, 1e5),
    )

    X_clust = [None for _ in range(n_clusters)]
    Y_clust = [None for _ in range(n_clusters)]
    rc_clust = [None for _ in range(n_clusters)]
    X_max_clust = [None for _ in range(n_clusters)]
    Y_max_clust = [None for _ in range(n_clusters)]
    acq_clust = [None for _ in range(n_clusters)]
    YPrior = [None for _ in range(n_clusters)]
    priorX = []
    priorRC = []
    priorExp = []

    start = time.time()

    for clust in tqdm(range(n_clusters), desc="Localizing clusters"):
        refCurves = true_recruitment_curves[:, clust : clust + 1, :]

        if clust == 0:
            priorInfoX = None
            rcPrior = None
        else:
            priorX = priorX + X_clust[clust - 1]
            priorRC = priorRC + rc_clust[clust - 1]
            priorInfoX = priorX
            rcPrior = priorRC

        (
            X_clust[clust],
            Y_clust[clust],
            rc_clust[clust],
            X_max_clust[clust],
            Y_max_clust[clust],
            acq_clust[clust],
            YPrior[clust],
        ) = performLocalizationCluster(
            experiment_info=experiment_info,
            refCurves=refCurves,
            candidatesGrid=candidates_grid,
            candidatesGridStandardized=standardized_candidates_grid,
            kernel=kernel,
            tr=args.tradeoff,
            maxIter=args.max_iter,
            batchSize=args.batch_size,
            priorInfoX=priorInfoX,
            rcPrior=rcPrior,
        )

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f} s")

    x_inferred = [None for _ in range(n_clusters)]
    y_inferred = [None for _ in range(n_clusters)]
    std_inferred = [None for _ in range(n_clusters)]
    num_inferred = [None for _ in range(n_clusters)]

    for i in range(n_clusters):
        best_candidate_std = X_max_clust[i][-1].reshape(1, -1)
        best_candidate = scaler.inverse_transform(best_candidate_std)[0]
        x_inferred[i] = best_candidate[0]
        y_inferred[i] = best_candidate[1]
        std_inferred[i] = best_candidate[2]
        num_inferred[i] = best_candidate[3]

    cluster_locs_inferred = np.column_stack((x_inferred, y_inferred))
    cluster_std_inferred = np.asarray(std_inferred)
    cluster_num_inferred = np.asarray(num_inferred)

    inferred_population, inferred_identities = MotorFiberPopulation.from_existing_population(
        cluster_locs=cluster_locs_inferred,
        cluster_std=cluster_std_inferred,
        cluster_num=cluster_num_inferred.astype(int),
        fiber_population=full_inferred_experiment.fiber_population,
    )

    inferred_identities = inferred_identities.astype(int)

    inferred_experiment = Experiment(
        fiber_population=inferred_population,
        nerve_topography=full_inferred_experiment.nerve_topography,
        implant=full_inferred_experiment.implant,
    )

    inferred_experiment._activation_predictor = full_inferred_experiment.activation_predictor
    inferred_experiment.load_lead_field_matrix(
        full_experiment=full_inferred_experiment,
        lead_field_matrix=full_inferred_experiment.lead_field_matrix,
        identities=inferred_identities,
        save_to_hdf5=False,
    )

    inferred_recruitment_curves_obj = inferred_experiment.generate_recruitment_curves(
        amp_lims=np.array(amp_lims),
        n_steps=args.n_stims_per_site,
        method="from_self",
    )

    if args.save_plots:
        fig, _ = plot_recruitment_superposed(
            true_recruitment_curves=true_recruitment_curves,
            inferred_recruitment_curves=inferred_recruitment_curves_obj.recruitment_values,
            amplitudes=np.linspace(args.min_stim, args.max_stim, args.n_stims_per_site),
            n_sites=true_recruitment_curves.shape[0],
            n_clusters=n_clusters,
            colorList=MEDIAN_COLORS,
            title="Recruitment curves reconstructed",
        )
        fig.savefig(results_folder / f"reconstructed_rc_{args.trial}.png", dpi=300)
        plt.close(fig)

    results_filename = f"results_{args.trial}_with_topography.pkl"

    if args.save_full_reference_objects:
        results_to_save = [
            mean_locs_x,
            mean_locs_y,
            new_std,
            true_recruitment_curves,
            args.max_stim,
            args.n_stims_per_site,
            standardized_candidates_grid,
            scaler,
            candidates_grid,
            X_clust,
            Y_clust,
            X_max_clust,
            Y_max_clust,
            acq_clust,
            rc_clust,
            YPrior,
            priorX,
            priorRC,
            priorExp,
            inferred_experiment,
            inferred_recruitment_curves_obj,
            inferred_identities,
        ]
    else:
        results_to_save = [
            mean_locs_x,
            mean_locs_y,
            new_std,
            true_recruitment_curves,
            args.max_stim,
            args.n_stims_per_site,
            standardized_candidates_grid,
            scaler,
            candidates_grid,
            X_clust,
            Y_clust,
            X_max_clust,
            Y_max_clust,
            acq_clust,
            YPrior,
            priorX,
            priorExp,
            inferred_experiment,
            inferred_recruitment_curves_obj,
            inferred_identities,
        ]

    save_pickle(results_folder / results_filename, results_to_save)
    print("Saved:", results_folder / results_filename)


if __name__ == "__main__":
    main()