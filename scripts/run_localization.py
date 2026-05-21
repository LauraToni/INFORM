"""
Run INFORM localization on one circular or median-nerve population.

This script is a cleaned, reproducible version of the original
``CIRCLE_4_localization_loop.ipynb`` notebook. It keeps the same scientific
pipeline but removes hard-coded local paths and imports the cleaned modules.

Typical usage
-------------
python scripts/run_localization.py \
    --project-folder /path/to/Models \
    --median \
    --block-id 10 \
    --segment-id 1 \
    --mask-id 1 \
    --trial cross \
    --population-start 1 \
    --population-stop 50
"""

from __future__ import annotations

import argparse
import pickle
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
    plot_recruitment_curves,
    plot_recruitment_superposed,
)


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

CIRCULAR_COLORS = [
    "#1f78b4",
    "#ee7674",
    "#F6BD60",
    "#8dd3c7",
    "#e31a1c",
    "#cab2d6",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run INFORM localization.")

    parser.add_argument(
        "--project-folder",
        type=Path,
        required=True,
        help="Root folder containing experiment and surrogate experiment data.",
    )

    parser.add_argument(
        "--median",
        action="store_true",
        help="Use median-nerve folder naming convention.",
    )

    parser.add_argument("--block-id", type=int, default=10)
    parser.add_argument("--segment-id", type=int, default=1)
    parser.add_argument("--mask-id", type=int, default=1)
    parser.add_argument("--shift", type=str, default="")
    parser.add_argument("--trial", type=str, default="cross")

    parser.add_argument("--population-start", type=int, default=1)
    parser.add_argument("--population-stop", type=int, default=2)

    parser.add_argument("--n-stims-per-site", type=int, default=20)
    parser.add_argument("--min-stim", type=float, default=0.0)
    parser.add_argument("--max-stim", type=float, default=1.0)

    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tradeoff", type=float, default=0.1)

    parser.add_argument("--std-min", type=float, default=0.1)
    parser.add_argument("--std-max", type=float, default=0.5)
    parser.add_argument("--num-fibers", type=int, default=200)
    parser.add_argument("--n-location-x", type=int, default=20)
    parser.add_argument("--n-location-y", type=int, default=20)
    parser.add_argument("--n-std", type=int, default=20)
    parser.add_argument("--n-num", type=int, default=20)

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save diagnostic plots to the results folder.",
    )

    return parser.parse_args()


def get_nerve_folders(args: argparse.Namespace) -> tuple[str, str, list[str]]:
    """Return true and surrogate nerve folders plus color list."""
    if args.median:
        true_nerve_folder = (
            f"block_{args.block_id}_segment_{args.segment_id}_"
            f"nerve_morphology_{args.mask_id}"
        )
        surrogate_nerve_folder = f"{true_nerve_folder}{args.shift}"
        colors = MEDIAN_COLORS
    else:
        true_nerve_folder = "nerve_2"
        surrogate_nerve_folder = "nerve_2"
        colors = CIRCULAR_COLORS

    return true_nerve_folder, surrogate_nerve_folder, colors


def load_pickle(path: Path):
    """Load a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path: Path, obj) -> None:
    """Save an object to pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_experiments(args: argparse.Namespace):
    """Load predicted full experiment and return base folders."""
    true_nerve_folder, surrogate_nerve_folder, colors = get_nerve_folders(args)

    surrogate_base_path = (
        args.project_folder
        / "Median nerve"
        / "surrogate_experiments"
        / surrogate_nerve_folder
        / args.trial
    )

    true_base_path = (
        args.project_folder
        / "Median nerve"
        / "experiments"
        / true_nerve_folder
        / args.trial
    )

    predicted_experiment_path = surrogate_base_path / f"experiment_trial_{args.trial}.pkl"
    full_pred_experiment = load_pickle(predicted_experiment_path)

    return full_pred_experiment, true_base_path, surrogate_base_path, colors


def get_population_filename(args: argparse.Namespace, population_id: int) -> str:
    """Return population experiment filename."""
    if args.median:
        return f"experiment_pop{population_id}trial{args.trial}.pkl"

    return f"experiment_pop_{population_id}trial_{args.trial}.pkl"


def compute_reference_recruitment_curves(true_experiment, args: argparse.Namespace):
    """Compute true/reference monopolar recruitment curves."""
    recruitment_curves = true_experiment.generate_recruitment_curves(
        amp_lims=[args.min_stim, args.max_stim],
        n_steps=args.n_stims_per_site,
        method="from_self",
    )
    return recruitment_curves


def build_candidate_grid(full_pred_experiment, args: argparse.Namespace):
    """Build and standardize localization candidate grid."""
    nerve_radius = full_pred_experiment.nerve_topography.nerve_radius

    candidates_grid = create_loc_candidates(
        nerve_radius=nerve_radius,
        limCandidateStd=(args.std_min, args.std_max),
        limCandidateNum=(args.num_fibers, args.num_fibers),
        nTriesLocs=(args.n_location_x, args.n_location_y),
        nTriesStd=args.n_std,
        nTriesNum=args.n_num,
    )

    scaler = StandardScaler()
    standardized_candidates_grid = scaler.fit_transform(candidates_grid)

    return candidates_grid, standardized_candidates_grid, scaler


def run_cluster_localization(
    full_pred_experiment,
    full_pred_lfm,
    true_recruitment_values: np.ndarray,
    candidates_grid: np.ndarray,
    standardized_candidates_grid: np.ndarray,
    args: argparse.Namespace,
):
    """Run sequential Bayesian localization over all functional clusters."""
    n_clusters = true_recruitment_values.shape[1]

    experiment_info = {
        "full_experiment": full_pred_experiment,
        "full_lfm": full_pred_lfm,
        "amp_lims": np.array([args.min_stim, args.max_stim]),
        "n_stims_per_site": args.n_stims_per_site,
    }

    kernel = Matern(
        length_scale=[1.0, 1.0, 1.0, 1.0],
        nu=2.5,
        length_scale_bounds=(1e-5, float("inf")),
    )

    x_clust = [None for _ in range(n_clusters)]
    y_clust = [None for _ in range(n_clusters)]
    rc_clust = [None for _ in range(n_clusters)]
    x_max_clust = [None for _ in range(n_clusters)]
    y_max_clust = [None for _ in range(n_clusters)]
    acq_clust = [None for _ in range(n_clusters)]
    y_prior = [None for _ in range(n_clusters)]

    prior_x = []
    prior_rc = []

    start_time = time.time()

    for cluster_idx in tqdm(range(n_clusters), desc="Localizing clusters"):
        reference_curves = true_recruitment_values[:, cluster_idx : cluster_idx + 1, :]

        if cluster_idx == 0:
            prior_info_x = None
            prior_recruitment_curves = None
        else:
            prior_x = prior_x + x_clust[cluster_idx - 1]
            prior_rc = prior_rc + rc_clust[cluster_idx - 1]
            prior_info_x = prior_x
            prior_recruitment_curves = prior_rc

        (
            x_clust[cluster_idx],
            y_clust[cluster_idx],
            rc_clust[cluster_idx],
            x_max_clust[cluster_idx],
            y_max_clust[cluster_idx],
            acq_clust[cluster_idx],
            y_prior[cluster_idx],
        ) = performLocalizationCluster(
            experiment_info=experiment_info,
            refCurves=reference_curves,
            candidatesGrid=candidates_grid,
            candidatesGridStandardized=standardized_candidates_grid,
            kernel=kernel,
            tr=args.tradeoff,
            maxIter=args.max_iter,
            batchSize=args.batch_size,
            priorInfoX=prior_info_x,
            rcPrior=prior_recruitment_curves,
        )

    elapsed_time = time.time() - start_time
    print(f"Localization elapsed time: {elapsed_time:.2f} s")

    return {
        "X_clust": x_clust,
        "Y_clust": y_clust,
        "rc_clust": rc_clust,
        "X_max_clust": x_max_clust,
        "Y_max_clust": y_max_clust,
        "acq_clust": acq_clust,
        "YPrior": y_prior,
        "priorX": prior_x,
        "priorRC": prior_rc,
        "elapsed_time": elapsed_time,
    }


def extract_predicted_cluster_parameters(localization_outputs, scaler, n_clusters: int):
    """Convert best standardized candidates back to physical cluster parameters."""
    cluster_locs_pred = np.zeros((n_clusters, 2))
    cluster_std_pred = np.zeros(n_clusters)
    cluster_num_pred = np.zeros(n_clusters)

    for cluster_idx in range(n_clusters):
        best_candidate = localization_outputs["X_max_clust"][cluster_idx][-1].reshape(1, -1)
        best_candidate_real = scaler.inverse_transform(best_candidate)[0]

        cluster_locs_pred[cluster_idx, :] = best_candidate_real[0:2]
        cluster_std_pred[cluster_idx] = best_candidate_real[2]
        cluster_num_pred[cluster_idx] = best_candidate_real[3]

    return cluster_locs_pred, cluster_std_pred, cluster_num_pred


def build_predicted_experiment(
    full_pred_experiment,
    cluster_locs_pred,
    cluster_std_pred,
    cluster_num_pred,
):
    """Create predicted population and experiment from localized parameters."""
    pred_population, pred_identities = MotorFiberPopulation.from_existing_population(
        cluster_locs=cluster_locs_pred,
        cluster_std=cluster_std_pred,
        cluster_num=cluster_num_pred.astype(int),
        fiber_population=full_pred_experiment.fiber_population,
    )

    pred_identities = pred_identities.astype(int)

    pred_experiment = Experiment(
        fiber_population=pred_population,
        nerve_topography=full_pred_experiment.nerve_topography,
        implant=full_pred_experiment.implant,
    )

    pred_experiment._activation_predictor = full_pred_experiment.activation_predictor
    pred_experiment.load_lead_field_matrix(
        full_experiment=full_pred_experiment,
        lead_field_matrix=full_pred_experiment.lead_field_matrix,
        identities=pred_identities,
        save_to_hdf5=False,
    )

    return pred_experiment, pred_population, pred_identities


def run_population(
    population_id: int,
    full_pred_experiment,
    true_base_path: Path,
    surrogate_base_path: Path,
    colors: list[str],
    args: argparse.Namespace,
) -> None:
    """Run localization for one population."""
    population_file = true_base_path / get_population_filename(args, population_id)
    true_experiment = load_pickle(population_file)

    n_clusters = true_experiment.fiber_population.n_groups
    mean_locs_x = true_experiment.fiber_population.cluster_locs[:, 0]
    mean_locs_y = true_experiment.fiber_population.cluster_locs[:, 1]
    empirical_std = true_experiment.fiber_population.cluster_std

    reference_rc = compute_reference_recruitment_curves(true_experiment, args)
    reference_rc_values = reference_rc.recruitment_values
    amplitudes = reference_rc.amplitudes

    if args.save_plots:
        fig, _ = plot_recruitment_curves(
            recruitment_values=reference_rc_values,
            n_clusters=n_clusters,
            amplitudes=amplitudes,
            n_sites=reference_rc.n_sites,
            colors=colors,
            title="Reference recruitment curves",
        )
        fig.savefig(surrogate_base_path / "results" / f"reference_rc_pop_{population_id}.png", dpi=300)
        plt.close(fig)

    candidates_grid, standardized_candidates_grid, scaler = build_candidate_grid(
        full_pred_experiment=full_pred_experiment,
        args=args,
    )

    localization_outputs = run_cluster_localization(
        full_pred_experiment=full_pred_experiment,
        full_pred_lfm=full_pred_experiment.lead_field_matrix,
        true_recruitment_values=reference_rc_values,
        candidates_grid=candidates_grid,
        standardized_candidates_grid=standardized_candidates_grid,
        args=args,
    )

    cluster_locs_pred, cluster_std_pred, cluster_num_pred = extract_predicted_cluster_parameters(
        localization_outputs=localization_outputs,
        scaler=scaler,
        n_clusters=n_clusters,
    )

    pred_experiment, pred_population, pred_identities = build_predicted_experiment(
        full_pred_experiment=full_pred_experiment,
        cluster_locs_pred=cluster_locs_pred,
        cluster_std_pred=cluster_std_pred,
        cluster_num_pred=cluster_num_pred,
    )

    pred_recruitment_curves = pred_experiment.generate_recruitment_curves(
        amp_lims=np.array([args.min_stim, args.max_stim]),
        n_steps=args.n_stims_per_site,
        method="from_self",
    )

    if args.save_plots:
        fig, _ = plot_recruitment_superposed(
            true_recruitment_curves=reference_rc_values,
            pred_recruitment_curves=pred_recruitment_curves.recruitment_values,
            amplitudes=np.linspace(args.min_stim, args.max_stim, args.n_stims_per_site),
            n_sites=reference_rc_values.shape[0],
            n_clusters=reference_rc_values.shape[1],
            colors=colors,
            title="Recruitment curves reconstructed",
        )
        fig.savefig(surrogate_base_path / "results" / f"reconstructed_rc_pop_{population_id}.png", dpi=300)
        plt.close(fig)

    results = {
        "mean_locs_x": mean_locs_x,
        "mean_locs_y": mean_locs_y,
        "empirical_std": empirical_std,
        "true_recruitment_curves": reference_rc_values,
        "max_stim": args.max_stim,
        "n_stims_per_site": args.n_stims_per_site,
        "standardized_candidates_grid": standardized_candidates_grid,
        "scaler": scaler,
        "candidates_grid": candidates_grid,
        "localization_outputs": localization_outputs,
        "pred_experiment": pred_experiment,
        "pred_recruitment_curves": pred_recruitment_curves,
        "pred_identities": pred_identities,
    }

    results_filename = f"results_{args.trial}_pop_{population_id}_with_topography.pkl"
    save_pickle(surrogate_base_path / "results" / results_filename, results)


def main() -> None:
    """Run INFORM localization."""
    args = parse_args()

    np.set_printoptions(suppress=True)

    full_pred_experiment, true_base_path, surrogate_base_path, colors = load_experiments(args)

    for population_id in range(args.population_start, args.population_stop):
        print(f"Running population {population_id}")
        run_population(
            population_id=population_id,
            full_pred_experiment=full_pred_experiment,
            true_base_path=true_base_path,
            surrogate_base_path=surrogate_base_path,
            colors=colors,
            args=args,
        )


if __name__ == "__main__":
    main()
