"""
Bayesian localization for INFORM.

This module contains the core recruitment-matching routine used to infer the
location and dispersion of one functional fiber cluster.

The objective function is the negative mean-squared error between reference
recruitment curves and recruitment curves generated from a candidate functional
cluster. The objective is maximized with Bayesian Optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from modAL.acquisition import max_EI, optimizer_EI
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor

from nerve_model.experiment import Experiment


@dataclass
class LocalizationResult:
    """Output of one Bayesian localization run.

    Attributes
    ----------
    x_iter : list of ndarray
        Candidate parameters evaluated at each iteration, in standardized space.
    y_iter : list of list of float
        Objective values evaluated at each iteration.
    recruitment_curves_iter : list
        Recruitment curves generated for evaluated candidates.
    x_max : list of ndarray
        Best standardized candidate after each iteration.
    y_max : list of float
        Best objective value after each iteration.
    acquisition_iter : list of ndarray
        Acquisition function values over the candidate grid at each iteration.
    y_prior : list of list of float
        Objective values computed from prior/warm-start candidates.
    """

    x_iter: list[np.ndarray]
    y_iter: list[list[float]]
    recruitment_curves_iter: list[Any]
    x_max: list[np.ndarray]
    y_max: list[float]
    acquisition_iter: list[np.ndarray]
    y_prior: list[list[float]]

    @property
    def best_x(self) -> np.ndarray:
        """Best standardized candidate found by the optimizer."""
        return self.x_max[-1]

    @property
    def best_score(self) -> float:
        """Best objective value found by the optimizer."""
        return self.y_max[-1]


def _score_recruitment_curves(candidate_curves, reference_curves: np.ndarray, candidate_group_id: int = 0) -> float:
    """Compute negative MSE between candidate and reference recruitment curves."""
    candidate_values = candidate_curves.recruitment_values[
        :,
        candidate_group_id : candidate_group_id + 1,
        :,
    ]
    error = candidate_values - reference_curves
    return -float(np.mean(error**2))


def _teach_prior(
    optimizer: BayesianOptimizer,
    prior_info_x,
    prior_recruitment_curves,
    reference_curves: np.ndarray,
) -> list[list[float]]:
    """Warm-start the optimizer with previously evaluated candidates."""
    y_prior: list[list[float]] = []

    if prior_info_x is None:
        return y_prior

    for prior_cluster_idx in range(len(prior_info_x)):
        y_prior_current = []

        for candidate_idx in range(len(prior_info_x[prior_cluster_idx])):
            candidate_values = prior_recruitment_curves[prior_cluster_idx].recruitment_values[
                :,
                candidate_idx : candidate_idx + 1,
                :,
            ]
            error = candidate_values - reference_curves
            y_prior_current.append(-float(np.mean(error**2)))

        y_prior.append(y_prior_current)
        optimizer.teach(prior_info_x[prior_cluster_idx], y_prior_current)

    return y_prior


def _evaluate_candidate(
    candidate_index: int,
    experiment_info: dict[str, Any],
    reference_curves: np.ndarray,
    candidates_grid: np.ndarray,
):
    """Generate and score recruitment curves for one candidate."""
    full_experiment = experiment_info["full_experiment"]
    full_lfm = experiment_info["full_lfm"]
    amp_lims = experiment_info["amp_lims"]
    n_stims_per_site = experiment_info["n_stims_per_site"]
    activation_predictor = full_experiment.activation_predictor

    candidate = candidates_grid[candidate_index, :]

    candidate_experiment, identities = Experiment.from_existing_experiment(
        experiment=full_experiment,
        has_struct_info=True,
        cluster_locs=candidate[0:2],
        cluster_std=candidate[2],
        cluster_num=int(candidate[3]),
    )

    candidate_experiment.load_lead_field_matrix(
        identities=identities,
        lead_field_matrix=full_lfm,
        full_experiment=full_experiment,
    )

    candidate_experiment._activation_predictor = activation_predictor

    recruitment_curves = candidate_experiment.generate_recruitment_curves(
        amp_lims=amp_lims,
        n_steps=n_stims_per_site,
        method="from_self",
    )

    score = _score_recruitment_curves(
        candidate_curves=recruitment_curves,
        reference_curves=reference_curves,
        candidate_group_id=0,
    )

    return recruitment_curves, score


def localize_functional_cluster(
    experiment_info: dict[str, Any],
    reference_curves: np.ndarray,
    candidates_grid: np.ndarray,
    candidates_grid_standardized: np.ndarray,
    kernel,
    tradeoff: float = 0.01,
    max_iter: int = 100,
    batch_size: int = 1,
    prior_info_x=None,
    prior_recruitment_curves=None,
    initial_random_samples: int = 30,
    ei_stop_threshold: float = 1e-8,
    unchanged_best_patience: int = 15,
    random_state: int = 0,
) -> LocalizationResult:
    """Localize one functional cluster by Bayesian recruitment matching.

    Parameters
    ----------
    experiment_info : dict
        Dictionary containing:

        - ``"full_experiment"``: experiment with full structural information;
        - ``"full_lfm"``: lead-field matrix of the full experiment;
        - ``"amp_lims"``: stimulation amplitude limits;
        - ``"n_stims_per_site"``: number of amplitudes per stimulation site.

    reference_curves : ndarray
        Reference recruitment curves for the target cluster, with shape
        ``(n_sites, 1, n_amplitudes)``.
    candidates_grid : ndarray
        Candidate parameters in physical units. Columns are ``x``, ``y``,
        ``std``, and ``n_fibers``.
    candidates_grid_standardized : ndarray
        Standardized candidate parameters used by the Gaussian process.
    kernel : sklearn-compatible kernel
        Kernel used by the Gaussian-process regressor.
    tradeoff : float, default=0.01
        Expected Improvement exploration-exploitation tradeoff.
    max_iter : int, default=100
        Maximum number of Bayesian-optimization iterations.
    batch_size : int, default=1
        Number of candidates queried at each iteration after initialization.
    prior_info_x : list, optional
        Previously evaluated standardized candidates used as warm-start data.
    prior_recruitment_curves : list, optional
        Recruitment curves associated with ``prior_info_x``.
    initial_random_samples : int, default=30
        Number of random candidates evaluated for the first cluster.
    ei_stop_threshold : float, default=1e-8
        Stop when the maximum Expected Improvement falls below this value.
    unchanged_best_patience : int, default=15
        Stop when the best objective value remains unchanged for this many
        consecutive iterations.
    random_state : int, default=0
        Random seed used for initialization.

    Returns
    -------
    LocalizationResult
        Optimization history and best candidate information.
    """
    rng = np.random.default_rng(random_state)

    regressor = GaussianProcessRegressor(kernel=kernel, random_state=random_state)
    max_ei = partial(max_EI, tradeoff=tradeoff)
    optimizer_ei = partial(optimizer_EI, tradeoff=tradeoff)

    optimizer = BayesianOptimizer(
        estimator=regressor,
        query_strategy=max_ei,
    )

    y_prior = _teach_prior(
        optimizer=optimizer,
        prior_info_x=prior_info_x,
        prior_recruitment_curves=prior_recruitment_curves,
        reference_curves=reference_curves,
    )

    x_iter: list[np.ndarray] = []
    y_iter: list[list[float]] = []
    recruitment_curves_iter: list[Any] = []
    x_max: list[np.ndarray] = []
    y_max: list[float] = []
    acquisition_iter: list[np.ndarray] = []

    unchanged_best_count = 0

    for iter_idx in range(max_iter):
        if iter_idx == 0 and prior_info_x is None:
            n_initial = min(initial_random_samples, candidates_grid_standardized.shape[0])
            candidate_indices = rng.choice(
                candidates_grid_standardized.shape[0],
                size=n_initial,
                replace=False,
            )
        else:
            candidate_indices, _ = optimizer.query(
                candidates_grid_standardized,
                n_instances=batch_size,
            )

        candidate_indices = np.asarray(candidate_indices, dtype=int)
        x_current = candidates_grid_standardized[candidate_indices, :]
        x_iter.append(x_current)

        recruitment_curves_current = []
        scores_current = []

        for candidate_index in candidate_indices:
            recruitment_curves, score = _evaluate_candidate(
                candidate_index=candidate_index,
                experiment_info=experiment_info,
                reference_curves=reference_curves,
                candidates_grid=candidates_grid,
            )
            recruitment_curves_current.append(recruitment_curves)
            scores_current.append(score)

        recruitment_curves_iter.append(recruitment_curves_current)
        y_iter.append(scores_current)

        optimizer.teach(x_current, scores_current)

        x_best_current, y_best_current = optimizer.get_max()
        x_max.append(x_best_current)
        y_max.append(float(y_best_current))

        if iter_idx > 0 and y_max[-1] == y_max[-2]:
            unchanged_best_count += 1
        else:
            unchanged_best_count = 0

        acquisition_values = optimizer_ei(optimizer, candidates_grid_standardized)
        acquisition_iter.append(acquisition_values)

        if np.max(acquisition_values) < ei_stop_threshold:
            break

        if unchanged_best_count >= unchanged_best_patience:
            break

    return LocalizationResult(
        x_iter=x_iter,
        y_iter=y_iter,
        recruitment_curves_iter=recruitment_curves_iter,
        x_max=x_max,
        y_max=y_max,
        acquisition_iter=acquisition_iter,
        y_prior=y_prior,
    )


def performLocalizationCluster(
    experiment_info,
    refCurves,
    candidatesGrid,
    candidatesGridStandardized,
    kernel,
    tr=0.01,
    maxIter=100,
    batchSize=1,
    priorInfoX=None,
    rcPrior=None,
):
    """Backward-compatible wrapper for the original localization function."""
    result = localize_functional_cluster(
        experiment_info=experiment_info,
        reference_curves=refCurves,
        candidates_grid=candidatesGrid,
        candidates_grid_standardized=candidatesGridStandardized,
        kernel=kernel,
        tradeoff=tr,
        max_iter=maxIter,
        batch_size=batchSize,
        prior_info_x=priorInfoX,
        prior_recruitment_curves=rcPrior,
    )

    return (
        result.x_iter,
        result.y_iter,
        result.recruitment_curves_iter,
        result.x_max,
        result.y_max,
        result.acquisition_iter,
        result.y_prior,
    )
