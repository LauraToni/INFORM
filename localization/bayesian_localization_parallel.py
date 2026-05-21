"""
Parallel Bayesian localization for INFORM.

This module provides a parallel version of the single-cluster localization
routine. It reuses the same recruitment-matching objective as
``bayesian_localization.py`` but evaluates queried candidates with joblib.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from modAL.acquisition import max_EI, optimizer_EI
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor

from bayesian_localization import (
    LocalizationResult,
    _evaluate_candidate,
    _teach_prior,
)


def localize_functional_cluster_parallel(
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
    ei_stop_threshold: float = 1e-10,
    unchanged_best_patience: int = 15,
    improvement_tolerance: float | None = 1e-3,
    improvement_patience: int = 5,
    n_jobs: int = -1,
    random_state: int = 0,
) -> LocalizationResult:
    """Localize one functional cluster with parallel candidate evaluation.

    Parameters
    ----------
    experiment_info : dict
        Dictionary containing the full experiment, full lead-field matrix,
        amplitude limits, and number of stimulation amplitudes per site.
    reference_curves : ndarray
        Reference recruitment curves for the target cluster, typically with
        shape ``(n_sites, 1, n_amplitudes)``.
    candidates_grid : ndarray
        Candidate parameters in physical units.
    candidates_grid_standardized : ndarray
        Standardized candidate parameters used by the Gaussian process.
    kernel : sklearn-compatible kernel
        Kernel used by the Gaussian-process regressor.
    tradeoff : float, default=0.01
        Expected Improvement exploration-exploitation tradeoff.
    max_iter : int, default=100
        Maximum number of Bayesian-optimization iterations.
    batch_size : int, default=1
        Number of queried candidates per iteration.
    prior_info_x : list, optional
        Previously evaluated standardized candidates used for warm-starting.
    prior_recruitment_curves : list, optional
        Recruitment curves associated with ``prior_info_x``.
    initial_random_samples : int, default=30
        Number of random candidates evaluated for the first cluster.
    ei_stop_threshold : float, default=1e-10
        Stop when the maximum Expected Improvement falls below this threshold.
    unchanged_best_patience : int, default=15
        Stop when the best observed value does not change for this many
        consecutive iterations.
    improvement_tolerance : float, optional
        Stop when the absolute improvement remains below this tolerance for
        ``improvement_patience`` iterations.
    improvement_patience : int, default=5
        Patience for improvement-based early stopping.
    n_jobs : int, default=-1
        Number of joblib workers. ``-1`` uses all available cores.
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
    no_improvement_count = 0

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

        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_candidate)(
                candidate_index=candidate_index,
                experiment_info=experiment_info,
                reference_curves=reference_curves,
                candidates_grid=candidates_grid,
            )
            for candidate_index in candidate_indices
        )

        recruitment_curves_current, scores_current = zip(*results)
        scores_current = [float(score) for score in scores_current]

        recruitment_curves_iter.append(list(recruitment_curves_current))
        y_iter.append(scores_current)

        optimizer.teach(x_current, scores_current)

        x_best_current, y_best_current = optimizer.get_max()
        x_max.append(x_best_current)
        y_max.append(float(y_best_current))

        if iter_idx > 0 and y_max[-1] == y_max[-2]:
            unchanged_best_count += 1
        else:
            unchanged_best_count = 0

        if improvement_tolerance is not None and iter_idx > 0:
            if abs(y_max[-1] - y_max[-2]) < improvement_tolerance:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

        acquisition_values = optimizer_ei(optimizer, candidates_grid_standardized)
        acquisition_iter.append(acquisition_values)

        if np.max(acquisition_values) < ei_stop_threshold:
            break

        if unchanged_best_count >= unchanged_best_patience:
            break

        if improvement_tolerance is not None and no_improvement_count >= improvement_patience:
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


def performLocalizationClusterParallel(
    experiment_info,
    refCurves,
    candidatesGrid,
    candidatesGridStandardized,
    kernel,
    tr,
    maxIter=100,
    batchSize=1,
    priorInfoX=None,
    rcPrior=None,
    tol=1e-3,
    patience=5,
    n_jobs=-1,
):
    """Backward-compatible wrapper for the original parallel localization call."""
    result = localize_functional_cluster_parallel(
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
        improvement_tolerance=tol,
        improvement_patience=patience,
        n_jobs=n_jobs,
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
