"""
Clean interface layer for INFORM cluster localization.

This module provides a small, readable wrapper around the validated
``performLocalizationCluster`` routine defined in
``localization_utils_reference.py``. It does NOT change the scientific logic:
it only calls the original function and packages its outputs into a structured
result object.

Rationale
---------
The reference implementation returns a 7-tuple of parallel lists whose meaning
is positional and easy to misuse. New code (and the smoke test) can instead use
``localize_functional_cluster`` and read named attributes from the returned
``LocalizationResult``. Existing scripts that call ``performLocalizationCluster``
directly are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Import directly from the reference module (not from bayesian_localization) to
# avoid a circular import: bayesian_localization re-exports this module's names.
from .localization_utils_reference import performLocalizationCluster


@dataclass
class LocalizationResult:
    """Structured result of a single-cluster Bayesian localization run.

    Attributes
    ----------
    x_iter, y_iter, rc_iter
        Per-iteration queried candidates (standardized), their objective values
        (negative MSE), and the associated recruitment curves.
    x_max, y_max
        Running best candidate (standardized) and best objective value per
        iteration.
    acq_iter
        Acquisition-function values per iteration.
    y_prior
        Objective values inherited from previously optimized clusters
        (warm start); empty for the first cluster.
    best_candidate_standardized
        Standardized parameter vector of the final best candidate.
    best_score
        Final best objective value (negative MSE; higher is better).
    """

    x_iter: list[Any] = field(default_factory=list)
    y_iter: list[Any] = field(default_factory=list)
    rc_iter: list[Any] = field(default_factory=list)
    x_max: list[Any] = field(default_factory=list)
    y_max: list[Any] = field(default_factory=list)
    acq_iter: list[Any] = field(default_factory=list)
    y_prior: list[Any] = field(default_factory=list)

    @property
    def best_candidate_standardized(self) -> np.ndarray:
        """Standardized parameter vector of the final best candidate."""
        return np.asarray(self.x_max[-1]).reshape(-1)

    @property
    def best_score(self) -> float:
        """Final best objective value (negative MSE)."""
        return float(self.y_max[-1])

    def to_reference_tuple(self) -> tuple:
        """Return the original 7-tuple layout for backward compatibility."""
        return (
            self.x_iter,
            self.y_iter,
            self.rc_iter,
            self.x_max,
            self.y_max,
            self.acq_iter,
            self.y_prior,
        )

    @classmethod
    def from_reference_tuple(cls, outputs: tuple) -> "LocalizationResult":
        """Build a result object from the reference function's raw output."""
        x_iter, y_iter, rc_iter, x_max, y_max, acq_iter, y_prior = outputs
        return cls(
            x_iter=x_iter,
            y_iter=y_iter,
            rc_iter=rc_iter,
            x_max=x_max,
            y_max=y_max,
            acq_iter=acq_iter,
            y_prior=y_prior,
        )


def localize_functional_cluster(
    experiment_info: dict,
    reference_curves: np.ndarray,
    candidates_grid: np.ndarray,
    candidates_grid_standardized: np.ndarray,
    kernel,
    tradeoff: float = 0.01,
    max_iter: int = 20,
    batch_size: int = 1,
    prior_info_x=None,
    prior_recruitment_curves=None,
    initial_random_samples: int = 30,
    random_state: int | None = None,
) -> LocalizationResult:
    """Localize a single functional cluster (clean wrapper).

    This is a thin, readable wrapper around
    :func:`performLocalizationCluster`. All scientific behavior is delegated to
    that validated routine; this function only renames arguments and packages
    the output into a :class:`LocalizationResult`.

    Parameters
    ----------
    experiment_info : dict
        Must contain ``full_experiment``, ``full_lfm``, ``amp_lims`` and
        ``n_stims_per_site`` (see the reference implementation).
    reference_curves : ndarray
        Target recruitment curves for the cluster being localized, with shape
        ``(n_sites, 1, n_amplitudes)``.
    candidates_grid, candidates_grid_standardized : ndarray
        Physical and z-scored candidate grids, shape ``(n_candidates, 4)``.
    kernel : sklearn kernel
        Gaussian Process kernel (e.g. Matern 5/2).
    tradeoff : float
        Exploration/exploitation tradeoff of the Expected Improvement.
    max_iter : int
        Maximum number of Bayesian-optimization iterations.
    batch_size : int
        Number of candidates queried per iteration.
    prior_info_x, prior_recruitment_curves
        Warm-start information inherited from previously optimized clusters.
    initial_random_samples : int
        Kept for API clarity; the reference routine seeds the first cluster
        with 30 random samples internally. Provided here for documentation and
        forward compatibility.
    random_state : int, optional
        If provided, seeds NumPy's global RNG before the run for reproducibility
        of the initial random seeding. The reference GP itself already uses
        ``random_state=0`` internally.

    Returns
    -------
    LocalizationResult
    """
    if random_state is not None:
        np.random.seed(random_state)

    outputs = performLocalizationCluster(
        experiment_info=experiment_info,
        refCurves=reference_curves,
        candidatesGrid=candidates_grid,
        candidatesGridStandardized=candidates_grid_standardized,
        kernel=kernel,
        tr=tradeoff,
        maxIter=max_iter,
        batchSize=batch_size,
        priorInfoX=prior_info_x,
        rcPrior=prior_recruitment_curves,
    )

    return LocalizationResult.from_reference_tuple(outputs)


__all__ = ["LocalizationResult", "localize_functional_cluster"]
