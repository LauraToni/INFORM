"""
Selectivity objective functions for INFORM stimulation optimization.

These utilities convert optimizer parameters into stimulation protocols and
evaluate selectivity from predicted recruitment patterns.
"""

from __future__ import annotations

import numpy as np


def params_to_stimulation_protocol(
    params: np.ndarray,
    n_sites: int,
    n_active_sites: int,
    batch_size: int,
) -> np.ndarray:
    """Convert optimizer parameters to sparse stimulation protocols.

    Parameters
    ----------
    params : ndarray
        Optimizer parameters with shape ``(batch_size, 2 * n_sites)`` after
        reshaping. The first ``n_sites`` columns encode amplitudes, while the
        second ``n_sites`` columns encode ranking weights.
    n_sites : int
        Number of stimulation sites.
    n_active_sites : int
        Number of non-zero stimulation sites retained in each protocol.
    batch_size : int
        Number of stimulation protocols encoded in ``params``.

    Returns
    -------
    ndarray
        Stimulation protocols with shape ``(batch_size, n_sites)``.
    """
    params = np.reshape(params, [batch_size, n_sites * 2])
    stimulation_protocols = np.zeros((batch_size, n_sites))

    for batch_idx in range(batch_size):
        ranked_sites = np.argsort(params[batch_idx, n_sites:])[::-1]

        for active_idx in range(n_active_sites):
            site_idx = ranked_sites[active_idx]
            stimulation_protocols[batch_idx, site_idx] = params[batch_idx, site_idx]

    return stimulation_protocols


def compute_selectivity_from_recruitment(
    recruitment_patterns: np.ndarray,
    target_group: int,
) -> np.ndarray:
    """Compute selectivity for one target group from recruitment patterns.

    The metric is:

    ``target_recruitment ** 2 / total_recruitment``

    with selectivity set to zero when total recruitment is zero.

    Parameters
    ----------
    recruitment_patterns : ndarray
        Recruitment fractions with shape ``(batch_size, n_groups)``.
    target_group : int
        Index of the target functional group.

    Returns
    -------
    ndarray
        Selectivity values with shape ``(batch_size,)``.
    """
    total_recruitment = np.sum(recruitment_patterns, axis=1)
    selectivity = np.zeros(recruitment_patterns.shape[0])

    active = total_recruitment > 0
    selectivity[active] = (
        recruitment_patterns[active, target_group] ** 2
        / total_recruitment[active]
    )

    return selectivity


def params_to_selectivity_objective(
    params,
    n_sites: int,
    n_active_sites: int,
    experiment,
    target_group: int,
    batch_size: int,
    return_recruitment: bool = False,
):
    """Evaluate negative selectivity from optimizer parameters.

    This objective is designed for optimizers that minimize the objective
    function. Therefore, the returned value is ``-selectivity``.

    Parameters
    ----------
    params : ndarray
        Optimizer parameters encoding stimulation amplitudes and site-ranking
        weights.
    n_sites : int
        Number of stimulation sites.
    n_active_sites : int
        Number of active sites retained in each protocol.
    experiment : Experiment
        Experiment object used to compute recruitment patterns.
    target_group : int
        Functional group for which selectivity is optimized.
    batch_size : int
        Number of protocols encoded in ``params``.
    return_recruitment : bool, default=False
        If true, return stimulation protocols, recruitment patterns, and
        selectivity instead of the negative objective.

    Returns
    -------
    ndarray or tuple
        Negative selectivity values, or detailed outputs when
        ``return_recruitment=True``.
    """
    stimulation_protocols = params_to_stimulation_protocol(
        params=params,
        n_sites=n_sites,
        n_active_sites=n_active_sites,
        batch_size=batch_size,
    )

    recruitment_patterns = experiment.compute_recruitment_patterns(
        stimulation_protocols=stimulation_protocols,
        method="from_self",
    )

    selectivity = compute_selectivity_from_recruitment(
        recruitment_patterns=recruitment_patterns,
        target_group=target_group,
    )

    if return_recruitment:
        return stimulation_protocols, recruitment_patterns, selectivity

    return -selectivity


def params_to_selectivity_tf(
    params,
    n_sites,
    n_active_sites,
    true_population,
    experiment,
    musc_selective,
    batch_size,
    return_recruitment=False,
):
    """Backward-compatible wrapper for the original objective function.

    The ``true_population`` argument is retained for compatibility with older
    notebooks but is not used by the current implementation.
    """
    return params_to_selectivity_objective(
        params=params,
        n_sites=n_sites,
        n_active_sites=n_active_sites,
        experiment=experiment,
        target_group=musc_selective,
        batch_size=batch_size,
        return_recruitment=return_recruitment,
    )
