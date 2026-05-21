"""
Candidate generation for INFORM localization.

This module creates the finite candidate grid used by the Bayesian optimization
localization routine. Candidate parameters are:

1. cluster center x-coordinate;
2. cluster center y-coordinate;
3. cluster spatial standard deviation;
4. number of fibers in the cluster.

All spatial quantities are expressed in millimeters.
"""

from __future__ import annotations

import numpy as np


def create_localization_candidates(
    nerve_radius: float,
    std_limits: tuple[float, float],
    num_limits: tuple[int, int],
    n_location_samples: tuple[int, int],
    n_std_samples: int,
    n_num_samples: int,
) -> np.ndarray:
    """Create a candidate grid for functional-cluster localization.

    Parameters
    ----------
    nerve_radius : float
        Radius of the circular nerve section.
    std_limits : tuple of float
        Lower and upper limits for the candidate cluster standard deviation.
    num_limits : tuple of int
        Lower and upper limits for the candidate number of fibers.
    n_location_samples : tuple of int
        Number of candidate x and y locations.
    n_std_samples : int
        Number of candidate standard-deviation values.
    n_num_samples : int
        Number of candidate fiber-count values.

    Returns
    -------
    ndarray
        Candidate grid with shape ``(n_candidates, 4)``. Columns correspond to
        ``x``, ``y``, ``std``, and ``n_fibers``.
    """
    if nerve_radius <= 0:
        raise ValueError("nerve_radius must be positive.")

    if len(n_location_samples) != 2:
        raise ValueError("n_location_samples must contain two values: (n_x, n_y).")

    n_x, n_y = n_location_samples

    if min(n_x, n_y, n_std_samples, n_num_samples) <= 0:
        raise ValueError("All sampling counts must be positive.")

    x_candidates = np.linspace(-nerve_radius, nerve_radius, n_x)
    y_candidates = np.linspace(-nerve_radius, nerve_radius, n_y)
    std_candidates = np.linspace(std_limits[0], std_limits[1], n_std_samples)
    num_candidates = np.linspace(num_limits[0], num_limits[1], n_num_samples)

    x_grid, y_grid, std_grid, num_grid = np.meshgrid(
        x_candidates,
        y_candidates,
        std_candidates,
        num_candidates,
        indexing="ij",
    )

    candidates = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            std_grid.ravel(),
            num_grid.ravel(),
        )
    )

    distance_from_origin = np.sqrt(candidates[:, 0] ** 2 + candidates[:, 1] ** 2)
    inside_nerve = distance_from_origin < nerve_radius

    return candidates[inside_nerve]


def create_loc_candidates(
    nerve_radius,
    limCandidateStd,
    limCandidateNum,
    nTriesLocs,
    nTriesStd,
    nTriesNum,
):
    """Backward-compatible wrapper for older localization notebooks.

    This keeps the original function name and argument names while delegating to
    :func:`create_localization_candidates`.
    """
    return create_localization_candidates(
        nerve_radius=nerve_radius,
        std_limits=tuple(limCandidateStd),
        num_limits=tuple(limCandidateNum),
        n_location_samples=tuple(nTriesLocs),
        n_std_samples=nTriesStd,
        n_num_samples=nTriesNum,
    )
