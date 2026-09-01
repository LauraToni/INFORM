"""
Candidate generation for INFORM localization.

Candidate parameters are:
1. x-coordinate of the candidate functional cluster center;
2. y-coordinate of the candidate functional cluster center;
3. spatial standard deviation of the candidate cluster;
4. number of fibers in the candidate cluster.

All spatial quantities are expressed in millimeters.
"""

from __future__ import annotations

import numpy as np


def create_loc_candidates(
    nerve_radius,
    limCandidateStd,
    limCandidateNum,
    nTriesLocs,
    nTriesStd,
    nTriesNum,
):
    """Create the localization candidate grid.

    This keeps the original argument names and grid logic for compatibility
    with the validated INFORM localization notebooks.

    Returns
    -------
    ndarray
        Candidate grid with shape ``(n_candidates, 4)``. Columns are
        ``x``, ``y``, ``std``, and ``n_fibers``.
    """
    limCandidatePosX = [-nerve_radius, nerve_radius]
    limCandidatePosY = [-nerve_radius, nerve_radius]

    nTotalCandidates = nTriesLocs[0] * nTriesLocs[1] * nTriesStd * nTriesNum

    xCandidates = np.linspace(limCandidatePosX[0], limCandidatePosX[1], nTriesLocs[0])
    yCandidates = np.linspace(limCandidatePosY[0], limCandidatePosY[1], nTriesLocs[1])
    stdCandidates = np.linspace(limCandidateStd[0], limCandidateStd[1], nTriesStd)
    numCandidates = np.linspace(limCandidateNum[0], limCandidateNum[1], nTriesNum)

    xCandidates, yCandidates, stdCandidates, numCandidates = np.meshgrid(
        xCandidates,
        yCandidates,
        stdCandidates,
        numCandidates,
    )

    xCandidates = np.reshape(xCandidates, [nTotalCandidates, 1])
    yCandidates = np.reshape(yCandidates, [nTotalCandidates, 1])
    stdCandidates = np.reshape(stdCandidates, [nTotalCandidates, 1])
    numCandidates = np.reshape(numCandidates, [nTotalCandidates, 1])

    dist_from_origin = np.sqrt(xCandidates**2 + yCandidates**2)
    active_locs = dist_from_origin < nerve_radius

    candidatesGrid = np.hstack(
        (xCandidates, yCandidates, stdCandidates, numCandidates)
    )
    candidatesGrid = candidatesGrid[active_locs[:, 0], :]

    return candidatesGrid


def create_localization_candidates(
    nerve_radius: float,
    std_limits,
    num_limits,
    n_location_samples,
    n_std_samples: int,
    n_num_samples: int,
):
    """Alias with clearer argument names for new code."""
    return create_loc_candidates(
        nerve_radius=nerve_radius,
        limCandidateStd=std_limits,
        limCandidateNum=num_limits,
        nTriesLocs=n_location_samples,
        nTriesStd=n_std_samples,
        nTriesNum=n_num_samples,
    )


__all__ = [
    "create_loc_candidates",
    "create_localization_candidates",
]
