"""
INFORM localization package.

Bayesian localization framework used to infer the functional organization of a
peripheral nerve from recruitment matching.

The scientific implementation lives in ``localization_utils_reference.py`` and
is exposed through ``bayesian_localization``. A thin, readable interface
(``localize_functional_cluster`` / ``LocalizationResult``) is provided in
``localization_result`` for new code and testing; it does not alter the
validated optimization logic.
"""

from __future__ import annotations

from .candidate_generation import (
    create_loc_candidates,
    create_localization_candidates,
)
from .bayesian_localization import (
    performLocalizationCluster,
    performLocalizationClusterParallel,
)
from .localization_result import (
    LocalizationResult,
    localize_functional_cluster,
)

__all__ = [
    # candidate generation
    "create_loc_candidates",
    "create_localization_candidates",
    # validated reference routines (original names)
    "performLocalizationCluster",
    "performLocalizationClusterParallel",
    # clean interface layer
    "LocalizationResult",
    "localize_functional_cluster",
]
