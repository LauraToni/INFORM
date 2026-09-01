"""
Bayesian localization interface for INFORM.

This module exposes the paper/reference localization functions while preserving
the original batch-based Bayesian optimization behavior.

Important
---------
The scientific implementation lives in ``localization_utils_reference.py``.
This file is intentionally a thin interface layer so that downstream code can
import from ``localization.bayesian_localization`` without changing the
validated localization logic.

For a structured, readable API around the reference routine, see
``localization.localization_result.localize_functional_cluster``, which is also
re-exported here for convenience.
"""

from __future__ import annotations

from .localization_utils_reference import (
    performLocalizationCluster,
    performLocalizationClusterParallel,
    Experiment,
)

# Re-exported so that ``from localization.bayesian_localization import
# localize_functional_cluster`` works (used by the smoke test and new code).
# Imported at module end to avoid a circular import at package load time.
from .localization_result import (  # noqa: E402
    LocalizationResult,
    localize_functional_cluster,
)

__all__ = [
    "performLocalizationCluster",
    "performLocalizationClusterParallel",
    "LocalizationResult",
    "localize_functional_cluster",
]
