"""
INFORM localization package.

This package contains the Bayesian localization framework used to infer
functional organization from recruitment matching.
"""

from .candidate_generation import (
    create_localization_candidates,
    create_loc_candidates,
)

from .bayesian_localization import (
    LocalizationResult,
    localize_functional_cluster,
    performLocalizationCluster,
)

from .bayesian_localization_parallel import (
    localize_functional_cluster_parallel,
    performLocalizationClusterParallel,
)

from .localization_utils import (
    extract_experiment_data,
    generate_lfm_per_fiber,
    recompute_cluster_mean_std,
)

from .selectivity_objective import (
    params_to_selectivity_objective,
    compute_selectivity_from_recruitment,
)

__all__ = [
    "LocalizationResult",
    "create_localization_candidates",
    "create_loc_candidates",
    "localize_functional_cluster",
    "localize_functional_cluster_parallel",
    "performLocalizationCluster",
    "performLocalizationClusterParallel",
    "extract_experiment_data",
    "generate_lfm_per_fiber",
    "recompute_cluster_mean_std",
    "params_to_selectivity_objective",
    "compute_selectivity_from_recruitment",
]
