"""
Utility functions for INFORM localization workflows.

These functions support data extraction, lead-field matrix reshaping, and
empirical recomputation of cluster descriptors from sampled fiber locations.
They are not part of the Bayesian optimization core.
"""

from __future__ import annotations

import numpy as np


def extract_experiment_data(data: dict):
    """Extract common arrays and dimensions from a loaded simulation dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing at least ``lead_field_matrix``,
        ``n_fem_nodes_per_fiber``, and ``diameters``.

    Returns
    -------
    tuple
        ``lead_field_matrix``, ``n_fibers``, ``n_sites``, ``n_nodes``,
        and ``diameters``.
    """
    lead_field_matrix = data["lead_field_matrix"]
    n_fem_nodes_per_fiber = data["n_fem_nodes_per_fiber"][0].astype(np.int32)
    diameters = data["diameters"]

    n_fibers = n_fem_nodes_per_fiber.shape[0]
    n_sites = lead_field_matrix.shape[1]
    n_internodes = 41 - 1
    n_nodes = n_internodes * 11 + 1

    return lead_field_matrix, n_fibers, n_sites, n_nodes, diameters


def generate_lfm_per_fiber(
    lead_field_matrix: np.ndarray,
    n_fibers: int,
    n_nodes: int,
    n_sites: int,
    n_fem_nodes_per_fiber: np.ndarray,
    scale_factor: float = 1e3,
) -> np.ndarray:
    """Convert a FEM lead-field matrix into a fiber-wise representation.

    Parameters
    ----------
    lead_field_matrix : ndarray
        FEM lead-field matrix with shape ``(n_fem_nodes, n_sites)``.
    n_fibers : int
        Number of fibers.
    n_nodes : int
        Number of model nodes per fiber.
    n_sites : int
        Number of stimulation sites.
    n_fem_nodes_per_fiber : ndarray
        Number of FEM nodes associated with each fiber.
    scale_factor : float, default=1e3
        Multiplicative scale factor applied to the lead-field values.

    Returns
    -------
    ndarray
        Lead-field matrix organized by fiber, with shape
        ``(n_fibers, n_nodes, n_sites)``.
    """
    lead_field_per_fiber = np.zeros((n_fibers, n_nodes, n_sites))
    n_max_nodes = np.max(n_fem_nodes_per_fiber)
    current_node = 0

    for fiber_idx in range(n_fibers):
        selected_nodes = np.arange(
            current_node,
            current_node + n_fem_nodes_per_fiber[fiber_idx],
        )

        n_selected_nodes = selected_nodes.size
        first_node = int((n_max_nodes - n_fem_nodes_per_fiber[fiber_idx]) / 2)

        lead_field_per_fiber[
            fiber_idx,
            first_node : first_node + n_selected_nodes,
            :,
        ] = lead_field_matrix[selected_nodes, :] * scale_factor

        current_node += n_fem_nodes_per_fiber[fiber_idx]

    return lead_field_per_fiber


def recompute_cluster_mean_std(fiber_population):
    """Recompute empirical cluster centers and radial dispersions.

    This is useful when sampled fibers do not exactly match the nominal Gaussian
    parameters used to generate a functional cluster.

    Parameters
    ----------
    fiber_population : MotorFiberPopulation-like object
        Fiber population exposing ``locs``, ``cluster_ids``, and ``cluster_num``.

    Returns
    -------
    tuple of ndarray
        Empirical x-centers, y-centers, and radial standard deviations for each
        cluster.
    """
    n_clusters = len(fiber_population.cluster_num)

    mean_locs_x = np.zeros(n_clusters)
    mean_locs_y = np.zeros(n_clusters)
    radial_std = np.zeros(n_clusters)

    radial_distance = np.sqrt(
        fiber_population.locs[:, 0] ** 2 + fiber_population.locs[:, 1] ** 2
    )

    for cluster_idx in range(n_clusters):
        idx = np.where(fiber_population.cluster_ids == cluster_idx)

        mean_locs_x[cluster_idx] = np.mean(fiber_population.locs[idx, 0])
        mean_locs_y[cluster_idx] = np.mean(fiber_population.locs[idx, 1])
        radial_std[cluster_idx] = np.std(radial_distance[idx])

    return mean_locs_x, mean_locs_y, radial_std


def recompute_mean_std(full_experiment, true_experiment, true_population, show: bool = False):
    """Backward-compatible wrapper for the original notebook function.

    The plotting option is intentionally ignored here; plotting should live in
    the visualization module.
    """
    return recompute_cluster_mean_std(true_population)


# Backward-compatible aliases for older notebooks.
get_data = extract_experiment_data
generate_lfm_fiber = generate_lfm_per_fiber
