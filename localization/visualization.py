"""
Visualization utilities for INFORM localization.

These functions are kept outside the Bayesian localization core so that the
algorithm can be imported and executed without plotting dependencies.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nerve_model.experiment import Experiment


DEFAULT_CLUSTER_COLORS = [
    "#1f78b4",
    "#ee7674",
    "#F6BD60",
    "#8dd3c7",
    "#e31a1c",
    "#cab2d6",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]


def plot_recruitment_curves(
    recruitment_values: np.ndarray,
    amplitudes: np.ndarray,
    n_sites: int,
    n_clusters: int,
    colors=None,
    title: str | None = None,
):
    """Plot recruitment curves for each stimulation site.

    Parameters
    ----------
    recruitment_values : ndarray
        Recruitment values with shape ``(n_sites, n_clusters, n_amplitudes)``.
    amplitudes : ndarray
        Stimulation amplitudes.
    n_sites : int
        Number of stimulation sites.
    n_clusters : int
        Number of functional clusters.
    colors : list, optional
        Colors used for cluster curves.
    title : str, optional
        Figure title.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    colors = colors or DEFAULT_CLUSTER_COLORS

    n_cols = int(np.ceil(np.sqrt(n_sites)))
    n_rows = int(np.ceil(n_sites / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, layout="constrained", squeeze=False)
    axes_flat = axes.flatten()

    for site_idx in range(n_sites):
        ax = axes_flat[site_idx]

        for cluster_idx in range(n_clusters):
            ax.plot(
                amplitudes,
                recruitment_values[site_idx, cluster_idx, :].T,
                color=colors[cluster_idx],
            )

        ax.set_title(f"{site_idx + 1}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Recruitment")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes_flat[n_sites:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    return fig, axes


def plot_recruitment_superposed(
    true_recruitment_curves: np.ndarray,
    pred_recruitment_curves: np.ndarray,
    amplitudes: np.ndarray,
    n_sites: int,
    n_clusters: int,
    colors=None,
    title: str | None = None,
):
    """Plot true and predicted recruitment curves on the same axes.

    True curves are plotted as solid lines; predicted curves are plotted as
    dashed lines.
    """
    colors = colors or DEFAULT_CLUSTER_COLORS

    n_cols = int(np.ceil(np.sqrt(n_sites)))
    n_rows = int(np.ceil(n_sites / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, layout="constrained", squeeze=False)
    axes_flat = axes.flatten()

    for site_idx in range(n_sites):
        ax = axes_flat[site_idx]

        for cluster_idx in range(n_clusters):
            ax.plot(
                amplitudes,
                true_recruitment_curves[site_idx, cluster_idx, :].T,
                color=colors[cluster_idx],
            )
            ax.plot(
                amplitudes,
                pred_recruitment_curves[site_idx, cluster_idx, :].T,
                "--",
                color=colors[cluster_idx],
            )

        ax.set_title(f"{site_idx + 1}")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes_flat[n_sites:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, size=20)

    return fig, axes


def plot_section(
    experiment,
    ax,
    colors=None,
    activation=None,
    cluster_centers=None,
    fiber_population=None,
    radius: float = 2,
    marker_color: str = "red",
):
    """Plot a nerve section with fibers, implant sites, and cluster centers.

    Parameters
    ----------
    experiment : Experiment-like object
        Experiment containing nerve topography and implant.
    ax : matplotlib axis
        Axis on which to plot.
    colors : list, optional
        Cluster colors.
    activation : ndarray, optional
        Binary activation vector used to gray out inactive fibers.
    cluster_centers : ndarray, optional
        Cluster centers with shape ``(n_clusters, 2)``.
    fiber_population : MotorFiberPopulation-like object, optional
        Fiber population to plot. Defaults to ``experiment.fiber_population``.
    radius : float, default=2
        Radius used for the background circle.
    marker_color : str, default="red"
        Color for stimulation site markers.

    Returns
    -------
    matplotlib axis
        Updated axis.
    """
    colors = colors or DEFAULT_CLUSTER_COLORS

    nerve_background = plt.Circle(
        (0.0, 0.0),
        radius,
        fill=True,
        edgecolor="none",
        facecolor="#f3f5f7",
        label="Nerve section",
    )
    ax.add_artist(nerve_background)

    if fiber_population is None:
        fiber_population = experiment.fiber_population

    n_fibers = len(fiber_population.locs)
    fiber_colors = [colors[int(fiber_population.cluster_ids[i])] for i in range(n_fibers)]

    if activation is not None:
        fiber_colors = [
            fiber_colors[idx] if is_active else "#ced4da"
            for idx, is_active in enumerate(activation)
        ]

    ax.scatter(
        fiber_population.locs[:, 0],
        fiber_population.locs[:, 1],
        c=fiber_colors,
        alpha=0.3,
        zorder=1,
    )

    ax.scatter(
        experiment.implant.site_locs[:, 0],
        experiment.implant.site_locs[:, 1],
        c=marker_color,
        edgecolor="white",
        marker="^",
        s=100,
        label="Sites",
    )

    if experiment.nerve_topography is not None:
        experiment.nerve_topography.plot(ax=ax)

    if cluster_centers is not None:
        for cluster_idx, center in enumerate(cluster_centers):
            ax.scatter(
                center[0],
                center[1],
                c=colors[cluster_idx],
                label=f"Cluster {cluster_idx + 1}",
                s=40,
                edgecolors="black",
            )

    ax.set_ylim(-radius - 0.3, radius + 0.3)
    ax.set_xlim(-radius - 0.3, radius + 0.3)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, width=2)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_aspect("equal")

    return ax


def plot_population(pop_base_path, full_experiment, pop_lfm_filename, n_pop, show: bool = True):
    """Load and optionally plot one saved population.

    This function is mainly kept for compatibility with older notebooks.
    """
    pop_base_path = Path(pop_base_path)

    with open(pop_base_path / f"Pop{n_pop}.pkl", "rb") as file:
        true_population, true_identities = pickle.load(file)

    true_identities = true_identities.astype(int)
    true_experiment = Experiment(
        fiber_population=true_population,
        nerve_topography=full_experiment.nerve_topography,
        implant=full_experiment.implant,
    )

    true_experiment.load_lead_field_matrix(
        hdf5_file_path=pop_base_path / pop_lfm_filename,
        full_experiment=full_experiment,
        identities=true_identities,
    )

    if show:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plot_section(
            experiment=full_experiment,
            fiber_population=true_population,
            cluster_centers=true_population.cluster_locs,
            ax=ax,
            marker_color="black",
        )
        full_experiment.nerve_topography.plot(ax=ax)
        plt.show()

    return true_experiment, true_population, true_identities


def off_diagonal_frobenius_norm(matrix: np.ndarray) -> tuple[float, float]:
    """Compute off-diagonal Frobenius norm and its ratio to the full norm."""
    matrix = np.asarray(matrix)

    full_norm = np.linalg.norm(matrix)
    off_diagonal_mask = np.ones(matrix.shape, dtype=bool)
    np.fill_diagonal(off_diagonal_mask, False)

    off_diagonal_norm = np.linalg.norm(matrix[off_diagonal_mask])
    ratio = off_diagonal_norm / full_norm if full_norm != 0 else 0.0

    return float(off_diagonal_norm), float(ratio)


def plot_matrix(matrix: np.ndarray, size: int, title: str):
    """Plot a lower-triangular colored matrix.

    This is useful for visualizing cross-recruitment or selectivity matrices.
    """
    _, frobenius_ratio = off_diagonal_frobenius_norm(matrix)

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("white")

    for row_idx in range(size):
        for col_idx in range(size):
            intensity = matrix[col_idx, row_idx]
            color = DEFAULT_CLUSTER_COLORS[row_idx]

            ax.add_patch(
                plt.Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    color=color,
                    alpha=float(intensity),
                    ec="white",
                    lw=2,
                )
            )

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{matrix[col_idx, row_idx]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_xticklabels(np.arange(1, size + 1))
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_yticklabels(np.arange(1, size + 1))
    ax.set_title(f"{title}{frobenius_ratio:.3f}", size=15)

    return fig


# Backward-compatible aliases.
plot_recruitment_extended = plot_recruitment_curves
