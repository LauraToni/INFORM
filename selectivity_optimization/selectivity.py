"""
Selectivity optimization for INFORM.

This module packages the selectivity logic used in the CIRCLE/MEDIAN notebooks
into a reusable form, without changing the scientific behavior.

Two selectivity metrics are provided. The naming here describes the ROLE each
metric plays in this codebase, which is what the code actually does:

- ``selectivity_opt``  : the metric MAXIMIZED during protocol optimization
                         (the "rasp"/margin form:  r_target - off_target/(M-1)).
- ``selectivity_eval`` : the metric used to EVALUATE/visualize results
                         (the squared-ratio form:  r_i^2 / sum_j r_j).

The Particle Swarm Optimization (PSO) objective ``params_to_selectivity_rasp``
optimizes ``selectivity_opt`` via pyswarms' GlobalBestPSO. Nothing is rescaled
(no legacy current factor is applied).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Selectivity metrics
# ---------------------------------------------------------------------------
def selectivity_opt(recruitment_pattern, target_cluster, eps=1e-12):
    """Optimization metric (margin form) for a single stimulation protocol.

    Parameters
    ----------
    recruitment_pattern : ndarray, shape (n_clusters,)
        Recruitment fraction of each cluster for one stimulation protocol.
    target_cluster : int
        Index of the cluster to be selectively activated.
    eps : float
        Guard value (unused for division here; kept for API symmetry).

    Returns
    -------
    float
        Selectivity of the target cluster:  r_target - off_target/(M-1).
        Returns 0.0 when there is a single cluster or no activation.
    """
    rc = np.asarray(recruitment_pattern, dtype=float)
    n_clusters = rc.shape[0]
    if n_clusters <= 1 or np.sum(rc) == 0:
        return 0.0
    r_target = rc[target_cluster]
    off_target = np.sum(rc) - r_target
    return float(r_target - off_target / (n_clusters - 1))


def selectivity_eval(rc_array, normalize=True, squared=True, eps=1e-12):
    """Evaluation metric (squared-ratio form) from a recruitment matrix.

    Parameters
    ----------
    rc_array : ndarray, shape (n_clusters, n_clusters)
        Recruitment matrix; ``rc[i, j]`` = response of cluster ``i`` when the
        protocol optimized for cluster ``j`` is applied (diagonal = on-target).
    normalize : bool
        If True, divide the whole matrix by its maximum before computing.
    squared : bool
        If True, use ``rc[i, i]**2`` in the numerator (the form used to report
        results). If False, use ``rc[i, i]``.
    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    ndarray, shape (n_clusters,)
        Per-cluster selectivity, clipped to [0, 1].
    """
    rc = np.copy(np.asarray(rc_array, dtype=float))
    if normalize:
        max_val = np.max(rc)
        if max_val > 0:
            rc /= max_val

    n_clusters = rc.shape[0]
    selectivity = np.zeros(n_clusters)
    for i in range(n_clusters):
        denom = np.sum(rc[i, :]) + eps
        num = rc[i, i] ** 2 if squared else rc[i, i]
        selectivity[i] = np.clip(num / denom, 0, 1)
    return selectivity


# ---------------------------------------------------------------------------
# PSO objective
# ---------------------------------------------------------------------------
def params_to_selectivity_rasp(
    params,
    n_sites,
    n_active_sites,
    true_population,
    experiment,
    musc_selective,
    batch_size,
    return_recruitment=False,
):
    """PSO objective: selectivity (opt metric) for candidate protocols.

    ``params`` encodes, for each particle, ``2 * n_sites`` values: the first
    ``n_sites`` are stimulation amplitudes, the last ``n_sites`` are ranking
    weights. Only the top ``n_active_sites`` ranked electrodes are kept active.

    Returns the selectivity of ``musc_selective`` for each particle. This value
    is MAXIMIZED; when handing it to a minimizing optimizer, negate it (see
    ``run_pso_selectivity``).
    """
    params = np.reshape(params, [batch_size, n_sites * 2])
    stimulation_protocol = np.zeros((batch_size, n_sites))

    for j in range(batch_size):
        # rank electrodes by the weight half of params
        ranking = np.argsort(params[j, n_sites:])[::-1]
        # activate only the top n_active_sites electrodes
        for i in range(n_active_sites):
            stimulation_protocol[j, ranking[i]] = params[j, ranking[i]]

    recruitment_curves = experiment.compute_recruitment_patterns(
        stimulation_protocols=stimulation_protocol,
        method="from_self",
    )

    selectivity = np.zeros(batch_size)
    for b in range(batch_size):
        selectivity[b] = selectivity_opt(recruitment_curves[b, :], musc_selective)

    if return_recruitment:
        return stimulation_protocol, recruitment_curves, selectivity
    return selectivity


def run_pso_selectivity(
    experiment,
    true_population,
    target_cluster,
    n_active_sites,
    min_stim,
    max_stim,
    n_particles=25,
    n_iters=50,
    options=None,
):
    """Optimize a stimulation protocol for one target cluster via GlobalBestPSO.

    Thin wrapper around pyswarms that maximizes ``selectivity_opt`` (by
    minimizing its negative). Bounds and the top-k active-site encoding follow
    the notebook logic. Returns ``(best_cost, best_pos)`` from pyswarms.
    """
    from pyswarms.single import GlobalBestPSO

    if options is None:
        options = {"c1": 1.5, "c2": 1.5, "w": 0.9}

    n_sites = experiment.implant.n_sites

    # first half: amplitudes in [min_stim, max_stim]; second half: weights [0,1]
    lower = np.concatenate([np.full(n_sites, min_stim), np.zeros(n_sites)])
    upper = np.concatenate([np.full(n_sites, max_stim), np.ones(n_sites)])
    bounds = (lower, upper)

    def objective(x):
        # pyswarms minimizes: negate the selectivity we want to maximize.
        # x has shape (n_particles, 2*n_sites); evaluate per particle.
        out = np.zeros(x.shape[0])
        for p in range(x.shape[0]):
            sel = params_to_selectivity_rasp(
                x[p, :],
                n_sites=n_sites,
                n_active_sites=n_active_sites,
                true_population=true_population,
                experiment=experiment,
                musc_selective=target_cluster,
                batch_size=1,
            )
            out[p] = -sel[0]
        return out

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2 * n_sites,
        options=options,
        bounds=bounds,
    )
    best_cost, best_pos = optimizer.optimize(objective, iters=n_iters)
    return best_cost, best_pos


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
DEFAULT_MATRIX_COLORS = ["#725defff", "#dd217dff", "#ff5f00ff", "#ffb00dff"]


def plot_matrix(rc, size, title, sel_data=None, sel=False, colors=None):
    """Plot a recruitment/selectivity matrix as a colored lower triangle.

    Faithful port of the notebook helper: each cell (j, i) is shaded with the
    cluster color at intensity ``rc[j, i]`` and annotated with its value.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    rc = np.array(rc)
    color_list = colors if colors is not None else DEFAULT_MATRIX_COLORS
    mask = np.triu(np.ones_like(rc, dtype=bool))

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("white")
    sns.heatmap(rc, mask=mask, cmap=["white"], cbar=False, annot=False,
                linewidths=2, linecolor="white")

    for i in range(size):
        for j in range(size):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_list[i],
                                       alpha=rc[j, i], ec="white", lw=2))

    for i in range(size):
        for j in range(size):
            plt.text(j + 0.5, i + 0.5, f"{rc[j, i]:.2f}",
                     ha="center", va="center", color="black")
        if sel and sel_data is not None:
            ax.text(sel_data.shape[0] + 0.1, i + 0.5, f"{sel_data[i]:.2f}",
                    ha="left", va="center", fontsize=12, color=color_list[i])

    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_xticklabels(np.arange(1, size + 1))
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_yticklabels(np.arange(1, size + 1))
    ax.set_title(title, size=15)
    return fig


def plot_color_section(experiment, fiber_in_fascicle, ax, norm_factor=None,
                       marker_color="black", fill_color="#f3f5f7", alpha=0.8,
                       fascicle_colors=None):
    """Plot fibers colored per fascicle, faded from white to a fascicle color.

    Faithful port of the notebook/utils_sel helper. Uses the polygonal
    epineurium if the topography exposes one, otherwise a circular section from
    ``nerve_radius``. Returns ``(legend_elements, has_epineurium)``.
    """
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize, to_rgb

    fiber_in_fascicle = np.asarray(fiber_in_fascicle)
    n_fibers = len(fiber_in_fascicle)
    n_fascicles = int(np.max(fiber_in_fascicle)) + 1

    if fascicle_colors is None:
        fascicle_colors = ["#fdbf6f", "#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c"]

    has_epineurium = hasattr(experiment.nerve_topography, "epineurium")
    rgb_targets = [to_rgb(c) for c in fascicle_colors]
    white = np.array([1.0, 1.0, 1.0])

    f_colors = []
    for i in range(n_fibers):
        target_rgb = rgb_targets[fiber_in_fascicle[i]]
        if norm_factor is not None:
            mask = fiber_in_fascicle == fiber_in_fascicle[i]
            vals = norm_factor[mask]
            t = Normalize(vmin=np.min(vals), vmax=np.max(vals))(norm_factor[i])
        else:
            t = 1.0
        f_colors.append(white * (1 - t) + np.array(target_rgb) * t)

    ax.set_aspect(1)
    if has_epineurium:
        ax.add_patch(patches.Polygon(experiment.nerve_topography.epineurium,
                                     closed=True, facecolor=fill_color, edgecolor="none"))
    else:
        radius = experiment.nerve_topography.nerve_radius
        ax.add_patch(patches.Circle((0, 0), radius, facecolor=fill_color, edgecolor="none"))

    ax.scatter(experiment.fiber_population.locs[:, 0],
               experiment.fiber_population.locs[:, 1], color=f_colors, alpha=alpha)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.tick_params(labelsize=15)
    ax.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"Fascicle {i+1}",
               markerfacecolor=fascicle_colors[i], markersize=15)
        for i in range(n_fascicles)
    ]
    legend_elements.append(
        Line2D([0], [0], marker="^", color="w", markerfacecolor=marker_color,
               label="Site", markersize=15)
    )
    return legend_elements, has_epineurium


def off_diagonal_frobenius_norm(A):
    """Frobenius norm of the off-diagonal part of A, and its ratio to the full norm.

    Used as a scalar summary of how much recruitment "leaks" off the diagonal
    (lower ratio = more selective).
    """
    A = np.asarray(A, dtype=float)
    full = np.linalg.norm(A)
    mask = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    off = np.linalg.norm(A[mask])
    ratio = off / full if full != 0 else 0.0
    return off, ratio


__all__ = [
    "selectivity_opt",
    "selectivity_eval",
    "params_to_selectivity_rasp",
    "run_pso_selectivity",
    "plot_matrix",
    "plot_color_section",
    "off_diagonal_frobenius_norm",
    "DEFAULT_MATRIX_COLORS",
]
