"""
Visualization utilities for INFORM localization.

These functions are plotting helpers only. They do not modify the localization
algorithm or the underlying experiment objects.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def plot_recruitment_extended(
    true_recruitment_curves=None,
    n_clusters=None,
    amplitudes=None,
    n_sites=None,
    colorList=None,
    title=None,
    *,
    recruitment_values=None,
    colors=None,
):
    """Plot recruitment curves for each stimulation site.

    Accepts both the canonical argument names (``true_recruitment_curves``,
    ``colorList``) and the aliases used by the run scripts
    (``recruitment_values``, ``colors``), so that all callers work unchanged.
    """
    # Normalize argument aliases.
    if true_recruitment_curves is None and recruitment_values is not None:
        true_recruitment_curves = recruitment_values
    if colorList is None and colors is not None:
        colorList = colors
    if true_recruitment_curves is None:
        raise ValueError(
            "Provide recruitment curves via 'true_recruitment_curves' "
            "(or the alias 'recruitment_values')."
        )

    if colorList is None:
        colorList = DEFAULT_CLUSTER_COLORS

    xsub = np.ceil(np.sqrt(n_sites)).astype(int)
    ysub = np.ceil(n_sites / xsub).astype(int)

    fig, ax = plt.subplots(xsub, ysub)
    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=1.55,
        top=0.7,
        wspace=0.4,
        hspace=0.5,
    )

    for i in range(xsub):
        for j in range(ysub):
            if i * ysub + j >= n_sites:
                fig.delaxes(ax[i, j])
                break

            for k in range(n_clusters):
                if n_sites == 1:
                    ax.plot(
                        amplitudes,
                        true_recruitment_curves[:, k, :].T,
                        color=colorList[k],
                    )
                    ax.set_xticks([0, amplitudes[-1]])
                    ax.set_title(f"#{i * ysub + j + 1}")
                    ax.set_ylim(0, 1)
                else:
                    ax[i, j].plot(
                        amplitudes,
                        true_recruitment_curves[i * ysub + j, k, :].T,
                        color=colorList[k],
                    )
                    ax[i, j].set_xticks([0, amplitudes[-1]])
                    ax[i, j].set_title(f"{i * ysub + j + 1}")
                    ax[i, j].set_ylim(0, 1)
                    ax[i, j].spines[["top", "right"]].set_visible(False)

    if title is not None:
        fig.suptitle(title, y=0.8)

    return fig, ax


def plot_recruitment_superposed(
    true_recruitment_curves,
    pred_recruitment_curves=None,
    inferred_recruitment_curves=None,
    amplitudes=None,
    n_sites=None,
    n_clusters=None,
    colorList=None,
    title=None,
    *,
    colors=None,
):
    """Plot true and inferred recruitment curves on the same axes.

    ``pred_recruitment_curves`` is kept for backward compatibility.
    Prefer ``inferred_recruitment_curves`` in new code. The ``colors`` alias is
    accepted for ``colorList`` so that the run scripts work unchanged.
    """
    if colorList is None and colors is not None:
        colorList = colors
    if colorList is None:
        colorList = DEFAULT_CLUSTER_COLORS

    if inferred_recruitment_curves is None:
        inferred_recruitment_curves = pred_recruitment_curves

    if inferred_recruitment_curves is None:
        raise ValueError(
            "Provide either inferred_recruitment_curves or pred_recruitment_curves."
        )

    xsub = np.ceil(np.sqrt(n_sites)).astype(int)
    ysub = np.ceil(n_sites / xsub).astype(int)

    fig, ax = plt.subplots(xsub, ysub, layout="constrained")
    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=1.55,
        top=0.7,
        wspace=0.4,
        hspace=0.7,
    )

    for i in range(xsub):
        for j in range(ysub):
            if i * ysub + j >= n_sites:
                fig.delaxes(ax[i, j])
                break

            for k in range(n_clusters):
                if n_sites == 1:
                    ax.plot(
                        amplitudes,
                        true_recruitment_curves[:, k, :].T,
                        color=colorList[k],
                    )
                    ax.plot(
                        amplitudes,
                        inferred_recruitment_curves[:, k, :].T,
                        "--",
                        color=colorList[k],
                    )
                    ax.set_xticks([0, amplitudes[-1]])
                    ax.set_title(f"{i * ysub + j + 1}")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(axis="y")
                    ax.set_ylim(0, 1)
                else:
                    ax[i, j].plot(
                        amplitudes,
                        true_recruitment_curves[i * ysub + j, k, :].T,
                        color=colorList[k],
                    )
                    ax[i, j].plot(
                        amplitudes,
                        inferred_recruitment_curves[i * ysub + j, k, :].T,
                        "--",
                        color=colorList[k],
                    )
                    ax[i, j].set_xticks([0, amplitudes[-1]])
                    ax[i, j].set_title(f"{i * ysub + j + 1}")
                    ax[i, j].grid(axis="y")
                    ax[i, j].set_ylim(0, 1.1)
                    ax[i, j].spines[["top", "right"]].set_visible(False)

                    if i != xsub - 1:
                        ax[i, j].set_xticks([])

    if title is not None:
        fig.suptitle(title, size=20)

    return fig, ax


def plot_section(
    experiment,
    ax,
    colors=None,
    act=False,
    activation=None,
    pop_clusters=None,
    fiber_population=None,
    topographic=True,
    radius=2,
    marker_color="red",
):
    """Plot nerve section, fibers, implant sites, and optional cluster centers."""
    fill_color = "#f3f5f7"

    if colors:
        all_colors = colors
    else:
        all_colors = DEFAULT_CLUSTER_COLORS

    nerve_section = plt.Circle(
        (0.0, 0.0),
        radius,
        fill=True,
        edgecolor="none",
        facecolor=fill_color,
        label="Nerve and fascicle sections",
    )

    ax.set_aspect(1)
    ax.add_artist(nerve_section)

    if fiber_population is None:
        fiber_population = experiment.fiber_population

    n_fibers = len(fiber_population.locs)
    fiber_color = [
        all_colors[fiber_population.cluster_ids[i].astype(int)]
        for i in range(n_fibers)
    ]

    if act:
        fiber_colors = [
            fiber_color[i_val] if val else "#ced4da"
            for i_val, val in enumerate(activation)
        ]
    else:
        fiber_colors = fiber_color

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

    if topographic and experiment.nerve_topography is not None:
        experiment.nerve_topography.plot(ax=ax)

    if pop_clusters:
        n_clusters = len(pop_clusters[0])
        for i in range(n_clusters):
            ax.scatter(
                pop_clusters[0][i],
                pop_clusters[1][i],
                c=all_colors[i],
                label=f"Cluster {i + 1}",
                s=40,
                edgecolors="black",
            )

    ax.set_ylim(-2.3, 2.3)
    ax.set_xlim(-2.3, 2.3)
    ax.tick_params(labelsize=20)
    ax.yaxis.set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        width=2,
    )
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_aspect("equal")

    return ax


def off_diagonal_frobenius_norm(A):
    """Compute off-diagonal Frobenius norm and ratio."""
    full_frobenius_norm = np.linalg.norm(A)
    off_diagonal_mask = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(off_diagonal_mask, 0)
    off_diagonal_elements = A[off_diagonal_mask]
    off_diag_frobenius_norm = np.linalg.norm(off_diagonal_elements)
    off_diagonal_ratio = off_diag_frobenius_norm / full_frobenius_norm

    return off_diag_frobenius_norm, off_diagonal_ratio


def plot_matrix(rc, size, title):
    """Plot recruitment/selectivity matrix."""
    _, fro_ratio = off_diagonal_frobenius_norm(rc)
    mask = np.triu(np.ones_like(rc, dtype=bool))
    colorList = ["#1f78b4", "#ee7674", "#F6BD60", "#8dd3c7"]

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        rc,
        mask=mask,
        cmap=["white"],
        cbar=False,
        annot=False,
        annot_kws={"size": 13},
        linewidths=2,
        linecolor="white",
    )

    for i in range(size):
        for j in range(size):
            intensity = rc[j, i]
            color = colorList[i]
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    color=color,
                    alpha=intensity,
                    ec="white",
                    lw=2,
                )
            )

    for i in range(size):
        for j in range(size):
            plt.text(
                j + 0.5,
                i + 0.5,
                f"{rc[j, i]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_xticklabels(np.arange(1, size + 1))
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_yticklabels(np.arange(1, size + 1))
    ax.set_title(title + f"{fro_ratio:.3f}", size=15)

    return fig


__all__ = [
    "DEFAULT_CLUSTER_COLORS",
    "plot_recruitment_extended",
    "plot_recruitment_superposed",
    "plot_section",
    "off_diagonal_frobenius_norm",
    "plot_matrix",
]
