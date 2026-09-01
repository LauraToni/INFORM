"""
Implant geometry utilities.

This module defines the electrode implant geometry used by INFORM nerve models.
All spatial coordinates are expressed in millimeters.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Implant:
    """Electrode implant with one or more stimulation sites.

    Parameters
    ----------
    n_electrodes : int
        Number of implanted electrodes.
    elec_params : dict
        Dictionary describing the electrode layout. It must contain:

        - ``"n_sites"``: total number of stimulation sites;
        - ``"site_locs"``: array of site coordinates, with shape ``(n_sites, 2)``.

    Attributes
    ----------
    n_electrodes : int
        Number of implanted electrodes.
    elec_params : dict
        Original dictionary of electrode parameters.
    n_sites : int
        Total number of stimulation sites.
    site_locs : ndarray
        Coordinates of stimulation sites in the transverse nerve section,
        with shape ``(n_sites, 2)``.
    """

    def __init__(self, n_electrodes: int, elec_params: dict[str, Any]) -> None:
        if n_electrodes <= 0:
            raise ValueError("n_electrodes must be a positive integer.")

        required_keys = {"n_sites", "site_locs"}
        missing_keys = required_keys.difference(elec_params)
        if missing_keys:
            raise ValueError(f"elec_params is missing required keys: {sorted(missing_keys)}")

        site_locs = np.asarray(elec_params["site_locs"], dtype=float)
        n_sites = int(elec_params["n_sites"])

        if site_locs.ndim != 2 or site_locs.shape[1] != 2:
            raise ValueError("site_locs must have shape (n_sites, 2).")

        if site_locs.shape[0] != n_sites:
            raise ValueError(
                "Mismatch between n_sites and site_locs: "
                f"n_sites={n_sites}, site_locs has {site_locs.shape[0]} rows."
            )

        self._n_electrodes = int(n_electrodes)
        self._elec_params = dict(elec_params)
        self._elec_params["site_locs"] = site_locs
        self._elec_params["n_sites"] = n_sites

        self._n_sites = n_sites
        self._site_locs = site_locs

    def __repr__(self) -> str:
        return (
            f"Implant(n_electrodes={self.n_electrodes}, "
            f"n_sites={self.n_sites})"
        )

    def plot(self, ax, *, color: str = "red", marker: str = "^", **scatter_kwargs) -> None:
        """Plot stimulation sites on a Matplotlib axis."""
        ax.scatter(
            self.site_locs[:, 0],
            self.site_locs[:, 1],
            color=color,
            marker=marker,
            **scatter_kwargs,
        )

    def plot_site_labels(
        self,
        ax,
        *,
        marker_size: float = 150,
        marker_color: str = "yellow",
        text_color: str = "black",
        fontsize: float = 8,
    ) -> None:
        """Plot stimulation sites with numerical labels.

        Labels are one-based to match figure annotations and manuscript notation.
        """
        ax.scatter(
            self.site_locs[:, 0],
            self.site_locs[:, 1],
            s=marker_size,
            color=marker_color,
            marker="o",
        )

        for site_idx, site_loc in enumerate(self.site_locs, start=1):
            x_coord, y_coord = site_loc[:2]
            ax.text(
                x_coord,
                y_coord,
                str(site_idx),
                fontsize=fontsize,
                ha="center",
                va="center",
                color=text_color,
            )

    def plot_text(self, ax) -> None:
        """Backward-compatible alias for :meth:`plot_site_labels`."""
        self.plot_site_labels(ax)

    @property
    def n_electrodes(self) -> int:
        return self._n_electrodes

    @property
    def elec_params(self) -> dict[str, Any]:
        return self._elec_params

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def site_locs(self) -> np.ndarray:
        return self._site_locs
