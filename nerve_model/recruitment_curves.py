"""
Recruitment curve utilities.

This module defines a container for recruitment curves and helper methods to
compute recruitment thresholds and compare recruitment profiles.

Recruitment values are assumed to be expressed as fractions in the range [0, 1].
Stimulation amplitudes are assumed to be expressed in the same units throughout
the analysis, typically mA or microA depending on the calling code.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


class RecruitmentCurves:
    """Container for recruitment curves across stimulation sites and fiber groups.

    Parameters
    ----------
    recruitment_values : ndarray or list
        Recruitment curves. If an array is provided, the expected shape is
        ``(n_sites, n_groups, n_amplitudes)``. If a list is provided, each item
        should contain the recruitment curves for one stimulation site with
        shape ``(n_groups, n_amplitudes_for_site)``.
    amplitudes : ndarray or list of ndarray
        Stimulation amplitudes associated with the recruitment curves. For array
        recruitment values, this is expected to be a one-dimensional array. For
        list-based recruitment values, this can be a list with one amplitude
        vector per stimulation site.
    """

    def __init__(self, recruitment_values, amplitudes) -> None:
        self.recruitment_values = recruitment_values

        if isinstance(recruitment_values, list):
            self.amplitudes = [np.asarray(curr_amplitudes).flatten() for curr_amplitudes in amplitudes]
            self.amplitude_type = "list"
        else:
            self.recruitment_values = np.asarray(recruitment_values, dtype=float)
            self.amplitudes = np.asarray(amplitudes, dtype=float).flatten()
            self.amplitude_type = "array"

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate recruitment curve and amplitude dimensions."""
        if self.type == "array":
            if self.recruitment_values.ndim != 3:
                raise ValueError(
                    "Array recruitment_values must have shape "
                    "(n_sites, n_groups, n_amplitudes)."
                )
            if self.recruitment_values.shape[2] != self.amplitudes.size:
                raise ValueError(
                    "Mismatch between recruitment_values and amplitudes: "
                    f"{self.recruitment_values.shape[2]} recruitment samples, "
                    f"{self.amplitudes.size} amplitudes."
                )

        elif self.type == "list":
            if len(self.recruitment_values) != len(self.amplitudes):
                raise ValueError(
                    "For list-based recruitment curves, recruitment_values and "
                    "amplitudes must contain one entry per stimulation site."
                )
            for site_idx, (site_values, site_amplitudes) in enumerate(
                zip(self.recruitment_values, self.amplitudes)
            ):
                site_values = np.asarray(site_values)
                if site_values.ndim != 2:
                    raise ValueError(
                        "Each list entry in recruitment_values must have shape "
                        f"(n_groups, n_amplitudes). Error at site {site_idx}."
                    )
                if site_values.shape[1] != site_amplitudes.size:
                    raise ValueError(
                        f"Mismatch at site {site_idx}: "
                        f"{site_values.shape[1]} recruitment samples, "
                        f"{site_amplitudes.size} amplitudes."
                    )

    def compute_ix(self, recruitment_level: float) -> np.ndarray:
        """Compute the stimulation amplitude corresponding to a recruitment level.

        Parameters
        ----------
        recruitment_level : float
            Recruitment fraction at which the current is interpolated.

        Returns
        -------
        ndarray
            Interpolated amplitudes with shape ``(n_sites, n_groups)``.
        """
        if not 0 <= recruitment_level <= 1:
            raise ValueError("recruitment_level must be in the range [0, 1].")

        ix = np.zeros((self.n_sites, self.n_groups))

        if self.amplitude_type == "list":
            for site_idx in range(self.n_sites):
                for group_idx in range(self.n_groups):
                    ix[site_idx, group_idx] = np.interp(
                        recruitment_level,
                        self.recruitment_values[site_idx][group_idx, :],
                        self.amplitudes[site_idx],
                    )

        elif self.amplitude_type == "array":
            for site_idx in range(self.n_sites):
                for group_idx in range(self.n_groups):
                    ix[site_idx, group_idx] = np.interp(
                        recruitment_level,
                        self.recruitment_values[site_idx, group_idx, :],
                        self.amplitudes,
                    )

        return ix

    def compute_i50(self) -> np.ndarray:
        """Compute the current at 50% recruitment."""
        return self.compute_ix(recruitment_level=0.5)

    def compute_activation_threshold(self) -> np.ndarray:
        """Compute the current at 10% recruitment."""
        return self.compute_ix(recruitment_level=0.1)

    def compute_saturation_threshold(self) -> np.ndarray:
        """Compute the current at 90% recruitment."""
        return self.compute_ix(recruitment_level=0.9)

    def average_area_btw_recruitment_curves(
        self,
        group_id: int,
        recruitment_curves_2: "RecruitmentCurves",
    ) -> np.ndarray:
        """Compute average area between one group and candidate recruitment curves.

        This is used to compare the recruitment curves of one reference group
        against all groups in another recruitment-curve object.

        Parameters
        ----------
        group_id : int
            Index of the reference group in ``self``.
        recruitment_curves_2 : RecruitmentCurves
            Candidate recruitment curves to compare against.

        Returns
        -------
        ndarray
            Average area between curves for each candidate group.
        """
        if self.type != "array" or recruitment_curves_2.type != "array":
            raise NotImplementedError(
                "average_area_btw_recruitment_curves currently supports only "
                "array-based recruitment curves."
            )

        if self.n_sites != recruitment_curves_2.n_sites:
            raise ValueError("Both RecruitmentCurves objects must have the same number of sites.")

        if self.recruitment_values.shape[2] != recruitment_curves_2.recruitment_values.shape[2]:
            raise ValueError("Both RecruitmentCurves objects must have the same number of amplitudes.")

        n_candidate_groups = recruitment_curves_2.n_groups
        n_stims = self.recruitment_values.shape[2]
        areas = np.zeros((self.n_sites, n_candidate_groups))

        for candidate_group_idx in range(n_candidate_groups):
            abs_diff = np.abs(
                self.recruitment_values[:, group_id, :]
                - recruitment_curves_2.recruitment_values[:, candidate_group_idx, :]
            )
            for site_idx in range(self.n_sites):
                areas[site_idx, candidate_group_idx] = np.trapz(
                    y=abs_diff[site_idx, :],
                    x=np.arange(n_stims),
                )

        return np.mean(areas, axis=0)

    def plot(self, n_rows: int = 1):
        """Plot recruitment curves arranged by stimulation site.

        Parameters
        ----------
        n_rows : int, default=1
            Number of subplot rows.

        Returns
        -------
        tuple
            Matplotlib ``(fig, axes)`` objects.
        """
        if n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")

        n_axes = self.n_sites
        n_cols = int(np.ceil(n_axes / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, constrained_layout=True, squeeze=False)
        axes_flat = axes.flatten()

        for site_idx in range(n_axes):
            ax = axes_flat[site_idx]

            if self.type == "list":
                amplitudes = self.amplitudes[site_idx]
                values = self.recruitment_values[site_idx]
            else:
                amplitudes = self.amplitudes
                values = self.recruitment_values[site_idx, :, :]

            for group_idx in range(self.n_groups):
                ax.plot(amplitudes, values[group_idx, :])

            ax.set_title(f"Site {site_idx + 1}")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Recruitment")

        for ax in axes_flat[n_axes:]:
            ax.axis("off")

        return fig, axes

    @property
    def type(self) -> str:
        """Return whether recruitment values are stored as a list or array."""
        if isinstance(self.recruitment_values, list):
            return "list"
        return "array"

    @property
    def n_sites(self) -> int:
        if self.type == "list":
            return len(self.recruitment_values)
        return self.recruitment_values.shape[0]

    @property
    def n_groups(self) -> int:
        if self.type == "list":
            return np.asarray(self.recruitment_values[0]).shape[0]
        return self.recruitment_values.shape[1]
