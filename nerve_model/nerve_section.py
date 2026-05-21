"""Nerve section geometry utilities.

This module defines circular nerve cross-sections with circular fascicles.
All geometric quantities are expressed in millimeters.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


DEFAULT_OUTSIDE_GRID_STEP = 0.01
DEFAULT_N_TRIES = 100


class CircularFascicleTopography:
    """Topography of a nerve section with circular epineurium and fascicles.

    Parameters
    ----------
    fascicles : np.ndarray
        Circular fascicle parameters with shape ``(n_fascicles, 3)``.
        Columns correspond to ``x_center``, ``y_center``, and ``radius``.
    nerve_radius : float
        Radius of the circular nerve boundary.
    """

    def __init__(self, fascicles: np.ndarray, nerve_radius: float) -> None:
        fascicles = np.asarray(fascicles, dtype=float)
        if fascicles.ndim != 2 or fascicles.shape[1] != 3:
            raise ValueError("fascicles must have shape (n_fascicles, 3).")
        if nerve_radius <= 0:
            raise ValueError("nerve_radius must be positive.")

        self._fascicles = fascicles
        self._nerve_radius = float(nerve_radius)

    @classmethod
    def from_fascicle_stats(
        cls,
        nerve_radius: float,
        min_dist_btw_fasc: float,
        grid_step: float,
        fasc_radii: Optional[np.ndarray] = None,
        n_fascicles: Optional[int] = None,
        fasc_radius_lims: Optional[Sequence[float]] = None,
        n_tries: int = DEFAULT_N_TRIES,
    ) -> "CircularFascicleTopography":
        """Pack circular fascicles inside a circular nerve section.

        Fascicles are placed on a regular grid of admissible center locations,
        enforcing a minimum distance between fascicles and from the outer nerve
        boundary.

        Parameters
        ----------
        nerve_radius : float
            Radius of the nerve section.
        min_dist_btw_fasc : float
            Minimum distance allowed between fascicle boundaries.
        grid_step : float
            Step of the grid of admissible fascicle center locations.
        fasc_radii : np.ndarray, optional
            Fascicle radii with shape ``(n_fascicles,)``. If omitted, radii are
            sampled uniformly from ``fasc_radius_lims``.
        n_fascicles : int, optional
            Number of fascicles to generate. Required when ``fasc_radii`` is not
            provided.
        fasc_radius_lims : sequence of float, optional
            Minimum and maximum fascicle radius. Required when ``fasc_radii`` is
            not provided.
        n_tries : int
            Maximum number of packing attempts.

        Returns
        -------
        CircularFascicleTopography
            Packed circular fascicle topography.
        """
        if nerve_radius <= 0:
            raise ValueError("nerve_radius must be positive.")
        if grid_step <= 0:
            raise ValueError("grid_step must be positive.")
        if min_dist_btw_fasc < 0:
            raise ValueError("min_dist_btw_fasc must be non-negative.")
        if n_tries <= 0:
            raise ValueError("n_tries must be positive.")

        if fasc_radii is None:
            if n_fascicles is None or fasc_radius_lims is None:
                raise ValueError(
                    "n_fascicles and fasc_radius_lims must be provided when "
                    "fasc_radii is None."
                )
            if len(fasc_radius_lims) != 2:
                raise ValueError("fasc_radius_lims must contain two values.")
            radius_min, radius_max = fasc_radius_lims
            if radius_min <= 0 or radius_max <= radius_min:
                raise ValueError("fasc_radius_lims must be positive and increasing.")
            fasc_radii = (
                np.random.rand(n_fascicles) * (radius_max - radius_min) + radius_min
            )
        else:
            fasc_radii = np.asarray(fasc_radii, dtype=float)
            if fasc_radii.ndim != 1:
                raise ValueError("fasc_radii must be a one-dimensional array.")
            if np.any(fasc_radii <= 0):
                raise ValueError("All fascicle radii must be positive.")
            n_fascicles = fasc_radii.shape[0]

        grid = np.arange(start=-nerve_radius, stop=nerve_radius, step=grid_step)
        x_grid, y_grid = np.meshgrid(grid, grid)
        x = x_grid.ravel()
        y = y_grid.ravel()

        for _ in range(n_tries):
            fascicles = np.zeros((n_fascicles, 3), dtype=float)
            fascicles[:, 2] = np.sort(fasc_radii)[::-1]

            success = True
            for i in range(n_fascicles):
                is_admissible = (
                    x**2 + y**2
                    < (nerve_radius - fascicles[i, 2] - min_dist_btw_fasc) ** 2
                )

                for j in range(i):
                    dist_ij = (x - fascicles[j, 0]) ** 2 + (y - fascicles[j, 1]) ** 2
                    min_dist_ij = (
                        fascicles[j, 2] + fascicles[i, 2] + min_dist_btw_fasc
                    ) ** 2
                    is_admissible = np.logical_and(is_admissible, dist_ij > min_dist_ij)

                admissible_idx = np.nonzero(is_admissible)[0]
                if admissible_idx.size == 0:
                    success = False
                    break

                selected_idx = np.random.choice(admissible_idx)
                fascicles[i, 0] = x[selected_idx]
                fascicles[i, 1] = y[selected_idx]

            if success:
                return cls(fascicles=fascicles, nerve_radius=nerve_radius)

        raise RuntimeError(
            f"Could not pack fascicles with the desired features in {n_tries} attempts."
        )

    def __repr__(self) -> str:
        return (
            "CircularFascicleTopography("
            f"n_fascicles={self.n_fascicles}, "
            f"nerve_radius={self.nerve_radius})"
        )

    def plot(self, ax) -> None:
        """Plot nerve boundary and fascicle boundaries on a Matplotlib axis."""
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            self.nerve_radius * np.cos(theta),
            self.nerve_radius * np.sin(theta),
            color="black",
        )
        for i in range(self.n_fascicles):
            x_fascicle = self.fascicles[i, 2] * np.cos(theta) + self.fascicles[i, 0]
            y_fascicle = self.fascicles[i, 2] * np.sin(theta) + self.fascicles[i, 1]
            ax.plot(x_fascicle, y_fascicle, color="black")

    def are_inside_fascicles(
        self,
        point_locs: np.ndarray,
        buffer_width: float = 0,
    ) -> np.ndarray:
        """Check whether points are inside any fascicle.

        Parameters
        ----------
        point_locs : np.ndarray
            Point locations in the nerve transverse section, with shape
            ``(n_points, 2)``.
        buffer_width : float
            Additional distance added to fascicle radii when testing inclusion.

        Returns
        -------
        np.ndarray
            Boolean array with shape ``(n_points,)``.
        """
        point_locs = np.asarray(point_locs, dtype=float)
        if point_locs.ndim != 2 or point_locs.shape[1] != 2:
            raise ValueError("point_locs must have shape (n_points, 2).")

        are_inside = np.zeros(point_locs.shape[0], dtype=bool)
        for i in range(self.n_fascicles):
            distances_sq = (
                (point_locs[:, 0] - self.fascicles[i, 0]) ** 2
                + (point_locs[:, 1] - self.fascicles[i, 1]) ** 2
            )
            are_inside_current = distances_sq <= (self.fascicles[i, 2] + buffer_width) ** 2
            are_inside = np.logical_or(are_inside, are_inside_current)
        return are_inside

    def sample_points_in_fascicles(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points uniformly inside fascicles with fixed density.

        Points are allocated to fascicles proportionally to fascicle area. The
        second returned array is kept as ``cluster_ids`` for compatibility with
        the existing codebase.

        Parameters
        ----------
        n_points : int
            Number of points to sample.

        Returns
        -------
        point_locs : np.ndarray
            Locations of sampled points in the nerve transverse section, with
            shape ``(n_points, 2)``.
        cluster_ids : np.ndarray
            Identifier associated with each sampled point, with shape
            ``(n_points,)``.
        """
        if n_points <= 0:
            raise ValueError("n_points must be positive.")

        n_points_per_fascicle = np.round(
            n_points * self.fascicles_areas / np.sum(self.fascicles_areas)
        ).astype(int)
        n_points_per_fascicle[-1] = int(n_points - np.sum(n_points_per_fascicle[:-1]))

        n_points_per_fascicle_cum = np.zeros(self.n_fascicles + 1, dtype=int)
        n_points_per_fascicle_cum[1:] = np.cumsum(n_points_per_fascicle)

        point_locs = np.zeros((n_points, 2), dtype=float)
        cluster_ids = np.zeros(n_points, dtype=int)

        for i in range(self.n_fascicles):
            n_current = n_points_per_fascicle[i]
            if n_current == 0:
                continue

            rnd_rho = self.fascicles[i, 2] * np.sqrt(np.random.rand(n_current))
            rnd_theta = 2 * np.pi * np.random.rand(n_current)
            ind_start = n_points_per_fascicle_cum[i]
            ind_stop = n_points_per_fascicle_cum[i + 1]

            point_locs[ind_start:ind_stop, :] = np.column_stack(
                (
                    rnd_rho * np.cos(rnd_theta) + self.fascicles[i, 0],
                    rnd_rho * np.sin(rnd_theta) + self.fascicles[i, 1],
                )
            )
            cluster_ids[ind_start:ind_stop] = i

        return point_locs, cluster_ids

    def sample_points_outside_fascicles(
        self,
        n_points: int,
        buffer_width: float = 0,
        grid_step: float = DEFAULT_OUTSIDE_GRID_STEP,
    ) -> np.ndarray:
        """Sample points inside the nerve boundary and outside fascicles.

        Parameters
        ----------
        n_points : int
            Number of points to sample.
        buffer_width : float
            Minimum distance from the nerve boundary and fascicle boundaries.
        grid_step : float
            Step of the grid used to define candidate point locations.

        Returns
        -------
        np.ndarray
            Point locations in the nerve section, with shape ``(n_points, 2)``.
        """
        if n_points <= 0:
            raise ValueError("n_points must be positive.")
        if grid_step <= 0:
            raise ValueError("grid_step must be positive.")
        if buffer_width < 0:
            raise ValueError("buffer_width must be non-negative.")

        grid = np.arange(start=-self.nerve_radius, stop=self.nerve_radius, step=grid_step)
        x_grid, y_grid = np.meshgrid(grid, grid)
        x = x_grid.ravel()
        y = y_grid.ravel()

        is_admissible = x**2 + y**2 < (self.nerve_radius - buffer_width) ** 2

        for i in range(self.n_fascicles):
            dist_ip = (x - self.fascicles[i, 0]) ** 2 + (y - self.fascicles[i, 1]) ** 2
            min_dist_ip = (self.fascicles[i, 2] + buffer_width) ** 2
            is_admissible = np.logical_and(is_admissible, dist_ip > min_dist_ip)

        admissible_idx = np.nonzero(is_admissible)[0]
        if admissible_idx.size < n_points:
            raise ValueError(
                "Not enough admissible grid points to sample the requested "
                f"number of points ({admissible_idx.size} available, {n_points} requested)."
            )

        selected_idx = np.random.choice(admissible_idx, n_points, replace=False)
        return np.column_stack((x[selected_idx], y[selected_idx]))

    @property
    def fascicles(self) -> np.ndarray:
        return self._fascicles

    @property
    def nerve_radius(self) -> float:
        return self._nerve_radius

    @property
    def n_fascicles(self) -> int:
        if self.fascicles is None:
            return 0
        return self.fascicles.shape[0]

    @property
    def fascicles_areas(self) -> np.ndarray:
        return np.pi * self.fascicles[:, 2] ** 2

    @property
    def total_area(self) -> float:
        return float(np.sum(self.fascicles_areas))
