"""
Histological nerve-section topography utilities.

This module defines a polygon-based fascicular topography, suitable for
histological nerve sections where the epineurium and fascicles are represented
as segmented polygonal contours.

All geometric quantities are assumed to be expressed in millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.path import Path


DEFAULT_MARGIN = 1.0


@dataclass(frozen=True)
class PolylinearFascicleTopography:
    """Polygon-based nerve topography from histological segmentation.

    Parameters
    ----------
    fascicles:
        List-like collection of fascicle contours. Each fascicle is represented
        as an array with shape ``(n_vertices, 2)``.
    epineurium:
        External nerve contour, represented as an array with shape
        ``(n_vertices, 2)``.

    Notes
    -----
    This class is intended for segmented/histological nerve sections, where
    fascicles are not approximated as circles.
    """

    fascicles: list[np.ndarray]
    epineurium: np.ndarray

    def __post_init__(self) -> None:
        fascicles = [np.asarray(fascicle, dtype=float) for fascicle in self.fascicles]
        epineurium = np.asarray(self.epineurium, dtype=float)

        if epineurium.ndim != 2 or epineurium.shape[1] != 2:
            raise ValueError("epineurium must have shape (n_vertices, 2).")

        for idx, fascicle in enumerate(fascicles):
            if fascicle.ndim != 2 or fascicle.shape[1] != 2:
                raise ValueError(
                    f"fascicle {idx} must have shape (n_vertices, 2)."
                )

        object.__setattr__(self, "fascicles", fascicles)
        object.__setattr__(self, "epineurium", epineurium)

    def __repr__(self) -> str:
        return (
            "PolylinearFascicleTopography("
            f"n_fascicles={self.n_fascicles}, "
            f"total_area={self.total_area:.3f})"
        )

    def plot(
        self,
        ax: Axes,
        fascicle_color: str = "black",
        epineurium_edge_color: str = "grey",
        epineurium_face_color: str = "#f3f5f7",
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        """Plot the epineurium and fascicular contours."""

        epineurium_patch = Polygon(
            self.epineurium,
            closed=True,
            edgecolor=epineurium_edge_color,
            facecolor=epineurium_face_color,
            linewidth=2,
        )
        ax.add_patch(epineurium_patch)

        ax.set_xlim(self.epineurium[:, 0].min() - margin, self.epineurium[:, 0].max() + margin)
        ax.set_ylim(self.epineurium[:, 1].min() - margin, self.epineurium[:, 1].max() + margin)

        for fascicle in self.fascicles:
            ax.plot(fascicle[:, 0], fascicle[:, 1], color=fascicle_color)

        ax.set_aspect("equal", adjustable="box")

    def are_inside_fascicles(
        self,
        point_locs: np.ndarray,
        return_cluster_ids: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Check whether points are inside any fascicle polygon.

        Parameters
        ----------
        point_locs:
            Point coordinates with shape ``(n_points, 2)``.
        return_cluster_ids:
            If ``True``, also return the fascicle index containing each point.
            Points outside all fascicles are assigned ``-1``.

        Returns
        -------
        are_inside:
            Boolean array indicating whether each point is inside at least one
            fascicle.
        cluster_ids:
            Returned only if ``return_cluster_ids=True``. The fascicle index
            associated with each point, or ``-1`` for points outside fascicles.
        """

        point_locs = np.asarray(point_locs, dtype=float)
        if point_locs.ndim != 2 or point_locs.shape[1] != 2:
            raise ValueError("point_locs must have shape (n_points, 2).")

        are_inside = np.zeros(point_locs.shape[0], dtype=bool)
        cluster_ids = np.full(point_locs.shape[0], fill_value=-1, dtype=int)

        for idx, fascicle in enumerate(self.fascicles):
            inside_current = Path(fascicle).contains_points(point_locs)
            newly_assigned = np.logical_and(inside_current, ~are_inside)
            cluster_ids[newly_assigned] = idx
            are_inside = np.logical_or(are_inside, inside_current)

        if return_cluster_ids:
            return are_inside, cluster_ids
        return are_inside

    def sample_points_in_fascicles(
        self,
        n_points: int,
        oversampling_factor: int = 5,
        max_iterations: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly sample points inside fascicle polygons.

        Points are allocated to fascicles proportionally to their polygonal
        areas. The returned ``cluster_ids`` correspond to the fascicle index,
        preserving compatibility with the rest of the INFORM codebase.

        Parameters
        ----------
        n_points:
            Total number of points to sample.
        oversampling_factor:
            Number of candidate points sampled per missing accepted point at
            each rejection-sampling iteration.
        max_iterations:
            Maximum number of rejection-sampling iterations per fascicle.

        Returns
        -------
        point_locs:
            Sampled coordinates with shape ``(n_points, 2)``.
        cluster_ids:
            Fascicle index associated with each sampled point.
        """

        if n_points <= 0:
            raise ValueError("n_points must be positive.")

        n_points_per_fascicle = self._allocate_points_to_fascicles(n_points)
        point_locs = np.zeros((n_points, 2), dtype=float)
        cluster_ids = np.zeros(n_points, dtype=int)

        start = 0
        for fascicle_idx, n_current in enumerate(n_points_per_fascicle):
            if n_current == 0:
                continue

            sampled = self._sample_points_in_polygon(
                polygon=self.fascicles[fascicle_idx],
                n_points=n_current,
                oversampling_factor=oversampling_factor,
                max_iterations=max_iterations,
            )

            stop = start + n_current
            point_locs[start:stop, :] = sampled
            cluster_ids[start:stop] = fascicle_idx
            start = stop

        return point_locs, cluster_ids

    def _allocate_points_to_fascicles(self, n_points: int) -> np.ndarray:
        """Allocate points to fascicles proportionally to fascicle area."""

        if self.total_area <= 0:
            raise ValueError("Total fascicular area must be positive.")

        n_points_per_fascicle = np.round(
            n_points * self.fascicles_areas / self.total_area
        ).astype(int)
        n_points_per_fascicle[-1] = int(n_points - n_points_per_fascicle[:-1].sum())
        return n_points_per_fascicle

    @staticmethod
    def _sample_points_in_polygon(
        polygon: np.ndarray,
        n_points: int,
        oversampling_factor: int,
        max_iterations: int,
    ) -> np.ndarray:
        """Sample points uniformly inside a polygon via rejection sampling."""

        path = Path(polygon)
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        accepted: list[np.ndarray] = []
        n_accepted = 0
        n_iterations = 0

        while n_accepted < n_points and n_iterations < max_iterations:
            n_missing = n_points - n_accepted
            n_candidates = max(oversampling_factor * n_missing, 100)

            candidates = np.column_stack(
                (
                    np.random.uniform(x_min, x_max, size=n_candidates),
                    np.random.uniform(y_min, y_max, size=n_candidates),
                )
            )
            inside = path.contains_points(candidates)
            new_points = candidates[inside]

            if new_points.size > 0:
                accepted.append(new_points)
                n_accepted += new_points.shape[0]

            n_iterations += 1

        if n_accepted < n_points:
            raise RuntimeError(
                "Could not sample enough points inside polygon. "
                "Try increasing oversampling_factor or max_iterations."
            )

        return np.vstack(accepted)[:n_points, :]
    
    @property
    def fascicles(self):
        return self._fascicles

    @property
    def epineurium(self):
        return self._epineurium

    @property
    def n_fascicles(self):
        if self.fascicles is not None:
            return len(self.fascicles)
        return 0

    @property
    def fascicles_areas(self) -> np.ndarray:
        """Area of each fascicle polygon."""

        return np.array([polygon_area(fascicle) for fascicle in self.fascicles])

    @property
    def total_area(self) -> float:
        """Total area of all fascicle polygons."""

        return float(self.fascicles_areas.sum())


def polygon_area(vertices: np.ndarray) -> float:
    """Compute polygon area using the shoelace formula."""

    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (n_vertices, 2).")

    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, shift=-1)) - np.dot(y, np.roll(x, shift=-1)))
