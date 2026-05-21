"""
Experiment-level utilities for INFORM.

The :class:`Experiment` class connects the nerve geometry, fiber population,
implant layout, lead-field matrix, and activation predictor. It provides methods
to export FEM inputs, load FEM results, generate stimulation protocols, compute
extracellular potentials, predict fiber activations, and derive recruitment or
selectivity patterns.

Unless otherwise stated, spatial coordinates are expressed in millimeters and
currents are expressed in the units used by the calling analysis code.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import scipy.linalg
from scipy.io import savemat

from fiber_population import MotorFiberPopulation
from nerve_section import CircularFascicleTopography
from recruitment_curves import RecruitmentCurves


ActivationMethod = Literal["from_self", "mirror", "actfun"]


class Experiment:
    """Container for one nerve stimulation experiment.

    Parameters
    ----------
    fiber_population : MotorFiberPopulation
        Fiber population used in the experiment.
    implant : Implant
        Implant/electrode layout.
    nerve_topography : CircularFascicleTopography or compatible object, optional
        Nerve cross-sectional topography.
    lead_field_matrix : ndarray, optional
        Lead-field matrix mapping stimulation site currents to extracellular
        potentials at FEM nodes. Shape is ``(n_fem_nodes, n_sites)``.
    activation_predictor : object, optional
        Trained classifier exposing a ``predict`` method.
    lead_field_matrix_per_fiber : ndarray, optional
        Optional lead-field representation already organized by fiber.
    """

    def __init__(
        self,
        fiber_population: MotorFiberPopulation,
        implant,
        nerve_topography=None,
        lead_field_matrix: np.ndarray | None = None,
        activation_predictor=None,
        lead_field_matrix_per_fiber: np.ndarray | None = None,
    ) -> None:
        self._fiber_population = fiber_population
        self._nerve_topography = nerve_topography
        self._implant = implant
        self._lead_field_matrix = lead_field_matrix
        self._lead_field_matrix_per_fiber = lead_field_matrix_per_fiber
        self._activation_predictor = activation_predictor

    @classmethod
    def from_existing_experiment(
        cls,
        experiment: "Experiment",
        fiber_population: MotorFiberPopulation | None = None,
        has_struct_info: bool = False,
        cluster_locs: np.ndarray | None = None,
        cluster_std: np.ndarray | None = None,
        cluster_num: np.ndarray | None = None,
    ):
        """Create a new experiment from an existing one.

        This is used when a new functional organization is sampled or inferred
        on top of a previously defined experiment. If ``has_struct_info`` is
        false, only the external nerve radius is retained.

        Returns
        -------
        Experiment
            New experiment object.
        ndarray
            Fiber identities selected from the original population.
        """
        if has_struct_info:
            nerve_topography = experiment.nerve_topography
        else:
            nerve_topography = CircularFascicleTopography(
                fascicles=None,
                nerve_radius=experiment.nerve_topography.nerve_radius,
            )

        identities = None
        if fiber_population is None:
            fiber_population, identities = MotorFiberPopulation.from_existing_population(
                fiber_population=experiment.fiber_population,
                cluster_locs=cluster_locs,
                cluster_std=cluster_std,
                cluster_num=cluster_num,
            )

        obj = cls(
            fiber_population=fiber_population,
            nerve_topography=nerve_topography,
            implant=experiment.implant,
            activation_predictor=experiment.activation_predictor,
        )

        return obj, identities

    def __repr__(self) -> str:
        return (
            "Experiment("
            f"n_fibers={self.fiber_population.n_fibers}, "
            f"n_sites={self.implant.n_sites}, "
            f"lead_field_matrix_loaded={self.lead_field_matrix is not None}, "
            f"activation_predictor_loaded={self.activation_predictor is not None})"
        )

    # ---------------------------------------------------------------------
    # FEM export and loading
    # ---------------------------------------------------------------------
    def export_for_fem(self, folder_path: str | Path, tag: str = "0") -> None:
        """Export nerve, fiber-node, and implant information for FEM simulations.

        Files are written as:

        - ``nerve_topography_<tag>.mat``
        - ``node_locs_<tag>.txt``
        - ``site_locs_<tag>.mat``
        - ``experiment_<tag>.pkl``

        Coordinates are converted from millimeters to meters for FEM export.
        """
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        theta = np.linspace(0, 2 * np.pi, 100)
        x_epineurium = np.expand_dims(self.nerve_topography.nerve_radius * np.cos(theta), 1)
        y_epineurium = np.expand_dims(self.nerve_topography.nerve_radius * np.sin(theta), 1)
        epineurium = np.hstack((x_epineurium, y_epineurium)) * 1e-3

        mdict = {"epineurium": epineurium}
        if self.nerve_topography.fascicles is not None:
            mdict["fascicles"] = self.nerve_topography.fascicles * 1e-3

        savemat(folder_path / f"nerve_topography_{tag}.mat", mdict=mdict)

        fem_nodes = self.fiber_population.fem_node_locs * 1e-3
        np.savetxt(folder_path / f"node_locs_{tag}.txt", fem_nodes, delimiter=",")

        site_locs = self.implant.site_locs * 1e-3
        if site_locs.shape[1] == 2:
            site_locs = np.hstack((site_locs, np.zeros((self.implant.n_sites, 1))))

        savemat(folder_path / f"site_locs_{tag}.mat", mdict={"site_locs": site_locs})

        with open(folder_path / f"experiment_{tag}.pkl", "wb") as file:
            pickle.dump(self, file)

    def load_lead_field_matrix(
        self,
        base_file_path: str | Path | None = None,
        identities: np.ndarray | None = None,
        save_to_hdf5: bool = True,
        hdf5_file_path: str | Path | None = None,
        full_experiment: "Experiment" | None = None,
        lead_field_matrix: np.ndarray | None = None,
        comsol_scale_factor: float = 1e3,
    ) -> np.ndarray:
        """Load or subset the lead-field matrix.

        Parameters
        ----------
        base_file_path : str or Path, optional
            Prefix of COMSOL text files. Files are expected to be named
            ``<base_file_path>1.txt``, ``<base_file_path>2.txt``, etc.
        identities : ndarray, optional
            Fiber identities in ``full_experiment`` to keep when extracting a
            subset of a larger lead-field matrix.
        save_to_hdf5 : bool, default=True
            Whether to save the loaded COMSOL matrix to HDF5.
        hdf5_file_path : str or Path, optional
            Output HDF5 path used when ``save_to_hdf5`` is true.
        full_experiment : Experiment, optional
            Experiment from which ``identities`` refer.
        lead_field_matrix : ndarray, optional
            Lead-field matrix passed directly as an array.
        comsol_scale_factor : float, default=1e3
            Scaling applied to COMSOL potential values. Keep explicit to avoid
            hidden unit conversions.

        Returns
        -------
        ndarray
            Loaded lead-field matrix.
        """
        if base_file_path is not None:
            base_file_path = str(base_file_path)
            matrix = np.zeros((self.fiber_population.n_fem_nodes, self.implant.n_sites))

            for site_idx in range(self.implant.n_sites):
                file_path = f"{base_file_path}{site_idx + 1}.txt"
                comsol_output = np.loadtxt(fname=file_path, comments="%")
                matrix[:, site_idx] = comsol_output[:, 3] * comsol_scale_factor

            if save_to_hdf5:
                if hdf5_file_path is None:
                    raise ValueError("hdf5_file_path must be provided when save_to_hdf5=True.")
                with h5py.File(hdf5_file_path, "w") as hdf5_file:
                    hdf5_file.create_dataset("lead_field_matrix", data=matrix, chunks=True)

        elif lead_field_matrix is not None:
            data = np.asarray(lead_field_matrix)

            if identities is None:
                matrix = data
            else:
                if full_experiment is None:
                    raise ValueError("full_experiment must be provided when identities is used.")

                identities = identities.astype(int)
                matrix = np.zeros((self.fiber_population.n_fem_nodes, data.shape[1]))

                current_node = 0
                for fiber_idx in range(self.fiber_population.n_fibers):
                    source_fiber_idx = identities[fiber_idx]
                    start_node = int(full_experiment.fiber_population.fem_node_lims[source_fiber_idx, 0])
                    end_node = int(full_experiment.fiber_population.fem_node_lims[source_fiber_idx, 1])
                    n_nodes_current_fiber = end_node - start_node

                    matrix[current_node : current_node + n_nodes_current_fiber, :] = data[start_node:end_node, :]
                    current_node += n_nodes_current_fiber
        else:
            raise ValueError("Either base_file_path or lead_field_matrix must be provided.")

        self._lead_field_matrix = matrix
        return matrix

    def compute_homogeneous_lead_field_matrix(self, conductivity: float | np.ndarray) -> np.ndarray:
        """Compute the lead-field matrix for an infinite homogeneous medium.

        Parameters
        ----------
        conductivity : float or ndarray
            Isotropic conductivity if scalar, or anisotropic conductivity tensor
            diagonal with shape ``(3,)``.

        Returns
        -------
        ndarray
            Lead-field matrix with shape ``(n_fem_nodes, n_sites)``.
        """
        conductivity = np.asarray(conductivity if not np.isscalar(conductivity) else np.ones(3) * conductivity)

        node_locations = self.fiber_population.fem_node_locs
        site_locations = self.implant.site_locs

        if site_locations.shape[1] == 2:
            site_locations = np.hstack((site_locations, np.zeros((self.implant.n_sites, 1))))

        node_locations = np.tile(np.expand_dims(node_locations, axis=1), (1, self.implant.n_sites, 1))
        site_locations = np.tile(np.expand_dims(site_locations, axis=0), (self.fiber_population.n_fem_nodes, 1, 1))

        site_node_dist = np.abs(node_locations - site_locations)

        kappa = np.sqrt(
            site_node_dist[:, :, 0] ** 2 * conductivity[1] * conductivity[2]
            + conductivity[0] * site_node_dist[:, :, 1] ** 2 * conductivity[2]
            + conductivity[0] * conductivity[1] * site_node_dist[:, :, 2] ** 2
        )

        with np.errstate(divide="ignore"):
            matrix = 1 / (4 * np.pi * kappa)

        self._lead_field_matrix = matrix
        return matrix

    def load_activation_predictor(
        self,
        activation_predictor=None,
        activation_predictor_path: str | Path | None = None,
    ):
        """Load a trained fiber activation predictor."""
        if activation_predictor is None:
            if activation_predictor_path is None:
                raise ValueError("activation_predictor_path must be provided if activation_predictor is None.")
            with open(activation_predictor_path, "rb") as file:
                data = pickle.load(file)
            activation_predictor = data["activation_predictor"]

        self._activation_predictor = activation_predictor
        return activation_predictor

    # ---------------------------------------------------------------------
    # Stimulation protocols and extracellular potentials
    # ---------------------------------------------------------------------
    def generate_stimulation_batch(
        self,
        current_limits: np.ndarray,
        n_stims: int | None = None,
        n_stims_per_site: int | None = None,
        n_active_sites: int | list[int] = 1,
        out: Literal["current", "potential", "both"] = "potential",
        n_fibers_per_stim: int | None = None,
    ):
        """Generate random stimulation protocols.

        Parameters
        ----------
        current_limits : ndarray
            Current bounds. Either ``(2,)`` for shared limits across sites or
            ``(n_sites, 2)`` for site-specific limits.
        n_stims : int, optional
            Total number of random stimulation protocols.
        n_stims_per_site : int, optional
            Number of monopolar protocols per site.
        n_active_sites : int or list of int, default=1
            Number of active sites per random protocol.
        out : {"current", "potential", "both"}, default="potential"
            Requested output.
        n_fibers_per_stim : int, optional
            If provided, randomly sample this number of fibers per stimulation
            protocol when computing potentials.

        Returns
        -------
        ndarray or tuple
            Depends on ``out``.
        """
        current_limits = self._normalize_current_limits(current_limits)
        fiber_ids = None

        if n_stims is not None:
            stimulation_protocols = np.zeros((n_stims, self.implant.n_sites))

            if n_fibers_per_stim is not None:
                fiber_ids = np.random.choice(
                    self.fiber_population.n_fibers,
                    size=(n_stims, n_fibers_per_stim),
                )

            for stim_idx in range(n_stims):
                chosen_n_active_sites = self._sample_n_active_sites(n_active_sites)
                chosen_sites = np.random.choice(
                    self.implant.n_sites,
                    size=chosen_n_active_sites,
                    replace=False,
                )

                for site_idx in chosen_sites:
                    min_curr, max_curr = current_limits[site_idx]
                    stimulation_protocols[stim_idx, site_idx] = (
                        np.random.rand() * (max_curr - min_curr) + min_curr
                    )

        elif n_stims_per_site is not None:
            n_stims = int(n_stims_per_site * self.implant.n_sites)

            if n_fibers_per_stim is not None:
                fiber_ids = np.random.choice(
                    self.fiber_population.n_fibers,
                    size=(n_stims, n_fibers_per_stim),
                )

            amplitude_blocks = []
            for site_idx in range(self.implant.n_sites):
                min_curr, max_curr = current_limits[site_idx]
                site_amplitudes = np.random.rand(n_stims_per_site) * (max_curr - min_curr) + min_curr
                amplitude_blocks.append(np.expand_dims(site_amplitudes, 1))

            stimulation_protocols = scipy.linalg.block_diag(*amplitude_blocks)

        else:
            raise ValueError("Either n_stims or n_stims_per_site must be provided.")

        if out == "current":
            return stimulation_protocols

        extracellular_potential = self.compute_potential_from_stimulation_protocols(
            stimulation_protocols=stimulation_protocols,
            fiber_ids=fiber_ids,
        )

        if out == "potential":
            return extracellular_potential, fiber_ids
        if out == "both":
            return stimulation_protocols, extracellular_potential, fiber_ids

        raise ValueError("out must be one of 'current', 'potential', or 'both'.")

    def compute_potential_from_stimulation_protocols(
        self,
        stimulation_protocols: np.ndarray,
        fiber_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute extracellular potentials for stimulation protocols.

        Parameters
        ----------
        stimulation_protocols : ndarray
            Stimulation protocols with shape ``(n_stims, n_sites)``.
        fiber_ids : ndarray, optional
            Fiber subset for each stimulation protocol, with shape
            ``(n_stims, n_fibers_per_stim)``.

        Returns
        -------
        ndarray
            Extracellular potentials. Shape is
            ``(n_fibers, n_nodes, n_stims)`` if ``fiber_ids`` is None, otherwise
            ``(n_fibers_per_stim, n_nodes, n_stims)``.
        """
        if self.lead_field_matrix is None:
            raise ValueError("A lead-field matrix must be loaded before computing potentials.")

        stimulation_protocols = np.asarray(stimulation_protocols)
        n_stims = stimulation_protocols.shape[0]

        if fiber_ids is None:
            extracellular_potential_fem = self.lead_field_matrix @ stimulation_protocols.T
            extracellular_potential = np.zeros(
                (self.fiber_population.n_fibers, self.fiber_population.n_nodes, n_stims)
            )

            for fiber_idx in range(self.fiber_population.n_fibers):
                idx_current_fiber = np.squeeze(self.fiber_population.fem_node_fiber_ids == fiber_idx)
                idx_fem_nodes = np.squeeze(self.fiber_population.node_ids[fiber_idx, :] == 1)
                extracellular_potential[fiber_idx, idx_fem_nodes, :] = extracellular_potential_fem[idx_current_fiber, :]

        else:
            fiber_ids = np.asarray(fiber_ids, dtype=int)
            n_fibers_per_stim = fiber_ids.shape[1]
            extracellular_potential = np.zeros(
                (n_fibers_per_stim, self.fiber_population.n_nodes, n_stims)
            )

            for stim_idx in range(n_stims):
                for local_fiber_idx in range(n_fibers_per_stim):
                    fiber_idx = fiber_ids[stim_idx, local_fiber_idx]
                    idx_current_fiber = np.squeeze(self.fiber_population.fem_node_fiber_ids == fiber_idx)
                    idx_fem_nodes = np.squeeze(self.fiber_population.node_ids[fiber_idx, :] == 1)
                    extracellular_potential_fem = self.lead_field_matrix[idx_current_fiber, :] @ stimulation_protocols[stim_idx, :].T
                    extracellular_potential[local_fiber_idx, idx_fem_nodes, stim_idx] = extracellular_potential_fem

        return extracellular_potential

    # ---------------------------------------------------------------------
    # Dataset generation and activation prediction
    # ---------------------------------------------------------------------
    def generate_dataset(
        self,
        stimulation_protocols: np.ndarray | None = None,
        n_stims: int | None = None,
        current_limits: np.ndarray | None = None,
        n_active_sites: int | list[int] = 1,
        n_stims_per_site: int | None = None,
        n_fibers_per_stim: int | None = None,
        fiber_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate ML input features from stimulation protocols.

        The dataset contains extracellular potentials at fiber nodes plus fiber
        diameter as the last feature.

        Returns
        -------
        ndarray
            Feature matrix with shape ``(n_samples, n_nodes + 1)``.
        """
        if stimulation_protocols is None:
            if current_limits is None:
                raise ValueError("current_limits must be provided when stimulation_protocols is None.")

            if n_stims is not None:
                extracellular_potential, fiber_ids = self.generate_stimulation_batch(
                    n_stims=n_stims,
                    current_limits=current_limits,
                    n_active_sites=n_active_sites,
                    n_fibers_per_stim=n_fibers_per_stim,
                )
            elif n_stims_per_site is not None:
                n_stims = int(n_stims_per_site * self.implant.n_sites)
                extracellular_potential, fiber_ids = self.generate_stimulation_batch(
                    n_stims_per_site=n_stims_per_site,
                    current_limits=current_limits,
                    n_active_sites=n_active_sites,
                    n_fibers_per_stim=n_fibers_per_stim,
                )
            else:
                raise ValueError("Either stimulation_protocols, n_stims, or n_stims_per_site must be provided.")
        else:
            stimulation_protocols = np.asarray(stimulation_protocols)
            n_stims = stimulation_protocols.shape[0]
            extracellular_potential = self.compute_potential_from_stimulation_protocols(
                stimulation_protocols,
                fiber_ids=fiber_ids,
            )

        if fiber_ids is None:
            diameters = np.reshape(self.fiber_population.diameters, [self.fiber_population.n_fibers, 1, 1])
            diam_tiled = np.tile(diameters, [1, 1, n_stims])
            n_fibers_used = self.fiber_population.n_fibers
        else:
            fiber_ids = np.asarray(fiber_ids, dtype=int)
            diam_tiled = np.expand_dims(self.fiber_population.diameters[fiber_ids.T], 1)
            n_fibers_used = fiber_ids.shape[1]

        dataset_by_fiber = np.concatenate((extracellular_potential, diam_tiled), axis=1)
        dataset_by_stim = np.transpose(dataset_by_fiber, [2, 0, 1])

        n_features = self.fiber_population.n_nodes + 1
        dataset = np.reshape(dataset_by_stim, [n_stims * n_fibers_used, n_features])
        return dataset

    def generate_dataset_fr(self, *args, **kwargs) -> np.ndarray:
        """Backward-compatible alias for :meth:`generate_dataset`."""
        return self.generate_dataset(*args, **kwargs)

    def _shift_dataset_to_nonzero_nodes(self, dataset_reshaped: np.ndarray) -> np.ndarray:
        """Move non-zero node potentials to the beginning of each feature vector.

        The final feature, corresponding to fiber diameter, is preserved.
        """
        shifted_dataset = np.zeros(dataset_reshaped.shape)

        for stim_idx in range(dataset_reshaped.shape[0]):
            for fiber_idx in range(dataset_reshaped.shape[1]):
                non_zero_features = np.nonzero(dataset_reshaped[stim_idx, fiber_idx, :])[0]
                non_zero_potential_features = non_zero_features[
                    non_zero_features < dataset_reshaped.shape[2] - 1
                ]
                n_nonzero = len(non_zero_potential_features)
                shifted_dataset[stim_idx, fiber_idx, :n_nonzero] = dataset_reshaped[
                    stim_idx,
                    fiber_idx,
                    non_zero_potential_features,
                ]
                shifted_dataset[stim_idx, fiber_idx, -1] = dataset_reshaped[stim_idx, fiber_idx, -1]

        return shifted_dataset

    def predict_activations(
        self,
        stimulation_protocols: np.ndarray,
        method: ActivationMethod = "from_self",
        thr: float = 100,
        output_potential: bool = False,
    ):
        """Predict fiber activations for a batch of stimulation protocols.

        Parameters
        ----------
        stimulation_protocols : ndarray
            Stimulation protocols with shape ``(n_stims, n_sites)``.
        method : {"from_self", "mirror", "actfun"}, default="from_self"
            Activation prediction method.
        thr : float, default=100
            Threshold used by analytical methods.
        output_potential : bool, default=False
            If true, also return the shifted dataset used by the classifier.

        Returns
        -------
        ndarray or tuple
            Binary activation matrix with shape ``(n_stims, n_fibers)``.
        """
        dataset = self.generate_dataset(stimulation_protocols)
        n_stims = stimulation_protocols.shape[0]
        n_features = dataset.shape[1]
        dataset_reshaped = np.reshape(
            dataset,
            [n_stims, self.fiber_population.n_fibers, n_features],
        )

        shifted_dataset = None

        if method == "from_self":
            if self.activation_predictor is None:
                raise ValueError("activation_predictor must be loaded when method='from_self'.")

            shifted_dataset = self._shift_dataset_to_nonzero_nodes(dataset_reshaped)
            activations = np.zeros((n_stims, self.fiber_population.n_fibers))

            for stim_idx in range(n_stims):
                activations[stim_idx, :] = self.activation_predictor.predict(
                    shifted_dataset[stim_idx, :, :]
                )

        elif method == "mirror":
            ind_closest_to_origin = np.argmin(np.abs(self.fiber_population.locs_ranvier), axis=1)
            ind_selected_node = 11 * ind_closest_to_origin
            v_zero = dataset_reshaped[:, np.arange(self.fiber_population.n_fibers), ind_selected_node]
            activations = v_zero > thr

        elif method == "actfun":
            ranvier_node_order = np.argsort(np.abs(self.fiber_population.locs_ranvier), axis=1)
            ind_closest_to_origin = ranvier_node_order[:, 0]

            ind_zero_node = 11 * ind_closest_to_origin
            ind_pos_node = 11 * np.minimum(ind_closest_to_origin + 1, self.fiber_population.n_internodes)
            ind_neg_node = 11 * np.maximum(ind_closest_to_origin - 1, 0)

            v_zero = dataset_reshaped[:, np.arange(self.fiber_population.n_fibers), ind_zero_node]
            v_pos = dataset_reshaped[:, np.arange(self.fiber_population.n_fibers), ind_pos_node]
            v_neg = dataset_reshaped[:, np.arange(self.fiber_population.n_fibers), ind_neg_node]

            actfun = v_pos - 2 * v_zero + v_neg
            activations = actfun > thr

        else:
            raise ValueError("method must be one of 'from_self', 'mirror', or 'actfun'.")

        activations = (activations > 0.5).astype(int)

        if output_potential:
            if shifted_dataset is None:
                shifted_dataset = dataset_reshaped
            return activations, shifted_dataset[:, :, :-1]

        return activations

    def predict_activations_2(self, *args, **kwargs):
        """Backward-compatible alias for :meth:`predict_activations`."""
        return self.predict_activations(*args, **kwargs)

    def predict_firing_rates(self, stimulation_protocols: np.ndarray, method: str = "sigmoid") -> np.ndarray:
        """Predict approximate firing rates from extracellular potentials."""
        if method != "sigmoid":
            raise ValueError("Only method='sigmoid' is currently supported.")

        dataset = self.generate_dataset(stimulation_protocols)
        n_stims = stimulation_protocols.shape[0]
        dataset_reshaped = np.reshape(
            dataset,
            [n_stims, self.fiber_population.n_fibers, dataset.shape[1]],
        )

        ind_closest_to_origin = np.argmin(np.abs(self.fiber_population.locs_ranvier), axis=1)
        ind_selected_node = 11 * ind_closest_to_origin
        v_zero = dataset_reshaped[:, np.arange(self.fiber_population.n_fibers), ind_selected_node]
        firing_rates = (np.tanh(v_zero - 10) + 1) * 50
        return firing_rates

    # ---------------------------------------------------------------------
    # Recruitment and selectivity
    # ---------------------------------------------------------------------
    def compute_recruitment_patterns(
        self,
        stimulation_protocols: np.ndarray,
        method: ActivationMethod = "from_self",
    ) -> np.ndarray:
        """Compute recruitment fraction for each functional group."""
        activations = self.predict_activations(
            stimulation_protocols=stimulation_protocols,
            method=method,
        )

        recruitment_patterns = np.zeros((stimulation_protocols.shape[0], self.fiber_population.n_groups))

        for group_idx in range(self.fiber_population.n_groups):
            fibers_current_group = np.flatnonzero(self.fiber_population.cluster_ids == group_idx)
            n_fibers_current_group = fibers_current_group.size

            if n_fibers_current_group == 0:
                continue

            recruitment_patterns[:, group_idx] = (
                np.sum(activations[:, fibers_current_group], axis=1) / n_fibers_current_group
            )

        return recruitment_patterns

    def compute_selectivity_patterns(
        self,
        stimulation_protocols: np.ndarray,
        method: ActivationMethod = "from_self",
    ) -> np.ndarray:
        """Compute selectivity patterns from recruitment patterns."""
        recruitment_patterns = self.compute_recruitment_patterns(
            stimulation_protocols=stimulation_protocols,
            method=method,
        )

        denominator = np.sum(recruitment_patterns, axis=1, keepdims=True)
        denominator[denominator == 0] = 1
        return recruitment_patterns**2 / denominator

    def generate_recruitment_curves(
        self,
        amp_lims,
        n_steps,
        method: ActivationMethod = "from_self",
        return_amplitudes: bool = False,
    ):
        """Generate monopolar recruitment curves for all stimulation sites.

        Parameters
        ----------
        amp_lims : sequence or ndarray
            Amplitude limits. Use ``(2,)`` for shared limits across all sites or
            ``(n_sites, 2)`` for site-specific limits.
        n_steps : int or ndarray
            Number of sampled amplitudes per site.
        method : {"from_self", "mirror", "actfun"}, default="from_self"
            Activation prediction method.
        return_amplitudes : bool, default=False
            If true, return both the :class:`RecruitmentCurves` object and the
            amplitude array/list.

        Returns
        -------
        RecruitmentCurves or tuple
            Recruitment curves, optionally with amplitudes.
        """
        amplitude_list, amplitudes = self._build_monopolar_amplitude_list(amp_lims, n_steps)
        stimulation_protocols = scipy.linalg.block_diag(*amplitude_list)

        activations = self.predict_activations(
            stimulation_protocols=stimulation_protocols,
            method=method,
        )

        shared_n_steps = np.isscalar(n_steps)

        if shared_n_steps:
            recruitment_values = np.zeros((self.implant.n_sites, self.fiber_population.n_groups, int(n_steps)))
        else:
            n_steps = np.asarray(n_steps, dtype=int)
            recruitment_values = [
                np.zeros((self.fiber_population.n_groups, n_steps[site_idx]))
                for site_idx in range(self.implant.n_sites)
            ]

        current_ind = 0
        for site_idx in range(self.implant.n_sites):
            n_steps_current_site = int(n_steps if shared_n_steps else n_steps[site_idx])
            stims_current_site = np.arange(current_ind, current_ind + n_steps_current_site)

            for group_idx in range(self.fiber_population.n_groups):
                fibers_current_group = np.flatnonzero(self.fiber_population.cluster_ids == group_idx)
                n_fibers_current_group = fibers_current_group.size

                if n_fibers_current_group == 0:
                    continue

                group_recruitment = (
                    np.sum(activations[np.ix_(stims_current_site, fibers_current_group)], axis=1)
                    / n_fibers_current_group
                )

                if shared_n_steps:
                    recruitment_values[site_idx, group_idx, :] = group_recruitment
                else:
                    recruitment_values[site_idx][group_idx, :] = group_recruitment

            current_ind += n_steps_current_site

        recruitment_curves = RecruitmentCurves(
            recruitment_values=recruitment_values,
            amplitudes=amplitudes,
        )

        if return_amplitudes:
            return recruitment_curves, amplitudes
        return recruitment_curves

    def generate_recruitment_curves_from_points(
        self,
        amplitudes,
        shared_amplitudes: bool = True,
        method: ActivationMethod = "from_self",
    ) -> RecruitmentCurves:
        """Generate recruitment curves from explicitly provided amplitudes."""
        if shared_amplitudes:
            amplitudes = np.asarray(amplitudes)
            n_steps = len(amplitudes)
            amplitude_list = [np.reshape(amplitudes, [n_steps, 1]) for _ in range(self.implant.n_sites)]
            amplitudes_for_container = amplitudes
        else:
            n_steps = np.array([len(amplitudes[site_idx]) for site_idx in range(len(amplitudes))])
            amplitude_list = [
                np.reshape(amplitudes[site_idx], [n_steps[site_idx], 1])
                for site_idx in range(self.implant.n_sites)
            ]
            amplitudes_for_container = amplitudes

        stimulation_protocols = scipy.linalg.block_diag(*amplitude_list)
        activations = self.predict_activations(
            stimulation_protocols=stimulation_protocols,
            method=method,
        )

        if shared_amplitudes:
            recruitment_values = np.zeros((self.implant.n_sites, self.fiber_population.n_groups, n_steps))
        else:
            recruitment_values = [
                np.zeros((self.fiber_population.n_groups, n_steps[site_idx]))
                for site_idx in range(self.implant.n_sites)
            ]

        current_ind = 0
        for site_idx in range(self.implant.n_sites):
            n_steps_current_site = int(n_steps if shared_amplitudes else n_steps[site_idx])
            stims_current_site = np.arange(current_ind, current_ind + n_steps_current_site)

            for group_idx in range(self.fiber_population.n_groups):
                fibers_current_group = np.flatnonzero(self.fiber_population.cluster_ids == group_idx)
                n_fibers_current_group = fibers_current_group.size

                if n_fibers_current_group == 0:
                    continue

                group_recruitment = (
                    np.sum(activations[np.ix_(stims_current_site, fibers_current_group)], axis=1)
                    / n_fibers_current_group
                )

                if shared_amplitudes:
                    recruitment_values[site_idx, group_idx, :] = group_recruitment
                else:
                    recruitment_values[site_idx][group_idx, :] = group_recruitment

            current_ind += n_steps_current_site

        return RecruitmentCurves(
            recruitment_values=recruitment_values,
            amplitudes=amplitudes_for_container,
        )

    def compute_ix(
        self,
        recruitment_level: float,
        amp_lims,
        n_steps,
        method: ActivationMethod = "from_self",
    ) -> np.ndarray:
        """Compute current at a selected recruitment level."""
        recruitment_curves = self.generate_recruitment_curves(
            amp_lims=amp_lims,
            n_steps=n_steps,
            method=method,
        )
        return recruitment_curves.compute_ix(recruitment_level=recruitment_level)

    def compute_i50(self, amp_lims, n_steps, method: ActivationMethod = "from_self") -> np.ndarray:
        """Compute current at 50% recruitment."""
        recruitment_curves = self.generate_recruitment_curves(
            amp_lims=amp_lims,
            n_steps=n_steps,
            method=method,
        )
        return recruitment_curves.compute_i50()

    def compute_activation_threshold(self, amp_lims, n_steps, method: ActivationMethod = "from_self") -> np.ndarray:
        """Compute current at 10% recruitment."""
        recruitment_curves = self.generate_recruitment_curves(
            amp_lims=amp_lims,
            n_steps=n_steps,
            method=method,
        )
        return recruitment_curves.compute_activation_threshold()

    def compute_saturation_threshold(self, amp_lims, n_steps, method: ActivationMethod = "from_self") -> np.ndarray:
        """Compute current at 90% recruitment."""
        recruitment_curves = self.generate_recruitment_curves(
            amp_lims=amp_lims,
            n_steps=n_steps,
            method=method,
        )
        return recruitment_curves.compute_saturation_threshold()

    def generate_selectivity_curves(self):
        """Placeholder retained for backward compatibility."""
        raise NotImplementedError("generate_selectivity_curves is not implemented.")

    # ---------------------------------------------------------------------
    # Plotting and helpers
    # ---------------------------------------------------------------------
    def plot(self, ax) -> None:
        """Plot fiber population, nerve topography, and implant sites."""
        self.fiber_population.plot(ax=ax)
        if self.nerve_topography is not None:
            self.nerve_topography.plot(ax=ax)
        self.implant.plot(ax=ax)

    def _normalize_current_limits(self, current_limits) -> np.ndarray:
        """Convert current limits to shape ``(n_sites, 2)``."""
        current_limits = np.asarray(current_limits, dtype=float)

        if current_limits.size == 2:
            shared_limits = np.zeros((self.implant.n_sites, 2))
            shared_limits[:, 0] = current_limits[0]
            shared_limits[:, 1] = current_limits[1]
            return shared_limits

        if current_limits.shape != (self.implant.n_sites, 2):
            raise ValueError(
                "current_limits must have shape (2,) or (n_sites, 2). "
                f"Received {current_limits.shape}."
            )

        return current_limits

    @staticmethod
    def _sample_n_active_sites(n_active_sites: int | list[int]) -> int:
        """Sample or return the number of concurrently active sites."""
        if np.isscalar(n_active_sites):
            return int(n_active_sites)

        n_active_sites = list(n_active_sites)
        return int(n_active_sites[np.random.randint(len(n_active_sites))])

    def _build_monopolar_amplitude_list(self, amp_lims, n_steps):
        """Build amplitude blocks for monopolar stimulation curves."""
        if np.isscalar(n_steps):
            n_steps = int(n_steps)
            amp_lims = np.asarray(amp_lims, dtype=float)
            amplitudes = np.reshape(np.linspace(amp_lims[0], amp_lims[1], n_steps), [n_steps, 1])
            amplitude_list = [amplitudes for _ in range(self.implant.n_sites)]
            return amplitude_list, amplitudes.flatten()

        n_steps = np.asarray(n_steps, dtype=int)
        amp_lims = np.asarray(amp_lims, dtype=float)

        amplitude_list = []
        amplitudes = []

        for site_idx in range(self.implant.n_sites):
            site_amplitudes = np.reshape(
                np.linspace(amp_lims[site_idx, 0], amp_lims[site_idx, 1], n_steps[site_idx]),
                [n_steps[site_idx], 1],
            )
            amplitude_list.append(site_amplitudes)
            amplitudes.append(site_amplitudes.flatten())

        return amplitude_list, amplitudes

    @property
    def fiber_population(self):
        return self._fiber_population

    @property
    def nerve_topography(self):
        return self._nerve_topography

    @property
    def implant(self):
        return self._implant

    @property
    def lead_field_matrix(self):
        return self._lead_field_matrix

    @property
    def lead_field_matrix_per_fiber(self):
        return self._lead_field_matrix_per_fiber

    @property
    def activation_predictor(self):
        return self._activation_predictor
