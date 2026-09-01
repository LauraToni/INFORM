# INFORM — Data dictionary

This document describes the data archived on Zenodo and the folder layout the
analysis scripts expect. All spatial quantities are in **millimeters**; currents
follow the convention of the calling code (see per-item notes).

## 1. Archival dataset (Zenodo)

Produced by `scripts/save_data.py` from the working experiment objects. Layout:

```
inform-data-v1.0/
├── leadfield/          # lead-field matrices, one HDF5 per nerve/experiment
│   └── <stem>.h5           # dataset "lead_field_matrix"
├── surrogates/         # trained activation predictors
│   └── <stem>.json         # native XGBoost JSON (preferred)
├── nerve_sections/     # geometry + fiber data
│   └── <stem>.npz          # compressed NumPy arrays (see below)
└── manifest.json       # index of exported files with shapes
```

`<stem>` encodes the source path of the originating experiment pickle (path
separators replaced by `__`), so each item is traceable to its origin.

### 1.1 Lead-field matrices — `leadfield/<stem>.h5`

- Dataset name: `lead_field_matrix`
- Shape: `(n_fem_nodes, n_sites)`
- Meaning: maps a unit current injected at each stimulation site to the
  extracellular potential at each FEM node along the modeled fibers. Computed
  with COMSOL under the quasistatic approximation, one FEM simulation per site
  at 1 µA.

**Scaling — read carefully.** Two scale factors exist in the code and must not
be double-applied:
- When loading raw COMSOL text output, `Experiment.load_lead_field_matrix`
  multiplies by `comsol_scale_factor = 1e3`.
- The `Experiment.lead_field_matrix` *property* multiplies the stored array by
  `1e3` again on access.

The archived `.h5` stores the array as held in `experiment._lead_field_matrix`
(i.e. the raw stored attribute, **before** the property's `*1e3`). Consumers
should apply the same convention as the code: use the property (which applies
`*1e3`) or multiply manually once. Do not apply `1e3` more than once.

### 1.2 Activation predictors — `surrogates/<stem>.json`

- Format: native XGBoost JSON (`Booster.save_model` / `XGBClassifier.save_model`).
- Load with XGBoost's `load_model`, **not** with pickle. Native JSON is portable
  across XGBoost versions; the legacy pickled predictors are not.
- Task: binary classification of fiber activation (active / not active) from
  fiber diameter and extracellular potentials at Ranvier nodes during the
  cathodic phase. One classifier per nerve section, trained on 10,000 samples.
- If an item could not be serialized natively, a `*.pkl` fallback is written
  instead and flagged as `pickle-fallback` in `manifest.json`.

### 1.3 Nerve sections — `nerve_sections/<stem>.npz`

Compressed NumPy archive. Keys present depend on the topography type:

| Key                     | Shape              | Description                                   |
|-------------------------|--------------------|-----------------------------------------------|
| `fiber_locs`            | `(n_fibers, 2)`    | transverse (x, y) fiber locations             |
| `fiber_diameters`       | `(n_fibers,)`      | fiber diameters (µm, Aα range 12–20)          |
| `cluster_ids`           | `(n_fibers,)`      | cluster/muscle index per fiber                |
| `site_locs`             | `(n_sites, 2)`     | stimulation site coordinates                  |
| `cluster_locs`          | `(n_clusters, 2)`  | cluster centers (if available)                |
| `cluster_std`           | `(n_clusters,)`    | cluster dispersions (if available)            |
| `cluster_num`           | `(n_clusters,)`    | fibers per cluster (if available)             |
| `nerve_radius`          | scalar             | circular nerve radius (circular sections)     |
| `fascicles`             | `(n_fascicles, 3)` | circular fascicles: x, y, radius              |
| `fascicle_polygon_<i>`  | `(n_vertices, 2)`  | polygonal fascicle contour (histological)     |

Circular sections carry `fascicles` + `nerve_radius`; histological median-nerve
sections carry `fascicle_polygon_<i>` contours instead.

## 2. Working-data layout expected by the scripts

The run scripts read experiment pickles from a fixed tree under
`--project-folder`. Reproduce this layout after downloading, or regenerate the
pickles from the archived arrays.

```
<project-folder>/
└── Median nerve/
    ├── experiments/                     # "true" experiments
    │   └── <nerve_folder>/<trial>/
    │       ├── experiment_trial_<trial>.pkl
    │       └── experiment_pop<K>trial<trial>.pkl        # per-population (median)
    └── surrogate_experiments/           # "inferred"/surrogate experiments
        └── <nerve_folder><shift>/<trial>/
            ├── experiment_trial_<trial>.pkl
            └── results/                 # written by the scripts
```

Naming conventions (from the scripts):
- `<nerve_folder>` = `block_<block-id>_segment_<segment-id>_nerve_morphology_<mask-id>`
  for the median nerve; circular runs use `nerve_2` by default.
- `<shift>` is an optional suffix selecting the perturbed ("inexact") structure.
- `<trial>` selects the electrode configuration (default `cross`).
- Per-population filenames differ between circular and median runs — see
  `get_population_filename` in `scripts/run_localization.py`.

## 3. Legacy pickle compatibility

Older experiment pickles were saved when the model classes lived at the Python
top level (`experiment`, `fiber_population`, `nerve_section`,
`recruitment_curves`, `implant`). To load them after the move into the
`nerve_model` package, the scripts install a `sys.modules` shim mapping the old
names onto the package modules. `scripts/save_data.py` installs the same shim,
so it can read legacy pickles and re-export them into the portable formats
above. Once re-exported, the shim is no longer needed.

## 4. Reproducibility notes

- Localization uses a Gaussian Process with a Matérn 5/2 kernel and Expected
  Improvement; the reference routine seeds the GP with `random_state=0` and the
  first cluster with 30 random candidate evaluations.
- To make the initial random seeding reproducible run-to-run, set NumPy's seed
  before calling (the wrapper `localize_functional_cluster` accepts a
  `random_state` argument for this).
- Default candidate grid: 20×20 spatial × 20 dispersion × 1 count, restricted to
  centers inside the nerve, giving `(5520, 4)` candidates for a 3 mm radius.
