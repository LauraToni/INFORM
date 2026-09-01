# INFORM

**Inference of Neural Functional Organization from Recruitment Matching**

INFORM reconstructs the internal *functional* organization of a peripheral
nerve directly from standard post-implantation calibration data (monopolar
recruitment curves), and uses the inferred model to design selective
stimulation protocols, without requiring direct knowledge of the nerve's
internal anatomy.

This repository contains the code accompanying the paper *"Inferring the
internal functional organization of peripheral nerves for selective electrical
stimulation"*. The heavy data (lead-field matrices, trained surrogate
classifiers, nerve sections) are archived separately on Zenodo (see
[Data](#data)).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22229593.svg)](https://doi.org/10.5281/zenodo.22229593)

## What the method does

1. **Model.** A biophysically grounded "hybrid" neuroelectric model maps
   stimulation currents to fiber activation, accelerated by machine-learning
   surrogate classifiers.
2. **Localization.** For each muscle-specific fiber cluster, Bayesian
   optimization searches candidate spatial distributions (center + dispersion)
   to minimize the discrepancy between reference and candidate recruitment
   curves.
3. **Optimization.** Particle Swarm Optimization designs stimulation protocols
   that maximize selectivity on the inferred model; performance is then
   evaluated on the true model.

## Repository layout

```
INFORM/
├── nerve_model/            # geometry, fibers, implant, hybrid stimulation model
│   ├── experiment.py           # ties nerve + fibers + implant + LFM + predictor
│   ├── fiber_population.py      # motor fiber populations & Gaussian clusters
│   ├── nerve_section.py         # circular fascicle topography
│   ├── histological_nerve_section.py  # polygonal (histological) topography
│   ├── implant.py               # electrode/implant geometry
│   └── recruitment_curves.py    # recruitment-curve container & metrics
├── localization/           # Bayesian localization framework
│   ├── candidate_generation.py       # candidate grid (x, y, std, n)
│   ├── localization_utils_reference.py  # VALIDATED reference implementation
│   ├── bayesian_localization.py      # thin interface over the reference
│   ├── localization_result.py        # clean wrapper: localize_functional_cluster
│   └── visualization.py              # plotting helpers
├── selectivity_optimization/   # PSO-based stimulation protocol optimization
├── scripts/                # reproducible entry points (see below)
│   ├── run_localization.py
│   ├── run_localization_histological_median.py
│   └── save_data.py            # convert working pickles -> archival formats
├── tests/
├── pyproject.toml
├── requirements.txt
└── DATA_README.md          # data dictionary (formats, shapes, conventions)
```

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/<your-org>/INFORM.git
cd INFORM
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -e .
```

### Quick check

Run the smoke test to confirm the environment and imports are healthy:

```bash
jupyter nbconvert --to notebook --execute check_localization_inform.ipynb
```

It verifies imports, the candidate grid (expected shape `(5520, 4)` for the
default settings), the `RecruitmentCurves` container, and a tiny localization
run on synthetic objects.

## Data

The numerical data are hosted on Zenodo: **10.5281/zenodo.22229593**

After downloading, expand the archive so that `--project-folder` points to the
root of the expected folder tree. The scripts assume a specific layout
(documented in [`DATA_README.md`](DATA_README.md)), e.g. for the median nerve:

```
<project-folder>/
└── Median nerve/
    ├── experiments/<nerve_folder>/<trial>/experiment_trial_<trial>.pkl
    └── surrogate_experiments/<nerve_folder>/<trial>/experiment_trial_<trial>.pkl
```

Working data are distributed as portable formats (HDF5 for lead-field matrices,
native XGBoost JSON for classifiers, compressed NumPy for geometry). If you
have legacy pickles, `scripts/save_data.py` converts them; see `DATA_README.md`.

## Reproducing the results

The main analyses run through the `scripts/` entry points. Defaults match the
paper's settings.

**Circular simplified nerves** (populations 1–50):

```bash
python scripts/run_localization.py \
    --project-folder /path/to/Models \
    --trial cross \
    --population-start 1 --population-stop 51 \
    --save-plots
```

**Histological median nerve** (12 clusters):

```bash
python scripts/run_localization_histological_median.py \
    --project-folder /path/to/Models \
    --block-id 10 --segment-id 1 --mask-id 1 \
    --trial cross \
    --n-clusters-to-localize 12 \
    --save-plots
```

Add `--shift <suffix>` to run the perturbed-structure ("inexact structure")
condition. Key optimization settings (grid resolution, dispersion range,
tradeoff, iteration cap) are exposed as flags; see `--help` on each script.

## Citing

If you use this code, please cite the paper (citation to be added) and the
Zenodo dataset. A `CITATION.cff` is provided for convenience.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Archived data
on Zenodo are released under their own license, stated on the Zenodo record.
