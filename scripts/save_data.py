"""
Re-save INFORM experiment pickles as self-contained, future-proof files.

Goal (minimal-risk strategy)
----------------------------
The existing experiment pickles were saved when the model classes lived at the
Python top level (``experiment``, ``fiber_population``, ...). Loading them
therefore requires a ``sys.modules`` shim; a clean INFORM install cannot open
them as-is. This script removes that hidden dependency with the smallest
possible change:

1. It loads each original pickle using the legacy shim (read-only; originals
   are never modified).
2. It re-saves an identical copy under a NEW folder, this time pickled with the
   current ``nerve_model.*`` module layout, so the copy loads with no shim.
3. Alongside each experiment it also writes the embedded XGBoost classifier as
   a native JSON (``.json``) -- a format that survives library upgrades far
   better than pickle. This is a backup; the self-contained pickle still works
   on its own.
4. For one file, it runs a CLEAN-ENVIRONMENT check: it re-loads the produced
   copy in a subprocess WITHOUT the shim, proving the copy is truly
   self-contained before you trust the whole batch.

Nothing scientific is changed: arrays, classifiers and parameters are copied
verbatim. Only the module names recorded inside the pickle change.

MUST be run in the same environment ('scientific') that created the pickles, so
that library versions match and re-serialization is faithful.

Usage
-----
python scripts/save_data.py \
    --project-folder /path/to/Models \
    --output-folder  /path/to/inform-data-self-contained \
    [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path


def install_legacy_pickle_shim() -> None:
    """Map legacy top-level module names onto the nerve_model package."""
    try:
        import nerve_model.experiment as experiment
        import nerve_model.fiber_population as fiber_population
        import nerve_model.nerve_section as nerve_section
        import nerve_model.histological_nerve_section as histological_nerve_section
        import nerve_model.recruitment_curves as recruitment_curves
        import nerve_model.implant as implant
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Import of nerve_model failed. Run 'pip install -e .' in the "
            "'scientific' environment before running this script."
        ) from exc

    sys.modules.setdefault("experiment", experiment)
    sys.modules.setdefault("fiber_population", fiber_population)
    sys.modules.setdefault("nerve_section", nerve_section)
    sys.modules.setdefault("recruitment_curves", recruitment_curves)
    sys.modules.setdefault("implant", implant)
    # Legacy median pickles were saved with the polygonal topography in a
    # module called 'nerve_section_2'; it now lives in histological_nerve_section.
    sys.modules.setdefault("nerve_section_2", histological_nerve_section)


def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def export_classifier_json(experiment, out_path: Path) -> str | None:
    """Write the embedded XGBoost classifier as native JSON, if possible."""
    predictor = getattr(experiment, "_activation_predictor", None)
    if predictor is None:
        return None
    save_model = getattr(predictor, "save_model", None)
    if not callable(save_model):
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(str(out_path))
    return out_path.name


def clean_environment_check(copy_path: Path) -> bool:
    """Re-load a produced copy in a subprocess WITHOUT the legacy shim."""
    code = textwrap.dedent(
        f"""
        import pickle, sys
        # Deliberately do NOT install the legacy shim.
        try:
            with open(r"{copy_path}", "rb") as f:
                obj = pickle.load(f)
            print("CLEAN_LOAD_OK", type(obj).__name__)
        except Exception as exc:
            print("CLEAN_LOAD_FAIL", type(exc).__name__, exc)
            sys.exit(1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    print("   clean-check:", (result.stdout or result.stderr).strip())
    return result.returncode == 0


def discover_experiment_files(project_folder: Path) -> list[Path]:
    """Find experiment pickles under the project folder."""
    patterns = ["experiment_trial_*.pkl", "experiment_pop*.pkl"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(project_folder.rglob(pattern)))
    seen, unique = set(), []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-save INFORM pickles as self-contained files."
    )
    parser.add_argument("--project-folder", type=Path, required=True,
                        help="Root of the original working data.")
    parser.add_argument("--output-folder", type=Path, required=True,
                        help="Destination for the self-contained copies "
                             "(originals are never modified).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without writing anything.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files (useful for a first test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_legacy_pickle_shim()

    files = discover_experiment_files(args.project_folder)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print(f"No experiment pickles found under {args.project_folder}")
        return

    print(f"Found {len(files)} experiment pickle(s).")
    if args.dry_run:
        for path in files:
            print("  would re-save:", path.relative_to(args.project_folder))
        return

    checked_one = False
    for path in files:
        rel = path.relative_to(args.project_folder)
        copy_path = args.output_folder / rel
        print(f"\n- {rel}")

        try:
            experiment = load_pickle(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! load failed ({type(exc).__name__}: {exc}); skipping")
            continue

        try:
            save_pickle(copy_path, experiment)
            print(f"  re-saved -> {copy_path.relative_to(args.output_folder)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! re-save failed ({type(exc).__name__}: {exc}); skipping")
            continue

        try:
            json_name = export_classifier_json(
                experiment, copy_path.with_name(copy_path.stem + "_classifier.json")
            )
            if json_name:
                print(f"  classifier JSON -> {json_name}")
            else:
                print("  (no XGBoost-JSON-capable classifier found)")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! classifier JSON export failed ({type(exc).__name__}: {exc})")

        if not checked_one:
            print("  running clean-environment self-containment check...")
            ok = clean_environment_check(copy_path)
            checked_one = True
            if not ok:
                print("\n  !! The re-saved copy did NOT load without the shim.")
                print("     Stopping so you can inspect before processing the rest.")
                return
            print("  clean-environment check PASSED -- copies are self-contained.")

    print("\nDone. Originals were not modified.")


if __name__ == "__main__":
    main()