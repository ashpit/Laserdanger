# Repository Guidelines

## Project Structure & Module Organization
- `matlab/`: Core LiDAR pipelines. `L1_pipeline.m` builds 30-minute DEM-like outputs; `L2_pipeline_testing.m` experiments with higher-frequency swash/runup edges; helpers include `process_lidar_L1.m`, `ResidualKernelFilter.m`, and profile utilities.
- `data/`: Sample data (`plot_test05_17.mat`) for dry runs and visualization checks.
- `python/`: Staging area for the Python rewrite plus figures/tests; currently minimal—mirror MATLAB behavior when adding code.

## Build, Test, and Development Commands
- MATLAB batch runs from `matlab/` with a valid `livox_config.json` there: `matlab -batch "L1_pipeline"` for L1 processing; `matlab -batch "L2_pipeline_testing"` for swash/runup trials. Run from `matlab/` so relative paths resolve.
- MATLAB unit tests (once added): `matlab -batch "runtests"` with tests on the MATLAB path or inside `matlab/`.
- Python tests (for new code under `python/`): `python3 -m pytest` from `python/`.

## Coding Style & Naming Conventions
- MATLAB: 4-space indentation, vectorize when possible, preallocate arrays, and keep file names matching function names (lowerCamelCase with stage tags like `_L1`, `_L2`). Note config requirements at the top of scripts.
- Python: Follow PEP 8 with type hints. Place reusable modules in `python/code/`; keep notebooks or ad-hoc scripts out of version control.
- Logging/prints: favor concise progress messages; avoid hard-coded local paths.

## Testing Guidelines
- No formal suite is present; add MATLAB `runtests` coverage around binning, plane fitting, and runup detection edge cases using small synthetic clouds.
- For pipelines, validate on a short time window and confirm outputs (e.g., `DO` structs, Z(x,t) profiles) before full runs. Save plots/JSON outputs to `plotFolder` and sanity check elevations and timestamps.
- When extending Python, mirror MATLAB outputs on the same inputs and add regression tests in `python/tests/`.

## Commit & Pull Request Guidelines
- Existing history uses short imperative messages; continue that pattern (e.g., `add L2 noise filter`, `tune runup threshold`).
- PRs: include a clear summary, the pipeline(s) touched (L1/L2/Python), config assumptions, sample output paths or screenshots, and performance notes. Link related issues when available.

## Security & Configuration Tips
- `livox_config.json` is required but not versioned; include paths (`dataFolder`, `processFolder`, `plotFolder`), `transformMatrix`, and `LidarBoundary`. Keep credentials and raw data paths out of git; if sharing config, provide a redacted template only.
- Keep large raw scans outside the repo and reference them via environment-specific paths in the config.
