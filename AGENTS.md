# Repository Guidelines

## Project Structure & Module Organization
- `项目/` holds the core Python workflow for data generation, feature extraction, and model training.
- `项目/data/` stores generated `.npy` datasets by SNR (e.g., `cov_user_iq_normalized_snr-10.npy`, `user_energy_snr0.npy`).
- `文档/` contains the project specification, progress templates, and references used for reporting.
- `类似论文/` and `文献/` contain related papers for literature review.

## Build, Test, and Development Commands
- `python 项目/cov.py` generates multi-user signals, covariance features, and energy baselines into `项目/data/`.
- `python 项目/CNN.py` trains and evaluates the CNN, then reports Pd/Pf/AUC and plots Pd/Pf vs SNR.
- `python 项目/qpsk_singleuser.py` (optional) generates single-user QPSK data for baseline checks.

## Coding Style & Naming Conventions
- Indentation: 4 spaces in Python.
- Naming: snake_case for functions/variables; CapWords for classes (e.g., `SimpleCNN`).
- Data files: include SNR in filename, e.g., `cov_user_iq_normalized_snr-5.npy`.
- Keep numeric constants (SNR list, M, N, sample counts) near the top of scripts.

## Testing Guidelines
- No automated test suite is set up yet.
- Validate changes by running `cov.py` then `CNN.py` and checking:
  - No runtime errors
  - Reasonable Pd/Pf trends across SNR
  - Plots render correctly

## Commit & Pull Request Guidelines
- No established commit convention found; use concise, imperative messages (e.g., "Add ED baseline").
- If submitting a PR, include:
  - Summary of changes
  - Steps to reproduce results (commands run)
  - Any new data artifacts generated

## Security & Configuration Tips
- Datasets are synthetic; no sensitive data is expected.
- Keep large `.npy` outputs in `项目/data/` to avoid cluttering the repo root.
