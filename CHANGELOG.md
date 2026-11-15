# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-11-15

The app is now one file, `app.py`, which merges the earlier separate tools.

### Added

- Headless smoke test `tests/test_app_smoke.py` (Streamlit `AppTest`): every
  page renders and every button and mode runs without an error; number-input
  defaults are checked.
- English and Traditional Chinese READMEs with a guide to every page,
  installation, GitHub Codespaces and Streamlit Community Cloud notes,
  troubleshooting and limitations.
- MIT license, `CITATION.cff` and this changelog.

### Changed

- `app.py` (formerly `appV1.3.py`) is the only entry point; the dev container
  starts it automatically.

### Removed

- `app V1.py` and `app.V1.2.py`; they remain in the git history.

### Fixed

- Nine `st.number_input` calls passed the default value as the second
  positional argument, which Streamlit reads as `min_value`; for example the
  Fourier interval start could not go below −3.1415. They now pass `value=`
  (and `min_value=` where a lower bound is intended).

## 1.x

Versions 1.0–1.3 were the files `app V1.py`, `app.V1.2.py` and `appV1.3.py`
(added 2025-12-11), preceded by three separate small tools for Fourier
series, Legendre series and the Laplace equation. They were not tagged.

[Unreleased]: https://github.com/Celivem/Electromagnetics-Visualization-Hub/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Celivem/Electromagnetics-Visualization-Hub/releases/tag/v2.0.0
