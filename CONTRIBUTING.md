# Contributing to TrajectoryCache

Thank you for your interest in contributing! This document explains how to get involved.

## Development Setup

```bash
git clone https://github.com/trajectorycache/trajectorycache.git
cd trajectorycache
pip install -e ".[dev]"
sudo apt-get install libspatialindex-dev && pip install rtree  # optional but recommended
```

## Running Tests

```bash
make test          # Full test suite
make test-unit     # Unit tests only
make smoke         # Smoke test
```

## Code Style

We use **black** for formatting and **ruff** for linting:

```bash
make fmt    # Format with black
make lint   # Lint with ruff
```

All code must pass `ruff` and be formatted with `black --line-length 100`.

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `develop`
2. Write tests for new functionality
3. Ensure `make test` and `make smoke` pass
4. Add a brief description of your changes to `CHANGELOG.md`
5. Open a PR against `develop` (not `main`)

## Priority Contribution Areas

- **Multi-RSU cooperative caching** — Extending MRS to account for neighbor RSU state
- **EKF/UKF trajectory predictor** — Replace CTRV with Extended Kalman Filter
- **Authenticated BSM integration** — ETSI ITS PKI / SCMS certificate handling
- **Hardware testbed validation** — Cohda MK5/MK6 OBU + Savari RSU deployment
- **V2I Interest prediction** — Anticipatory pre-fetching based on route information

## Reporting Issues

Please use the GitHub Issues tracker. For security vulnerabilities, email the maintainers directly.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
