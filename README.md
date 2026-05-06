# Anomaly detection benchmark

This project implements a **reproducible benchmark** of classic tabular outlier detectors (Isolation Forest, one-class SVM, LOF, DBSCAN, ECOD, HBOS) over an ODDS-derived corpus plus two-dimensional stress datasets. Training runs are orchestrated with manifests and tracked in **MLflow**; sensitivity passes (oracle grids, Sobol indices, bootstrap stability, etc.) follow the roadmap below.

**Documentation**

- Full methodological roadmap: [`docs/plan.md`](docs/plan.md)
- Interspeech-style write-up (methods, datasets, experiments—results placeholders): [`report/report.tex`](report/report.tex)

## Requirements

- **[uv](https://docs.astral.sh/uv/)** for environments and scripts
- **Python** matching `requires-python` in [`pyproject.toml`](pyproject.toml) (currently pinned to **3.12.4**)
- **TeX Live** (or similar) only if you compile the PDF from `report/report.tex`

## Commands to reproduce outputs

Run from the repository root.

### 1. Install dependencies and git hooks

```bash
make install
```

### 2. Build datasets (downloads → canonical CSVs → preprocess artifacts → stats → validation)

Produces among others:

- `data/raw/`, `data/canonical/`
- `outputs/descriptive_stats.csv`, `outputs/validation_report.csv`
- PCA/subsample artefacts under `outputs/` per [`DatasetSettings`](src/anomaly_detection/config.py)

```bash
make datasets
```

### 3. Run experiments (primary ladders + optional oracle / blind phases)

The Makefile defaults to **`EXPERIMENTS_PHASE=all`**, which schedules every manifest task for phases 4–5 plus oracle-style workloads—expect long runtimes. Start with **`EXPERIMENTS_PHASE=phase4`** while iterating.

Uses MLflow (`sqlite:///mlruns.db` by default, artefacts under `./mlruns/`). After successful runs, the CLI exports aggregates where applicable:

- `outputs/phase4_summary.csv` — flattened metrics from finished phase-4 runs (when `phase4` or `all` completes cleanly)
- `test_labels.csv` — phase-5 blind export (when `phase5` or `all` completes cleanly)

```bash
make experiments
```

Useful overrides (see `Makefile`):

```bash
# Primary benchmark only (typical first pass)
make experiments EXPERIMENTS_PHASE=phase4 EXPERIMENTS_JOBS=4

# Track B–style grids + Sobol + bootstrap only
make experiments EXPERIMENTS_PHASE=oracle EXPERIMENTS_JOBS=4

# Everything defined in the manifest for the selected phase mode
make experiments EXPERIMENTS_PHASE=all EXPERIMENTS_JOBS=4
```

Show all Make targets:

```bash
make help
```

### 4. Optional — browse runs locally

```bash
make mlflow
# or another port
make mlflow MLFLOW_PORT=5001
```

After experiments, check that **SQLite tracking** (default `mlruns.db`, not `mlflow.db`) holds complete metric payloads for every phase-4 detector run (`MetricsReport` scalars, geometry extras, runtime, heartbeat). Aggregate export runs are skipped.

```bash
make mlflow-audit
# or
uv run python -m anomaly_detection.orchestration.mlflow_audit
```

### 5. Optional — compile the paper PDF

```bash
cd report
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

Output: `report/report.pdf`.

## Quality checks (not required for numerical results)

```bash
make test
make pre-commit-all
```
