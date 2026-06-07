# Contributing to APGI Framework

Thank you for your interest in contributing. This document covers the minimal
steps to get a working environment, run the tests, and open a pull request.

## Environment setup

```bash
git clone https://github.com/lesoto/apgi-framework.git
cd apgi-framework

# Option A — pip (any Python 3.11+)
pip install -e ".[dev]"

# Option B — conda (exact versions for HPC / clean-machine replication)
conda env create -f environment.yml
conda activate apgi
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v              # full suite with coverage report
pytest tests/ -k core         # run only core-equation tests
pytest --nbval-lax notebooks/quick_start.ipynb   # validate a notebook
```

All 213 tests must pass before a pull request is merged. Coverage must remain
at 100% for modules under `src/apgi/`.

## Regenerating seed data

Seed `.npz` files are not committed. To regenerate locally:

```bash
python data/generate_seeds.py           # all six datasets → data/seeds/
python data/generate_seeds.py --dataset sim1_ignition_dynamics   # one dataset
python data/generate_seeds.py --verify  # check digests
```

Upload `data/seeds/*.npz` to Zenodo before updating `src/apgi/scripts/fetch_data.py`.

## Reproducing all figures

```bash
bash reproduce_all.sh           # fetch → test → validate → 8 figures
bash reproduce_all.sh --local   # use locally generated seeds instead of Zenodo
```

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/<short-name>` | `feature/add-hep-filter` |
| Bug fix | `fix/<issue-number>-<description>` | `fix/42-theta-update` |
| Protocol | `protocol/<id>-<label>` | `protocol/p07-meg-source` |
| Figure | `figure/<number>-<label>` | `figure/9-parameter-space` |

## Adding a protocol

1. Create `protocols/protocol_N_<name>.json` following the schema in
   `protocols/schemas/protocol.schema.json`.
2. Run schema validation locally: `python -c "import json, jsonschema, pathlib; ..."`
   (see the CI step in `.github/workflows/test.yml` for the full one-liner).
3. Add a matching notebook in `notebooks/` demonstrating the APGI predictions.
4. Register the protocol on OSF and update `osf_url` in the JSON file.

## Adding a figure

1. Create `figures/generate_figureN.py` following the pattern of existing
   figure scripts — import from `figures.utils`, end with `save_figure(...)`.
2. Add `--no-show` argument support (used by CI).
3. Add the script to the Figure smoke tests block in `.github/workflows/test.yml`.
4. Commit the generated PDF in `figures/output/` so reviewers can inspect
   figures without running the full pipeline.

## Code style

- No formatter is enforced, but match the existing style (snake_case,
  Google-style docstrings, type annotations on all public functions).
- Do not add comments that merely restate what the code does — only add them
  when explaining a non-obvious constraint or equation derivation.
- New public API functions must appear in `src/apgi/__init__.py`'s `__all__`.

## Reporting issues

Open a GitHub issue at <https://github.com/lesoto/apgi-framework/issues>.
For falsification-criterion failures (a simulation produces results outside
the pre-registered bounds in a protocol JSON), label the issue `falsification`.
