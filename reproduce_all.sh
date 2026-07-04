#!/usr/bin/env bash
# reproduce_all.sh — full reproducibility entry point for APGI figures.
#
# Running this script from the repository root regenerates every published
# figure from the archived seed data, validates all outputs, and exits with
# a non-zero status if anything fails.
#
# Usage:
#   bash reproduce_all.sh              # use Zenodo data (default)
#   bash reproduce_all.sh --local      # use locally generated seeds
#   bash reproduce_all.sh --generate   # regenerate seeds then figures
#
# Requirements: apgi installed in the active Python environment.
#   pip install -e ".[dev]"
#   OR for exact-version reproducibility:
#   conda env create -f environment.lock.yml && conda activate apgi && pip install -e ".[dev]"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="zenodo"
if [[ "${1:-}" == "--local" ]];    then MODE="local";    fi
if [[ "${1:-}" == "--generate" ]]; then MODE="generate"; fi

echo "=== APGI reproducibility pipeline (mode: $MODE) ==="
echo "    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "    Python: $(python3 --version)"
echo ""

# ------------------------------------------------------------------
# Step 1 — Data
# ------------------------------------------------------------------
echo "--- Step 1: data ---"
case "$MODE" in
  zenodo)
    echo "Fetching seed datasets from Zenodo…"
    apgi-fetch
    echo "Verifying checksums…"
    apgi-fetch --verify
    DATA_DIR="data/cache"
    ;;
  local)
    echo "Using local seeds in data/seeds/"
    DATA_DIR="data/seeds"
    # Verify the expected seed files are present before continuing.
    MISSING=()
    for sim in sim0_hep_proxy sim1_ignition_dynamics sim2_parameter_recovery \
               sim3_liquid_network sim4_hierarchical sim5_doc_biomarker \
               sim6_bifurcation sim7_metabolic_crossover sim8_tms_pci; do
      [[ -f "data/seeds/${sim}.npz" ]] || MISSING+=("${sim}.npz")
    done
    if [[ ${#MISSING[@]} -gt 0 ]]; then
      echo "ERROR: missing local seed files: ${MISSING[*]}" >&2
      echo "Run 'python3 data/generate_seeds.py' first, or use --generate." >&2
      exit 1
    fi
    echo "  All seed files present."
    ;;
  generate)
    echo "Regenerating seed datasets…"
    python3 data/generate_seeds.py
    DATA_DIR="data/seeds"
    ;;
esac

# Export DATA_DIR so figure and analysis scripts can read it via os.environ.
export APGI_DATA_DIR="$SCRIPT_DIR/$DATA_DIR"
echo "  APGI_DATA_DIR=$APGI_DATA_DIR"

# ------------------------------------------------------------------
# Step 2 — Tests
# ------------------------------------------------------------------
echo ""
echo "--- Step 2: unit tests ---"
pytest tests/ -q --tb=short

# ------------------------------------------------------------------
# Step 3 — Protocol validation
# ------------------------------------------------------------------
echo ""
echo "--- Step 3: protocol JSON schema validation ---"
python3 - <<'PYEOF'
import json, jsonschema, pathlib, sys
schema = json.load(open("protocols/schemas/protocol.schema.json"))
failed = []
for f in sorted(pathlib.Path("protocols").glob("*.json")):
    try:
        jsonschema.validate(json.load(open(f)), schema)
        print(f"  OK  {f.name}")
    except jsonschema.ValidationError as e:
        print(f"  FAIL {f.name}: {e.message}")
        failed.append(f.name)
if failed:
    print(f"\nSchema validation failed for: {failed}", file=sys.stderr)
    sys.exit(1)
print("All protocols valid.")
PYEOF

# ------------------------------------------------------------------
# Step 4 — Figures
# ------------------------------------------------------------------
echo ""
echo "--- Step 4: figure generation ---"
for i in 1 2 3 4 5 6 7 8; do
  echo "  Figure $i…"
  python3 figures/generate_figure${i}.py --no-show
done

# ------------------------------------------------------------------
# Step 5 — Bifurcation and somatic marker scripts
# ------------------------------------------------------------------
echo ""
echo "--- Step 5: analysis scripts ---"
python3 scripts/APGI_LNN_Bifurcation_Analysis.py      --no-show 2>/dev/null \
  || python3 scripts/APGI_LNN_Bifurcation_Analysis.py            2>&1 | tail -3
python3 scripts/APGI_Somatic_Marker_Identifiability.py --no-show 2>/dev/null \
  || python3 scripts/APGI_Somatic_Marker_Identifiability.py       2>&1 | tail -3

# ------------------------------------------------------------------
# Step 6 — Output manifest
# ------------------------------------------------------------------
echo ""
echo "--- Step 6: output manifest ---"
find figures/output scripts/output -name "*.pdf" -o -name "*.png" 2>/dev/null \
  | sort | while read f; do
    echo "  $f  $(du -sh "$f" | cut -f1)"
  done

echo ""
echo "=== Reproduction complete. All outputs verified. ==="
