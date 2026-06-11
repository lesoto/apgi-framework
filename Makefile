# APGI Framework — Makefile
#
# Reproduces all figures, runs tests, validates protocols, and manages the
# development environment for the APGI paper series.
#
# Quick reference
# ───────────────
#   make install          — install package (runtime deps only)
#   make install-dev      — install package + all dev/lint/notebook deps
#   make reproduce        — full pipeline via reproduce_all.sh (Zenodo data)
#   make reproduce-local  — full pipeline using local seeds (no download)
#   make reproduce-gen    — regenerate seeds then run full pipeline
#   make figures          — generate all 8 paper figures (no-show mode)
#   make figure-1 … figure-8  — individual figure targets
#   make analysis         — run bifurcation + somatic-marker analysis scripts
#   make seeds            — regenerate seed datasets from scratch
#   make validate         — validate all protocol JSON files against schema
#   make notebooks        — execute and validate all Jupyter notebooks (nbval)
#   make test             — run full pytest suite with coverage
#   make test-fast        — run tests without coverage (faster)
#   make lint             — ruff style check
#   make typecheck        — mypy type check on src/apgi
#   make clean            — remove __pycache__ / .pyc (keeps figure outputs)
#   make clean-outputs    — remove generated PDFs and PNGs
#   make clean-all        — clean + clean-outputs

PYTHON       ?= python3
FIGURES_DIR   = figures
SCRIPTS_DIR   = scripts
DATA_DIR      = data
PROTOCOLS_DIR = protocols
NOTEBOOKS_DIR = notebooks
FIG_OUT       = $(FIGURES_DIR)/output
SCRIPT_OUT    = $(SCRIPTS_DIR)/output

# ─── Phony targets ────────────────────────────────────────────────────────────
.PHONY: install install-dev \
        reproduce reproduce-local reproduce-gen \
        figures \
        figure-1 figure-2 figure-3 figure-4 \
        figure-5 figure-6 figure-7 figure-8 \
        analysis seeds validate notebooks \
        test test-fast lint typecheck \
        clean clean-outputs clean-all \
        help

# ─── Default target ───────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "APGI Framework — available targets"
	@echo "─────────────────────────────────────────────────────────"
	@echo "  install          Install runtime package"
	@echo "  install-dev      Install package + dev/lint/notebook deps"
	@echo ""
	@echo "  reproduce        Full pipeline (Zenodo seed download)"
	@echo "  reproduce-local  Full pipeline using local seeds"
	@echo "  reproduce-gen    Regenerate seeds then full pipeline"
	@echo ""
	@echo "  figures          Generate all 8 figures (no-show)"
	@echo "  figure-1 … figure-8  Individual figure targets"
	@echo "  analysis         Bifurcation + somatic-marker scripts"
	@echo "  seeds            Regenerate seed datasets"
	@echo "  validate         Validate protocol JSON files"
	@echo "  notebooks        Execute + validate Jupyter notebooks"
	@echo ""
	@echo "  test             Full pytest suite with coverage"
	@echo "  test-fast        pytest without coverage"
	@echo "  lint             ruff style check"
	@echo "  typecheck        mypy type check"
	@echo ""
	@echo "  clean            Remove __pycache__ / .pyc"
	@echo "  clean-outputs    Remove generated PDFs and PNGs"
	@echo "  clean-all        clean + clean-outputs"
	@echo ""

# ─── Installation ─────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ─── Full reproducibility pipeline ────────────────────────────────────────────
reproduce:
	bash reproduce_all.sh

reproduce-local:
	bash reproduce_all.sh --local

reproduce-gen:
	bash reproduce_all.sh --generate

# ─── Figure generation ────────────────────────────────────────────────────────
$(FIG_OUT):
	mkdir -p $(FIG_OUT)

figures: $(FIG_OUT)
	@echo "Generating all figures…"
	@for i in 1 2 3 4 5 6 7 8; do \
	  echo "  Figure $$i…"; \
	  $(PYTHON) $(FIGURES_DIR)/generate_figure$$i.py --no-show; \
	done
	@echo "Figures written to $(FIG_OUT)/"

figure-1: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure1.py --no-show

figure-2: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure2.py --no-show

figure-3: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure3.py --no-show

figure-4: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure4.py --no-show

figure-5: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure5.py --no-show

figure-6: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure6.py --no-show

figure-7: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure7.py --no-show

figure-8: $(FIG_OUT)
	$(PYTHON) $(FIGURES_DIR)/generate_figure8.py --no-show

# ─── Analysis scripts ─────────────────────────────────────────────────────────
$(SCRIPT_OUT):
	mkdir -p $(SCRIPT_OUT)

analysis: $(SCRIPT_OUT)
	@echo "Running bifurcation analysis…"
	$(PYTHON) $(SCRIPTS_DIR)/APGI_LNN_Bifurcation_Analysis.py --no-show
	@echo "Running somatic-marker identifiability…"
	$(PYTHON) $(SCRIPTS_DIR)/APGI_Somatic_Marker_Identifiability.py --no-show

# ─── Seed data ────────────────────────────────────────────────────────────────
seeds:
	@echo "Regenerating seed datasets…"
	$(PYTHON) $(DATA_DIR)/generate_seeds.py

# ─── Protocol validation ──────────────────────────────────────────────────────
validate:
	@echo "Validating protocol JSON files…"
	$(PYTHON) $(SCRIPTS_DIR)/validate_protocols.py

# ─── Notebooks ────────────────────────────────────────────────────────────────
notebooks:
	@echo "Executing and validating notebooks…"
	$(PYTHON) -m pytest --nbval $(NOTEBOOKS_DIR)/ -v

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v --no-cov

# ─── Code quality ─────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check src/ figures/ scripts/ tests/

typecheck:
	$(PYTHON) -m mypy src/apgi

# ─── Cleaning ─────────────────────────────────────────────────────────────────
clean:
	$(PYTHON) delete_pycache.py --yes --keep-visualizations --keep-reports

clean-outputs:
	rm -f $(FIG_OUT)/*.pdf $(FIG_OUT)/*.png
	rm -f $(SCRIPT_OUT)/*.pdf $(SCRIPT_OUT)/*.png

clean-all: clean clean-outputs
