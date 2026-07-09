.PHONY: install data backtest report test smoke clean

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

install:
	$(PYTHON) -m pip install --break-system-packages -e ".[dev]"

data:
	$(PYTHON) -m graph_diffusion_signal.cli data

backtest:
	$(PYTHON) scripts/run_pipeline.py

report:
	$(PYTHON) scripts/build_report.py

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke_test.py

clean:
	rm -rf data/processed/* reports/results/* reports/figures/* reports/quant_research_report.md reports/quant_research_report.pdf
	find . -name "__pycache__" -exec rm -rf {} +
