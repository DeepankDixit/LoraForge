.PHONY: install test test-all lint format typecheck clean \
        activity1 activity2 activity3 activity4-train activity4-bench \
        activity5-dashboard activity6 full-pipeline

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	pip install -e ".[modelopt]"
	pre-commit install

# ── Quality ───────────────────────────────────────────────────────────────
test:
	pytest tests/ -v -x --ignore=tests/slow -m "not gpu and not slow"

test-all:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check .

format:
	black . && isort .

typecheck:
	mypy . --ignore-missing-imports

# ── Activities ────────────────────────────────────────────────────────────
activity1:
	@echo "Running Activity 1: Baseline Serving Stack..."
	python activity1_baseline/main.py --config activity1_baseline/config.yaml

activity2:
	@echo "Running Activity 2: Quantization Pipeline..."
	python activity2_quantization/main.py --config activity2_quantization/config.yaml

activity3:
	@echo "Running Activity 3: KV Cache Optimization..."
	python activity3_kv_cache/main.py --config activity3_kv_cache/config.yaml

activity4-train:
	@echo "Running Activity 4: EAGLE Draft Model Training..."
	python activity4_speculative_decoding/main.py \
		--config activity4_speculative_decoding/config.yaml \
		--mode train

activity4-bench:
	@echo "Running Activity 4: Speculative Decoding Benchmark..."
	python activity4_speculative_decoding/main.py \
		--config activity4_speculative_decoding/config.yaml \
		--mode benchmark

activity5-dashboard:
	@echo "Starting Benchmark Dashboard..."
	streamlit run activity5_benchmark_dashboard/reporting/dashboard/app.py \
		--server.port 8501

activity6:
	@echo "Running Activity 6: Domain-Adaptive CPT..."
	python activity6_domain_cpt/main.py --config activity6_domain_cpt/config.yaml

# ── Full Pipeline ─────────────────────────────────────────────────────────
full-pipeline:
	@echo "Running full LoraForge pipeline..."
	$(MAKE) activity1
	$(MAKE) activity2
	$(MAKE) activity3
	$(MAKE) activity4-bench  # skip training unless you have time
	@echo "Pipeline complete. Run 'make activity5-dashboard' to view results."

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "Cleaned up cache files."
