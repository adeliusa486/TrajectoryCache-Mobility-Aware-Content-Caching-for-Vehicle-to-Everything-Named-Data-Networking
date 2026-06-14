.PHONY: install test smoke lint fmt typecheck demo eval docker clean help

# ── Development ──────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-rtree:
	sudo apt-get install -y libspatialindex-dev
	pip install rtree>=1.2

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short -k "not test_tc_vs_lru"

smoke:
	python scripts/smoke_test.py

coverage:
	pytest tests/ --cov=trajectorycache --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check trajectorycache/ api/ tests/ scripts/

fmt:
	black trajectorycache/ api/ tests/ scripts/

typecheck:
	mypy trajectorycache/ --ignore-missing-imports

# ── Running ───────────────────────────────────────────────────────────────────

demo:
	python -m trajectorycache.demo --scenario highway --vehicles 100 --duration 60

demo-urban:
	python -m trajectorycache.demo --scenario urban --vehicles 100 --duration 60

api:
	uvicorn api.main:app --reload --port 8000

eval-fast:
	python scripts/run_evaluation.py --scenario highway --fast --sweep all

eval-full:
	python scripts/run_evaluation.py --scenario all --full --sweep all

plots:
	python scripts/run_evaluation.py --plot-only

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t trajectorycache:latest .

docker-run:
	docker-compose up -d api

docker-smoke:
	docker run --rm trajectorycache:latest python scripts/smoke_test.py

docker-eval:
	docker-compose --profile eval up eval

docker-monitoring:
	docker-compose --profile monitoring up -d

docker-down:
	docker-compose down

# ── Utility ───────────────────────────────────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build

help:
	@echo ""
	@echo "TrajectoryCache Makefile targets:"
	@echo "  install          Install package in editable mode + dev deps"
	@echo "  install-rtree    Install libspatialindex + rtree Python binding"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  smoke            Run smoke test script"
	@echo "  coverage         Run tests with coverage report"
	@echo "  lint             Lint with ruff"
	@echo "  fmt              Format with black"
	@echo "  typecheck        Type check with mypy"
	@echo "  demo             Quick highway demo (100 vehicles, 60 s)"
	@echo "  api              Start FastAPI server (dev mode)"
	@echo "  eval-fast        Fast evaluation sweep (3 seeds, 150 s)"
	@echo "  eval-full        Full evaluation sweep (10 seeds, 600 s)"
	@echo "  plots            Regenerate all figures from saved results"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Start API container"
	@echo "  docker-smoke     Run smoke test in container"
	@echo "  clean            Remove build artifacts"
	@echo ""
