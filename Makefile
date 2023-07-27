VERSION=$(shell cat model-generator/VERSION)

.DEFAULT_GOAL = help
.PHONY = help
help:
	@echo "Commands:
	@echo "- install					: installs required dependencies ."
	@echo "- install-dev				: installs required dependencies, including dev dependencies ."
	@echo "- check						: check if package is ready for a commit ."
	@echo "- audit						: audits code ."
	@echo "- test						: runs unit tests and behaviour tests ."
	@echo "- unit-tests					: runs unit tests ."
	@echo "- behaviour-tests			: runs behaviour tests ."
	@echo "- badge						: runs badge generation ."
	@echo "- build-docs					: build docs ."
	@echo "- clean						: cleans side effects from commands ."

.PHONY = install
install:
	poetry install --no-dev

.PHONY = install-dev
install-dev:
	poetry install

.PHONY = check
check: install-dev audit test badge clean

.PHONY = audit
audit:
	black model_compra_comigo --line-length=88
	flake8 model_compra_comigo
	isort model_compra_comigo --profile=black

.PHONY = test
test:
	pytest --cov-config=.coveragerc --cov=logger --cov-report html tests

.PHONY = badge
badge:
	coverage-badge -o badges/coverage.svg -f
	anybadge --value=$(VERSION) --file=badges/version.svg --label version -o

.PHONY = build-docs
build-docs:
	make -C docs html

.PHONY = clean
clean:
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pytest_cache' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*.ipynb_checkpoints' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name 'htmlcov' -exec rm -rf {} +
	find . -name '.coverage*' -exec rm -rf {} +
	find . -name 'conda-meta' -exec rm -rf {} +
	find . -name 'mlruns' -exec rm -rf {} +
	find . -name 'auto_model' -exec rm -rf {} +
	find . -name 'time_series_forecaster' -exec rm -rf {} +
