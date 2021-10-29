SHELL := /bin/bash
format:
	autoflake -r -i qsparse tests
	isort qsparse tests
	black qsparse tests setup.py

style-check:
	black --diff --check .

wheel:
	python3 setup.py sdist bdist_wheel

test:
	pytest tests/*.py \
		--cov=qsparse\
		--no-cov-on-fail \
		--cov-report=html:htmlcov \
		--cov-report term \
		--doctest-modules \
		--cov-config=.coveragerc \
		-v \
		qsparse tests


doc:
	mkdocs build

dependency:
	python3 -m pip install -r requirements.txt -r requirements.dev.txt --user

install: dependency
	python3 -m pip install -e . --user
	pre-commit install
