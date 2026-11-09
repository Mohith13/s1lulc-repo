.PHONY: install dev test lint docker

install:      ## runtime dependencies (prototype + training)
	pip install -e ".[gee,geo,train]"

dev:          ## development tools
	pip install -e ".[dev,geo]" segmentation-models-pytorch matplotlib

test:
	pytest -q

lint:
	ruff check .

docker:
	docker build -t s1lulc .
