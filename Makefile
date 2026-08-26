.PHONY: lint test format docker-build

lint:
	ruff check .
	mypy src/

test:
	pytest tests/

format:
	ruff format .

docker-build:
	docker build -t syntropyrl .
