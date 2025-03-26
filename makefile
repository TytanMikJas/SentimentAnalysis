build:
	docker build --tag lab03-06-python .

format:
	docker run --rm -v "$(CURDIR):/app" -w /app lab03-06-python ruff format

run_docker:
	docker run --memory=4g --memory-swap=4g --rm -v "$(CURDIR):/app" -w /app -it lab03-06-python /bin/bash
