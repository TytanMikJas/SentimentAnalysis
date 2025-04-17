build:
	docker build --tag lab03-06-python .

format:
	docker run --rm -v "$(CURDIR):/app" -w /app lab03-06-python ruff format

run_docker:
	docker run --env-file .env --memory=8g --memory-swap=8g --shm-size=4g --cpus=4 --rm -v "$(CURDIR):/app" -w /app -it lab03-06-python /bin/bash

run_jupyter:
	docker run --rm -p 8888:8888 -v "$(CURDIR):/app" -w /app lab03-06-python \
		jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
