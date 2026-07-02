run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals-retriever:
	uv sync
#for bash
#PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever

# Windows PowerShell
	$env:PYTHONPATH="$PWD\apps\api;$PWD\apps\api\src;$PWD"
	uv run --env-file .env python -m evals.eval_retriever