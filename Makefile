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



run-evals-retriever-extended:
	uv sync

	#for bash
	cd apps/api/src && PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file ../../../.env python -m evals.eval_retriever_extended

	# Windows PowerShell
	uv sync

	cd apps\api\src

	$env:PYTHONPATH="$PWD;$PWD\..;$PWD\..\.."

	uv run --env-file C:/Users/joshi/Documents/AI-Engineering/.env python -m evals.eval_retriever_extended