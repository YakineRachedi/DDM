IMAGE=helmholtz-ddm

build:
	docker build -t $(IMAGE) .

run-global:
	docker run --rm \
	$(IMAGE) python run_global_solver.py

run-ddm:
	docker run --rm \
	$(IMAGE) python run_ddm_solver.py

update:
	git pull
	docker compose build
	docker compose up