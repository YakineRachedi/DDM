IMAGE := helmholtz-ddm

.PHONY: all build run-global run-ddm clean

all: build run-global run-ddm

build:
	docker build -t $(IMAGE) .

run-global:
	docker run --rm \
		-v "$(PWD)/benchmarks:/app/benchmarks" \
		$(IMAGE) python -m bin.run_global_solver

run-ddm:
	docker run --rm \
		-v "$(PWD)/benchmarks:/app/benchmarks" \
		$(IMAGE) python -m bin.run_ddm_solver

clean:
	rm -f benchmarks/*.png