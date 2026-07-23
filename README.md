# Finite Element Solver for the 2D Helmholtz Equation using Domain Decomposition Methods

## Overview

This project focuses on the numerical resolution of the two-dimensional Helmholtz equation using the Finite Element Method (FEM).

The objective is to develop and study iterative solvers for large sparse complex linear systems arising from wave propagation problems, with a particular focus on **Domain Decomposition Methods (DDM)** and their potential for high-performance computing.

Two approaches are implemented:

- a global FEM solver using GMRES;
- a non-overlapping Domain Decomposition solver combined with GMRES.

The project also explores possible extensions toward parallel computing architectures such as MPI and GPU acceleration.

A detailed numerical study and analysis of the obtained results are available in:

```
docs/Rapport.pdf
```

---

# Mathematical problem

We consider the Helmholtz equation on a rectangular domain:

$$
\Omega = (0,L_x)\times(0,L_y)
$$

The problem is:

$$
-\Delta u-\kappa^2 u=f \quad \text{in } \Omega
$$

with an absorbing boundary condition:

$$
\partial_n u-i\kappa u=0
\quad \text{on } \partial\Omega
$$


where:

- $\kappa$ is the wave number;
- $\partial_n$ is the outward normal derivative;
- $f$ represents a set of localized sources.

The source term is modeled as a superposition of regularized point sources:

$$
f(x)=
\sum_{i=1}^{N_s}
w_i
\exp
\left(
-\frac{10}{\lambda^2}|x-s_i|^2
\right)
$$


with:

- $s_i$ : source position;
- $w_i$ : complex source weight;
- $\lambda=2\pi/\kappa$ : wavelength.


The absorbing boundary condition approximates outgoing wave radiation.

---

# Numerical method

## Finite Element discretization

The problem is discretized using:

- conforming finite elements;
- triangular mesh;
- first-order Lagrange elements ($P_1$).


The weak formulation leads to the linear system:

$$
Au=b
$$


where:

- $A$ is a sparse complex matrix;
- the system is indefinite due to the Helmholtz term;
- solving the system becomes increasingly expensive for refined meshes.


To correctly represent oscillations, the mesh size must satisfy approximately:

$$
h \lesssim \frac{\lambda}{10}
$$


---

# Implemented solvers

## 1. Global FEM solver + GMRES

The first approach consists in assembling the complete finite element system:

$$
Au=b
$$


and solving it using iterative methods:

- GMRES;
- fixed-point iterations for some operators.


This solver is used as a reference implementation.

Execution:

```bash
make run-global
```

or:

```bash
python run_global_solver.py
```

---

## 2. Domain Decomposition Method + GMRES

The second approach uses a non-overlapping domain decomposition strategy.

The computational domain is split into several independent subdomains:

```text
+----------------+
|   Domain 1     |
+----------------+
|   Domain 2     |
+----------------+
|   Domain 3     |
+----------------+
```


For each subdomain:

1. generate a local mesh;
2. assemble local FEM matrices;
3. solve local problems;
4. construct transmission operators;
5. solve the global interface problem using GMRES.


This approach introduces natural parallelism because local computations are independent.

Execution:

```bash
make run-ddm
```

or:

```bash
python run_ddm_solver.py
```

---

# Project structure

```text
DDM/
│
├── Dockerfile
├── Makefile
├── requirements.txt
├── config.py
│
├── run_global_solver.py
├── run_ddm_solver.py
│
├── mesh/
│   ├── mesh.py
│   ├── refine_mesh.py
│   ├── plot_mesh.py
│   └── dd_plot.py
│
├── fem/
│   ├── local_matrices.py
│   ├── global_matrices.py
│   └── RHS.py
│
├── operators/
│   ├── local_problems.py
│   └── global_operators.py
│
├── solvers/
│   ├── solvers.py
│   └── GMRES_sub_domaines.py
│
├── benchmarks/
│
└── docs/
    └── Rapport.pdf
```

---

# Module description

## mesh/

Contains all mesh-related operations:

- rectangular mesh generation;
- mesh refinement;
- boundary extraction;
- visualization of subdomains.

---

## fem/

Finite element implementation:

- local element matrices;
- global matrix assembly;
- right-hand-side construction;
- source term generation.

---

## operators/

Construction of operators required by the domain decomposition method:

- local subdomain problems;
- local factorizations;
- transmission operators;
- global interface operator.

---

## solvers/

Contains numerical solvers:

- classical GMRES;
- GMRES applied to domain decomposition;
- iterative procedures.

---

# Installation

## Python environment

Install dependencies:

```bash
pip install -r requirements.txt
```


Main dependencies:

- numpy;
- scipy;
- matplotlib.

---

# Docker support

The project provides a Docker environment containing all required dependencies.

## Build image

From the project root:

```bash
make build
```


This creates the Docker image:

```
helmholtz-ddm
```


## Run global solver

```bash
make run-global
```


## Run domain decomposition solver

```bash
make run-ddm
```


Docker ensures reproducible execution independently of the local Python environment.

---

# Results

The numerical experiments are described in:

```
docs/Rapport.pdf
```


The report contains:

- mesh refinement studies;
- GMRES convergence analysis;
- comparison between global resolution and DDM;
- influence of the number of subdomains;
- numerical performance analysis.

---

# Performance and HPC perspectives

Domain decomposition methods are naturally suited for parallel computing.

## MPI parallelization

Each subdomain can be assigned to an MPI process:


```text
MPI rank 0       MPI rank 1       MPI rank 2

Domain 0         Domain 1         Domain 2

Local FEM        Local FEM        Local FEM

      \             |             /

          Interface communication

                 GMRES
```


The following operations can be distributed:

- local mesh generation;
- FEM assembly;
- local matrix factorization;
- local solves.


Possible implementations:

- `mpi4py` for a Python implementation;
- `petsc4py` / PETSc for a more scalable HPC solver.

---

## GPU acceleration

Local finite element computations and linear algebra operations could also benefit from GPU acceleration.

A possible future architecture:


```text
        HPC Cluster

+----------------------+
| Node 1               |
| MPI + GPU            |
| Subdomain 1          |
+----------------------+

+----------------------+
| Node 2               |
| MPI + GPU            |
| Subdomain 2          |
+----------------------+


          MPI

     Global GMRES solver
```


Potential GPU targets:

- sparse matrix operations;
- local factorizations;
- matrix-vector products;
- preconditioners.

---

# Future improvements

Possible extensions:

- implement additive/multiplicative Schwarz preconditioners;
- compare different domain decomposition strategies;
- integrate PETSc;
- solve larger scale problems;
- add MPI parallel execution;
- explore GPU acceleration;
- benchmark on HPC clusters.

---

# Author

Project developed as a numerical simulation and scientific computing project.

Main topics:

- Finite Element Methods;
- Helmholtz equation;
- Iterative solvers;
- Domain Decomposition;
- High Performance Computing.