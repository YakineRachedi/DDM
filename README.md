# Finite Element Solver for the 2D Helmholtz Equation using Domain Decomposition Methods

## Description

Ce projet porte sur la résolution numérique de l'équation de Helmholtz en dimension deux par la méthode des éléments finis (FEM), avec une étude de méthodes itératives et de décomposition de domaines.

L'objectif principal est de développer un solveur capable de traiter des problèmes d'ondes avec une solution oscillante, tout en étudiant des stratégies permettant d'améliorer la résolution de systèmes linéaires de grande taille.

Le problème étudié est :

\[
-\Delta u - \kappa^2 u = f \quad \text{dans } \Omega
\]

avec une condition aux limites de rayonnement :

\[
\partial_n u - i\kappa u = 0 \quad \text{sur } \partial\Omega
\]

où :

- $\Omega$ est un domaine rectangulaire ;
- $\kappa$ est le nombre d'onde ;
- $\partial_n$ est la dérivée normale sortante ;
- $f$ représente une excitation composée de sources ponctuelles régularisées.

Le terme source est défini par :

\[
f(x)=\sum_{i=1}^{N_s} w_i
\exp\left(-\frac{10}{\lambda^2}|x-s_i|^2\right)
\]

avec :

- $s_i$ la position des sources ;
- $w_i$ leurs poids complexes ;
- $\lambda=2\pi/\kappa$ la longueur d'onde.

---

# Méthode numérique

## Discrétisation éléments finis

Le problème est discrétisé par une méthode des éléments finis conformes de type :

- éléments triangulaires ;
- éléments finis $P_1$ de Lagrange ;
- maillage uniforme du domaine.

La formulation variationnelle consiste à chercher :

\[
u \in H^1(\Omega)
\]

tel que :

\[
a(u,v)=\ell(v), \quad \forall v\in H^1(\Omega)
\]

avec :

\[
a(u,v)=
\int_\Omega
(\nabla u \cdot \nabla v-\kappa^2uv)
-
i\kappa\int_{\Gamma}uv
\]

La discrétisation conduit à un système linéaire :

\[
Au=b
\]

où :

- $A$ est une matrice creuse complexe ;
- le système est indéfini à cause du terme $-\kappa^2u$ ;
- la résolution devient difficile lorsque le maillage est raffiné.

Pour assurer une bonne résolution des oscillations, la taille du maillage doit respecter :

\[
h \lesssim \frac{\lambda}{10}
\]

---

# Approches étudiées

Deux approches de résolution sont implémentées.

---

## 1. Résolution globale avec GMRES

La première approche consiste à assembler directement le système global :

\[
Au=b
\]

puis à résoudre ce système avec :

- GMRES ;
- méthode de point fixe pour certains opérateurs.

Cette approche sert de référence pour comparer les performances.

Fichier associé : solve_global.py


---

## 2. Décomposition de domaines + GMRES

La seconde approche utilise une méthode de décomposition de domaines non recouvrante.

Le domaine $\Omega$ est séparé en plusieurs sous-domaines :

+------------+
| Domain 1 |
+------------+
| Domain 2 |
+------------+
| Domain 3 |
+------------+


Pour chaque sous-domaine :

1. construction du maillage local ;
2. assemblage des matrices locales ;
3. résolution du problème local ;
4. construction d'un opérateur global de transmission ;
5. résolution du problème global avec GMRES.

Cette approche permet d'exploiter l'indépendance entre les sous-domaines et constitue une base naturelle pour une future parallélisation.

Fichier associé : solve_ddm.py


---

# Organisation du projet

DDM/
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── solve_global.py
├── solve_ddm.py
├── config.py
├── perf.py
│
├── mesh/
│ ├── mesh.py
│ ├── refine_mesh.py
│ ├── plot_mesh.py
│ └── dd_plot.py
│
├── fem/
│ ├── local_matrices.py
│ ├── global_matrices.py
│ └── RHS.py
│
├── operators/
│ ├── local_problems.py
│ └── global_operators.py
│
├── solvers/
│ ├── solvers.py
│ └── GMRES_sub_domaines.py
│
├── benchmarks/
│
└── docs/
    └── Rapport.pdf

## Docker

Le projet fournit une image Docker contenant toutes les dépendances Python nécessaires à l'exécution des solveurs.

### Construction de l'image

Depuis la racine du projet pour construire l'image helmholtz-ddm : 

```bash
make build
```
Exécution du solveur global dans le conteneur Docker:

```bash
make run-global
```
Exécution du solveur par décomposition de domaines dans le conteneur Docker:
```bash
make run-ddm
```

---

# Description des modules

## mesh/

Gestion du maillage :

- génération du domaine triangulaire ;
- raffinement ;
- extraction des frontières ;
- visualisation des sous-domaines.

---

## fem/

Implémentation des opérateurs éléments finis :

- matrices locales ;
- assemblage global ;
- second membre associé aux sources ponctuelles.

---

## operators/

Construction des opérateurs nécessaires à la résolution :

- problèmes locaux sur chaque sous-domaine ;
- opérateurs de transmission ;
- opérateur global utilisé par GMRES.

---

## solvers/

Contient les solveurs numériques :

- GMRES classique ;
- GMRES appliqué à la décomposition de domaines ;
- méthodes itératives complémentaires.

---

# Résultats

Les expériences numériques sont détaillées dans :
docs/Rapport.pdf


Le rapport présente notamment :

- l'influence du raffinement du maillage ;
- la convergence de GMRES ;
- la comparaison entre résolution globale et DDM ;
- l'impact du nombre de sous-domaines ;
- l'analyse des performances numériques.

---

# Utilisation

Résolution classique : python solve_global.py
Résolution par décomposition de domaines : python solve_ddm.py

# Perspectives HPC : 

La méthode de décomposition de domaines est particulièrement adaptée aux architectures parallèles.

1. Parallélisation MPI

Chaque sous-domaine peut être associé à un processus MPI indépendant :

MPI Rank 0        MPI Rank 1        MPI Rank 2

Domain 0          Domain 1          Domain 2

Local FEM         Local FEM         Local FEM

      \              |              /

        Communication interfaces

              GMRES global

Les calculs pouvant être parallélisés :

génération des maillages locaux ;
assemblage FEM local ;
factorisations locales ;
résolutions locales.

Une implémentation possible :

mpi4py pour une version Python ;
PETSc/petsc4py pour une version HPC plus avancée.

2. Architecture hybride MPI + GPU

Une évolution possible serait un solveur hybride :

        Cluster HPC

+-----------------------+
| Node 1                |
| GPU + MPI process     |
| Domain 1              |
+-----------------------+

+-----------------------+
| Node 2                |
| GPU + MPI process     |
| Domain 2              |
+-----------------------+

          MPI

       GMRES global

Chaque nœud traiterait un ensemble de sous-domaines avec accélération GPU locale.

# Améliorations possibles
- ajout de préconditionneurs Schwarz ;
- étude de méthodes additives/multiplicatives ;
- résolution sur des maillages beaucoup plus grands ;
- comparaison avec PETSc ;
- parallélisation MPI ;
- accélération GPU ;
- benchmark sur architectures HPC.