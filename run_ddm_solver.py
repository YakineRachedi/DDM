import numpy as np
import matplotlib.pyplot as plt
from mesh.mesh import local_mesh, local_boundary
from fem.local_matrices import Bj_matrix, Cj_matrix
from fem.RHS import point_source
from operators.local_problems import Aj_matrix, Tj_matrix, Sj_factorization, bj_vector
from operators.global_operators import g_vector
from solvers.solvers import gmres_solver
from config import *

# Valeurs de J à tester (nombre de sous-domaines)
J_values = [2, 4, 8]

# Stockage des résidus
resultats_gmres = []

for J in J_values:
    print(f"Calcul pour J = {J} sous-domaines...")

    # Construction des données locales
    locals_ = []
    for j in range(J):
        vtxj, eltj = local_mesh(Lx, Ly, nx, ny, j, J)
        belt_phys, belt_artf = local_boundary(nx, ny, j, J)
        Aj = Aj_matrix(vtxj, eltj, belt_phys, k)
        Bj = Bj_matrix(nx, ny, j, J, belt_artf)
        Tj = Tj_matrix(vtxj, belt_artf, Bj, k)
        Sj, lu = Sj_factorization(Aj, Tj, Bj)
        Cj = Cj_matrix(nx, ny, j, J)
        bj = bj_vector(vtxj, eltj, ps)
        locals_.append({"Bj": Bj, "Tj": Tj, "Cj": Cj, "lu": lu, "bj": bj})

    g = g_vector(locals_, nx, J)

    # Résolution GMRES
    p_gm, res_gmres, info = gmres_solver(
        g, locals_, nx, ny, J,
        eps=1e-10,
        iter_max=200
    )

    print(f"GMRES info = {info}, iterations = {len(res_gmres)}, final residual = {res_gmres[-1] if len(res_gmres) else None}")
    resultats_gmres.append(res_gmres)

# Plot des résidus
plt.figure(figsize=(8,6))
for i, J in enumerate(J_values):
    res = resultats_gmres[i]
    plt.semilogy(res, label=f"GMRES J={J}")

plt.xlabel("Itération")
plt.ylabel(r"$\|r_k\|_2$")
plt.title("Convergence GMRES selon le nombre de sous-domaines (taille de domaine fixe)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig("benchmarks/GMRES_sub_domains.png")
plt.show()
