import numpy as np
import matplotlib.pyplot as plt
from mesh import local_mesh, local_boundary
from fem.local_matrices import Bj_matrix, Cj_matrix
from operators.local_problems import Aj_matrix, Tj_matrix, Sj_factorization, bj_vector
from operators.global_operators import S_operator, Pi_operator, g_vector
from solvers.solvers import fixed_point_solver, gmres_solver
from fem.RHS import point_source

Lx, Ly = 1.0, 2.0
J = 4
k = 16
ns = 8
sp_list = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]
ps = point_source(sp_list, k)

maillages = [
    (17, 33),
    (33, 65),
    (65, 129)
]

resultats_gmres = []

for nx, ny in maillages:
    print(f"Calcul pour maillage {nx}x{ny}...")

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

    p_gm, res_gmres, info = gmres_solver(
        g, locals_, nx, ny, J,
        eps=1e-10,
        iter_max=200
    )
    print(f"GMRES info = {info}, iterations = {len(res_gmres)}, final residual = {res_gmres[-1] if len(res_gmres) else None}")

    resultats_gmres.append(res_gmres)

plt.figure()
for i, (nx, ny) in enumerate(maillages):
    res = resultats_gmres[i]
    plt.semilogy(res, label=f"GMRES {nx}x{ny}")

plt.xlabel("Itération")
plt.ylabel(r"$\|r_k\|_2$")
plt.title("Convergence GMRES selon la taille du maillage")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.savefig("Refine_Mesh.png")
plt.show()
