import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse.linalg as spla
from mesh.mesh import local_mesh, local_boundary, mesh, boundary
from fem.local_matrices import Bj_matrix, Cj_matrix
from fem.RHS import point_source
from fem.global_matrices import mass, stiffness
from operators.local_problems import Aj_matrix, Tj_matrix, Sj_factorization, bj_vector
from operators.global_operators import g_vector
from solvers.solvers import gmres_solver

# =========================
# Paramètres globaux
# =========================
Lx, Ly = 1.0, 2.0
nx, ny = 33, 65
J = 4
k = 16

# =========================
# Sources
# =========================
ns = 8
sp_list = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]
ps = point_source(sp_list, k)

# =========================
# Données locales (décomposition domaine)
# =========================
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

    locals_.append({
        "Bj": Bj,
        "Tj": Tj,
        "Cj": Cj,
        "lu": lu,
        "bj": bj
    })

# =========================
# Second membre global d’interface
# =========================
g = g_vector(locals_, nx, J)

# =========================
# Résolution avec GMRES sur problème décomposé
# =========================
start_ddm = time.time()
p_gm, res_gmres, info = gmres_solver(
    g, locals_, nx, ny, J,
    eps=1e-10,
    iter_max=200
)
end_ddm = time.time()

print("GMRES décomposé info =", info,
      "| iters =", len(res_gmres),
      "| final res =", res_gmres[-1] if len(res_gmres) else None)
print(f"Temps calcul GMRES décomposé : {end_ddm - start_ddm:.4f} s")

# =========================
# Construction problème global complet
# =========================
vtx, elt = mesh(nx, ny, Lx, Ly)
belt = boundary(nx, ny)
M = mass(vtx, elt)
Mb = mass(vtx, belt)
K = stiffness(vtx, elt)
A = K - k**2 * M - 1j * k * Mb
b = M @ ps(vtx)

# =========================
# Résolution GMRES sur problème complet
# =========================
residuals_full = []

def callback_full(rk):
    residuals_full.append(rk)

start_full = time.time()
x_full, info_full = spla.gmres(A, b, rtol=1e-10, callback=callback_full, callback_type='pr_norm')
end_full = time.time()

print("GMRES complet info =", info_full,
      "| iters =", len(residuals_full),
      "| final res =", residuals_full[-1] if residuals_full else None)
print(f"Temps calcul GMRES complet : {end_full - start_full:.4f} s")

# =========================
# Visualisation convergence comparée
# =========================
plt.figure()
plt.semilogy(res_gmres, label="GMRES décomposé")
plt.semilogy(residuals_full, label="GMRES complet")
plt.xlabel("Itération")
plt.ylabel(r"$\|r_k\|_2$")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.title(f"Convergence GMRES: décomposé vs complet\n"
          f"Lx={Lx}, Ly={Ly}, nx={nx}, ny={ny}, J={J}, k={k}")
plt.savefig("benchmarks/perf.png")
plt.show()
