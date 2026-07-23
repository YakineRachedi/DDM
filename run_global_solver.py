#! /usr/bin/python3

import numpy as np

from mesh.mesh import local_mesh, local_boundary
from mesh.plot_mesh import plot_mesh
from fem.local_matrices import Bj_matrix, Cj_matrix
from operators.local_problems import Aj_matrix, Tj_matrix, Sj_factorization, bj_vector
from operators.global_operators import g_vector
from solvers.solvers import fixed_point_solver, gmres_solver, uj_solution
import matplotlib.pyplot as plt
from config import *

if __name__ == "__main__":
    locals_ = []

    for j in range(J):
        # Maillage local
        vtxj, eltj = local_mesh(Lx, Ly, nx, ny, j, J)

        # Frontières locales
        belt_phys, belt_artf = local_boundary(nx, ny, j, J)

        # Problème local Helmholtz
        Aj = Aj_matrix(vtxj, eltj, belt_phys, k)

        # Matrices d’interface
        Bj = Bj_matrix(nx, ny, j, J, belt_artf)
        Tj = Tj_matrix(vtxj, belt_artf, Bj, k)

        # Schur local + LU
        Sj, lu = Sj_factorization(Aj, Tj, Bj)

        # Projection interface globale
        Cj = Cj_matrix(nx, ny, j, J)

        # Second membre local
        bj = bj_vector(vtxj, eltj, ps)

        # Stockage STRICTEMENT nécessaire
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
    # Solveur point fixe relaxé
    # =========================
    p_fp, res_fp = fixed_point_solver(
        g, locals_, nx, ny, J,
        omega=0.5,
        tol=1e-10,
        maxiter=200
    )

    # =========================
    # Solveur GMRES
    # =========================
    p_gm, res_gmres, info = gmres_solver(
        g, locals_, nx, ny, J,
        eps=1e-10,
        iter_max=200
    )

    print(
        "GMRES info =", info,
        "| iters =", len(res_gmres),
        "| final res =", res_gmres[-1] if len(res_gmres) else None
    )

    # =========================
    # Visualisation convergence
    # =========================
    plt.figure()
    plt.semilogy(res_fp, label="Point fixe relaxé")
    plt.semilogy(res_gmres, label="GMRES")
    plt.xlabel("Itération")
    plt.ylabel(r"$\|r_k\|_2$")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.savefig("benchmarks/GMRES_vs_PF.png")
    plt.show()

    # Reconstruction des solutions locales
    u_list = uj_solution(p_gm, locals_, J)

    plot_mode = "abs"  # ou "real"

    vals = []
    for uj in u_list:
        if plot_mode == "abs":
            vals.append(np.abs(uj))
        elif plot_mode == "real":
            vals.append(np.real(uj))
        else:
            raise ValueError("plot_mode must be 'abs' or 'real'")


    # Bornes communes
    vmin = min(v.min() for v in vals)
    vmax = max(v.max() for v in vals)

    # Tracé superposé
    plt.figure()
    for j, val in enumerate(vals):
        vtxj, eltj = local_mesh(Lx, Ly, nx, ny, j, J)
        plot_mesh(vtxj, eltj, val=val, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Solutions locales superposées ({plot_mode})")
    plt.savefig("benchmarks/Sol_local_spp.png")
    plt.show()
