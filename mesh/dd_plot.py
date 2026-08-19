from mesh import local_mesh, local_boundary, plot_edges
from plot_mesh import plot_mesh
import matplotlib.pyplot as plt

Lx, Ly = 1.0, 1.0
nx, ny = 6, 7
J = 3

colors = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots(figsize=(6, 6))

for j in range(J):
    vtxj, eltj = local_mesh(Lx, Ly, nx, ny, j, J)
    phys, artf = local_boundary(nx, ny, j, J)

    # Maillage du sous-domaine j
    plot_mesh(
        vtxj, eltj,
        linewidth=0.6,
        color=colors[j]
    )

    # Bords physiques
    plot_edges(
        vtxj, phys,
        ax=ax,
        color=colors[j],
        linewidth=3
    )

    # Interfaces artificielles
    plot_edges(
        vtxj, artf,
        ax=ax,
        color=colors[j],
        linewidth=3,
        linestyle="--"
    )

    print(f"Sous-domaine {j} : {vtxj.shape[0]} points, {eltj.shape[0]} triangles")

ax.set_title("Décomposition du domaine en sous-domaines")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("benchmarks/DD.png")
