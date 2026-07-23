import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy.linalg as la
import numpy as np
na = np.newaxis


def mesh(nx,ny,Lx,Ly):
   i = np.arange(0,nx)[na,:] * np.ones((ny,1), np.int64)
   j = np.arange(0,ny)[:,na] * np.ones((1,nx), np.int64)
   p = np.zeros((2,ny-1,nx-1,3), np.int64)
   q = i+nx*j
   p[:,:,:,0] = q[:-1,:-1]
   p[0,:,:,1] = q[1: ,1: ]
   p[0,:,:,2] = q[1: ,:-1]
   p[1,:,:,1] = q[:-1,1: ]
   p[1,:,:,2] = q[1: ,1: ]
   v = np.concatenate(((Lx/(nx-1)*i)[:,:,na], (Ly/(ny-1)*j)[:,:,na]), axis=2)
   vtx = np.reshape(v, (nx*ny,2))
   elt = np.reshape(p, (2*(nx-1)*(ny-1),3))
   return vtx, elt 

def boundary(nx, ny):
    bottom = np.hstack((np.arange(0,nx-1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(ny-1),nx*ny-1,1)[:,na],
                        np.arange(nx*(ny-1)+1,nx*ny,1)[:,na]))
    left   = np.hstack((np.arange(0,nx*(ny-1),nx)[:,na],
                        np.arange(nx,nx*ny,nx)[:,na]))
    right  = np.hstack((np.arange(nx-1,nx*(ny-1),nx)[:,na],
                        np.arange(2*nx-1,nx*ny,nx)[:,na]))
    return np.vstack((bottom, top, left, right))

def get_area(vtx, elt):
    d = np.size(elt, 1)
    if d == 2:
        e = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        areas = la.norm(e, axis=1)
    else:
        e1 = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        e2 = vtx[elt[:, 2], :] - vtx[elt[:, 0], :]
        areas = 0.5 * np.abs(e1[:,0] * e2[:,1] - e1[:,1] * e2[:,0])
    return areas



def local_mesh(Lx, Ly, Nx, Ny, j, J):
    """
    Construit le maillage local associé au j-ième sous-domaine.

    Le domaine global Ω = (0, Lx) x (0, Ly) est découpé en J bandes horizontales
    non recouvrantes. Le j-ième sous-domaine est
        Ω_j = (0, Lx) x (j*Ly/J, (j+1)*Ly/J).

    Paramètres
    ----------
    Lx, Ly : float
        Dimensions du domaine global.
    Nx, Ny : int
        Nombre de points du maillage global dans les directions x et y.
    j : int
        Indice du sous-domaine (0 ≤ j < J).
    J : int
        Nombre total de sous-domaines.

    Retours
    -------
    vtxj : ndarray de taille (Nv_j, 2)
        Tableau des coordonnées des sommets du maillage local.
    eltj : ndarray de taille (Nt_j, 3)
        Tableau de connectivité des triangles du maillage local.
    """

    Nx, Ny = int(Nx), int(Ny)
    j, J = int(j), int(J)
    
    assert j < J and j >= 0, "L'indice j doit vérifier 0 ≤ j < J"

    Iy = Ny - 1

    # On impose une découpe régulière du maillage global
    assert Iy % J == 0, "Ny - 1 doit être divisible par J"

    # Paramètres du maillage local
    Iy_loc = Iy // J              # Nombre d'intervalles en y par sous-domaine
    Ny_loc = Iy_loc + 1           # Nombre de sommets en y pour le maillage local
    Ly_loc = Ly / J               # Hauteur de chaque sous-domaine

    # Construction du maillage local sur (0, Lx) × (0, Ly_loc)
    vtxj, eltj = mesh(Nx, Ny_loc, Lx, Ly_loc)

    # Translation verticale du maillage local vers sa position physique
    vtxj = vtxj.copy()
    vtxj[:, 1] += j * Ly_loc

    return vtxj, eltj

def local_boundary(Nx, Ny, j, J):
    """
    Construit les arêtes du bord du sous-domaine Ω_j dans une décomposition
    en bandes horizontales du domaine global Ω.

    Paramètres
    ----------
    Nx, Ny : int
        Nombre de sommets du maillage global dans les directions x et y.
    j : int
        Indice du sous-domaine (0 ≤ j < J).
    J : int
        Nombre total de sous-domaines.

    Retours
    -------
    beltj_phys : ndarray (N_phys, 2)
        Arêtes du bord physique ∂Ω_j ∩ ∂Ω, numérotées localement.
    beltj_artf : ndarray (N_artf, 2)
        Arêtes artificielles correspondant aux interfaces entre sous-domaines.
    """

    # --- Vérifications des entrées ---
    Nx, Ny = int(Nx), int(Ny)
    j, J   = int(j), int(J)

    assert j < J and j >= 0, "L'indice j doit vérifier 0 ≤ j < J"

    Iy = Ny - 1
    assert Iy % J == 0, "La décomposition impose que (ny - 1) soit divisible par J"

    # --- Dimensions du maillage local ---
    # Chaque sous-domaine contient Iy/J intervalles verticaux,
    # soit ny_loc = Iy/J + 1 sommets en direction y
    ny_loc = Iy // J + 1

    # --- Arêtes du bord du rectangle local ---
    # Rectangle local : (0, Lx) × (j·Ly/J, (j+1)·Ly/J)
    # La numérotation est locale au sous-domaine
    belt_all = boundary(Nx, ny_loc)

    # --- Nombre d'arêtes par côté ---
    n_bottom = Nx - 1          # segments horizontaux bas
    n_top    = Nx - 1          # segments horizontaux haut
    n_left   = ny_loc - 1      # segments verticaux gauche
    n_right  = ny_loc - 1      # segments verticaux droit

    # --- Découpage des arêtes par côté ---
    bottom = belt_all[0 : n_bottom]
    top    = belt_all[n_bottom : n_bottom + n_top]
    left   = belt_all[n_bottom + n_top :
                      n_bottom + n_top + n_left]
    right  = belt_all[n_bottom + n_top + n_left :
                      n_bottom + n_top + n_left + n_right]

    # --- Construction des bords physique et artificiel ---
    beltj_phys = []
    beltj_artf = []

    # Les bords verticaux sont toujours physiques
    beltj_phys.extend([left, right])

    # Bord inférieur
    if j == 0:
        beltj_phys.append(bottom)   # ∂Ω_j ∩ ∂Ω
    else:
        beltj_artf.append(bottom)   # interface avec Ω_{j-1}

    # Bord supérieur
    if j == J - 1:
        beltj_phys.append(top)      # ∂Ω_j ∩ ∂Ω
    else:
        beltj_artf.append(top)      # interface avec Ω_{j+1}

    # --- Assemblage final ---
    beltj_phys = np.vstack(beltj_phys) if beltj_phys else np.zeros((0, 2), dtype=int)
    beltj_artf = np.vstack(beltj_artf) if beltj_artf else np.zeros((0, 2), dtype=int)

    return beltj_phys, beltj_artf

def plot_edges(vtx, edges, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        return

    segments = vtx[edges]
    lc = LineCollection(segments, **kwargs)
    ax.add_collection(lc)