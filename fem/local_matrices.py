import numpy as np
from scipy.sparse import csr_matrix

def Rj_matrix(Nx, Ny, j, J):
    """
    Construit la matrice de restriction R_j telle que u_j = R_j u.

    Paramètres
    ----------
    Nx, Ny : int
        Nombre de points du maillage global en x et y.
    j : int
        Indice du sous-domaine (0 <= j < J).
    J : int
        Nombre total de sous-domaines.

    Retour
    ------
    Rj : scipy.sparse.csr_matrix
        Matrice de restriction de taille (N_j, N).
    """
    Nx, Ny = int(Nx), int(Ny)
    j, J = int(j), int(J)

    assert 0 <= j < J, "j doit vérifier 0 ≤ j < J"
    Iy = Ny - 1
    assert Iy % J == 0, "Ny - 1 doit être divisible par J"

    ny_loc = Iy // J + 1
    y_start = j * (Iy // J)
    y_end = y_start + ny_loc

    N = Nx * Ny
    Nj = Nx * ny_loc

    local_idx = np.arange(Nj)
    jy = np.arange(y_start, y_end)
    ix_grid, jy_grid = np.meshgrid(np.arange(Nx), jy)
    global_idx = (ix_grid + Nx * jy_grid).ravel()

    rows = local_idx
    cols = global_idx
    data = np.ones_like(rows, dtype=float)

    Rj = csr_matrix((data, (rows, cols)), shape=(Nj, N))

    return Rj

def Bj_matrix(Nx, Ny, j, J, beltj_artf):
    """
    Construit la matrice de restriction locale Bj associée aux degrés de liberté
    situés sur l'interface artificielle du sous-domaine j.

    Arguments :
    - nx, ny : int, nombre de points en x et y du maillage global
    - j : int, indice du sous-domaine courant (0 <= j < J)
    - J : int, nombre total de sous-domaines
    - beltj_artf : tableau numpy d'arêtes artificielles (interfaces) du sous-domaine j,
                   chaque arête est donnée par un couple d'indices de sommets locaux

    Retour :
    - Bj : matrice creuse CSR de forme (nombre_dll_interface, taille_sous_domaine_local)
    """

    Nx, Ny, j, J = int(Nx), int(Ny), int(j), int(J)
    assert 0 <= j < J, "j doit vérifier 0 ≤ j < J"
    Iy = Ny - 1
    assert Iy % J == 0, "Ny - 1 doit être divisible par J"

    # Calcul de la taille locale du sous-domaine
    nb_intervalles_y = (Ny - 1) // J
    ny_local = nb_intervalles_y + 1
    taille_locale = Nx * ny_local

    beltj_artf = np.array(beltj_artf, dtype=int)

    # Cas sans interface artificielle
    if beltj_artf.size == 0: return csr_matrix((0, taille_locale), dtype=np.int8), np.array([], dtype=int)

    # Extraction des sommets uniques présents sur l'interface artificielle
    sommets_interface = np.unique(beltj_artf.ravel())

    # Exclusion des coins du sous-domaine local
    coins = np.array([0, Nx - 1, taille_locale - Nx, taille_locale - 1], dtype=int)
    sommets_interface = np.setdiff1d(sommets_interface, coins)

    # Construction des données pour la matrice creuse
    nb_sommets = sommets_interface.size
    lignes = np.arange(nb_sommets)
    colonnes = sommets_interface
    valeurs = np.ones(nb_sommets, dtype=np.int8)

    Bj = csr_matrix((valeurs, (lignes, colonnes)), shape=(nb_sommets, taille_locale))

    return Bj

def n_sigma(k, m, J):
    if k == 0 or k == J - 1:
        return m
    else:
        return 2 * m

def Cj_matrix(nx, ny, j, J):
    nx = int(nx)
    ny = int(ny)
    j = int(j)
    J = int(J)

    assert 0 <= j < J, "Indice j hors bornes"
    assert (ny - 1) % J == 0, "Le nombre d'intervalles en y doit être divisible par J"

    m = nx - 2 # exclus les coins

    start = sum(n_sigma(k, m, J) for k in range(j))
    n_sig_j = n_sigma(j, m, J)
    n_sig_tot = sum(n_sigma(k, m, J) for k in range(J))

    rows = np.arange(n_sig_j)
    cols = start + rows
    data = np.ones(n_sig_j, dtype=np.int8)

    Cj = csr_matrix((data, (rows, cols)), shape=(n_sig_j, n_sig_tot))

    return Cj
