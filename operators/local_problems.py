import scipy.sparse as sps
from scipy.sparse.linalg import splu
from fem.global_matrices import mass, stiffness

def Aj_matrix(vtxj, eltj, beltj_phys, k):
    """
    Construction de la matrice locale Aj pour le sous-domaine j,
    correspondant à la discrétisation de l'opérateur Helmholtz avec condition
    aux limites sur le bord physique.

    Arguments :
    - vtxj : ndarray (n_loc x 2), coordonnées des sommets du sous-domaine local j
    - eltj : ndarray (n_triangles x 3), connectivité des triangles locaux
    - beltj_phys : ndarray (n_arêtes x 2), arêtes correspondant au bord physique local
    - k : float, nombre d'onde (wavenumber)

    Retour :
    - Aj : matrice creuse (csr_matrix), matrice locale complexe du système linéaire
    """

    # Matrice de masse locale (P1 éléments finis)
    Mj = mass(vtxj, eltj)

    # Matrice de rigidité locale
    Kj = stiffness(vtxj, eltj)

    # Matrice de masse sur le bord physique (support sur les arêtes du bord)
    Mbj = mass(vtxj, beltj_phys)

    # Assemblage final de la matrice locale pour Helmholtz avec terme de condition aux limites absorbantes
    Aj = Kj - (k ** 2) * Mj - 1j * k * Mbj

    return Aj


def Tj_matrix(vtxj, beltj_artf, Bj, k):
    """
    Construction de la matrice locale T_j associée à l'interface artificielle Σ_j.

    Arguments :
    -----------
    vtxj : ndarray (n_loc, 2)
        Coordonnées des sommets du sous-domaine local j.
    beltj_artf : ndarray (m, 2)
        Arêtes correspondant à l'interface artificielle Σ_j.
    Bj : csr_matrix (n_sigma, n_loc)
        Matrice de restriction locale sur l'interface artificielle.
    k : float
        Nombre d'onde (wavenumber) du problème.

    Retour :
    --------
    Tj : csr_matrix (n_sigma, n_sigma)
        Matrice locale associée à l'interface artificielle Σ_j.
    """

    # Calcul de la matrice de masse restreinte à l'interface artificielle
    Mb_artf = mass(vtxj, beltj_artf)  # Matrice creuse (n_loc x n_loc)

    # Construction de la matrice Tj par double restriction sur l'interface
    Tj = k * (Bj @ Mb_artf @ Bj.T)  # Produit matriciel creux

    return Tj


def Sj_factorization(Aj, Tj, Bj):
    """
    Calcule la factorisation locale S_j utilisée dans la méthode de décomposition de domaine.

    Arguments :
    -----------
    Aj : csr_matrix (n_loc, n_loc)
        Matrice locale Aj du sous-domaine j.
    Tj : csr_matrix (n_sigma, n_sigma)
        Matrice Tj associée à l'interface artificielle Σ_j.
    Bj : csr_matrix (n_sigma, n_loc)
        Matrice de restriction locale sur l'interface artificielle.

    Retour :
    --------
    Sj : csr_matrix (n_loc, n_loc)
        Matrice S_j factorisée localement.
    """

    # Calcul de la matrice S_j selon la formule du cours
    Sj = Aj - 1j * Bj.T @ (Tj @ Bj)
    lu = splu(Sj.tocsc())

    return Sj, lu


def bj_vector(vtxj, eltj, src_p):
    """
    Construit le vecteur local du second membre b_j.

    Arguments :
    -----------
    vtxj : ndarray (n_loc, 2)
        Coordonnées des sommets locaux du sous-domaine j.
    eltj : ndarray (n_elem, 3)
        Connectivité des éléments finis locaux (triangles).
    src_p : callable
        Fonction représentant la source ponctuelle régularisée, évaluée sur les sommets.

    Retour :
    --------
    bj : ndarray (n_loc,)
        Vecteur local second membre associé.
    """
    Mj = mass(vtxj, eltj)
    fj = src_p(vtxj)
    bj = Mj @ fj
    return bj
