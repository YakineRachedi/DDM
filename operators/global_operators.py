import numpy as np

def S_operator(x, locals_, J):
    """
    Calcule l'action de l'opérateur S sur un vecteur global x des inconnues
    d'interface.

    Args:
        x (np.ndarray): vecteur global des inconnues sur toutes les interfaces.
        locals_ (list of dict): liste des données locales par sous-domaine j,
                                chaque dict doit contenir les matrices et facteurs:
                                "Bj", "Tj", "lu", "Cj".
        nx (int): dimensions globales du maillage selon l'axe des x.
        J (int): nombre de sous-domaines.

    Returns:
        y (np.ndarray): résultat de S(x), vecteur global même taille que x.
    """
    x = np.asarray(x, dtype=complex)

    # Calcul des tailles des blocs et offsets pour extraire x_j
    #sizes = np.array([n_sigma(nx, j, J) for j in range(J)], dtype=int)
    #offsets = np.zeros(J+1, dtype=int)
    #offsets[1:] = np.cumsum(sizes)

    y = np.zeros_like(x, dtype=complex)

    for j in range(J):
        Bj = locals_[j]["Bj"]       # Matrice de restriction locale (interface -> volume local)
        Tj = locals_[j]["Tj"]       # Matrice Tj locale
        lu = locals_[j]["lu"]       # Factorisation LU de Aj
        Cj = locals_[j]["Cj"]       # Matrice de restriction globale -> locale

        # Extraction du sous-vecteur x_j correspondant à Σ_j
        xj = Cj @ x

        # Calcul du second membre local : r_j = B_j^T (T_j x_j)
        rj = Bj.T @ (Tj @ xj)

        # Résolution locale : u_j = Aj^{-1} r_j via la factorisation LU
        uj = lu.solve(np.asarray(rj, dtype=complex))

        # Calcul de la contribution locale sur l'interface : y_j = x_j + 2i * B_j u_j
        yj = xj + 2j * (Bj @ uj)

        # Assemblage dans le vecteur global y via Cj^T (transpose de la restriction)
        y += Cj.T @ yj

    return y


def Pi_operator(x, Nx, J):
    """
    Application de l'opérateur Π sur le vecteur global x des inconnues d'interface.
    Cet opérateur échange les valeurs entre sous-domaines voisins au niveau des interfaces.

    Args:
        x (array_like): vecteur global des inconnues d'interface
        nx (int): nombre de points en x dans le maillage local (doit être >= 3)
        J (int): nombre de sous-domaines en découpage vertical

    Returns:
        y (ndarray): résultat de Π x, même dimension que x
    """

    x = np.asarray(x)
    m = Nx - 2  # nombre de degrés de liberté par interface (coins exclus)

    # tailles des blocs d'interface pour chaque sous-domaine
    # 1er et dernier sous-domaine ont une interface de taille m,
    # les autres ont deux interfaces (top et bottom) de taille m chacune, soit 2*m
    sizes = [m] + [2*m]*(J-2) + [m]
    offsets = np.zeros(J+1, dtype=int)
    offsets[1:] = np.cumsum(sizes)

    y = np.empty_like(x)

    # Traitement du premier sous-domaine (j=0)
    # il ne reçoit que la partie "bottom" du sous-domaine j=1 (qui est son interface commune)
    y[offsets[0]:offsets[1]] = x[offsets[1]:offsets[1]+m]

    # Traitement des sous-domaines intérieurs (1 <= j <= J-2)
    for j in range(1, J-1):
        # Partie "bottom" de j reçoit la partie "top" de j-1
        if j-1 == 0:
            # j-1=0 a une seule interface (taille m)
            y[offsets[j]:offsets[j]+m] = x[offsets[j-1]:offsets[j-1]+m]
        else:
            # j-1 > 0 a deux interfaces (bottom et top)
            y[offsets[j]:offsets[j]+m] = x[offsets[j-1]+m:offsets[j-1]+2*m]

        # Partie "top" de j reçoit la partie "bottom" de j+1
        if j+1 == J-1:
            # j+1 dernier sous-domaine, une interface de taille m
            y[offsets[j]+m:offsets[j]+2*m] = x[offsets[j+1]:offsets[j+1]+m]
        else:
            # j+1 sous-domaine intérieur, deux interfaces
            y[offsets[j]+m:offsets[j]+2*m] = x[offsets[j+1]:offsets[j+1]+m]

    # Traitement du dernier sous-domaine (j=J-1)
    # il reçoit la partie "top" du sous-domaine j=J-2
    y[offsets[J-1]:offsets[J]] = x[offsets[J-2]+m:offsets[J-2]+2*m]

    return y
def g_vector(locals_, Nx, J):
    """
    Construction du second membre global g du problème d'interface.

    Arguments :
    - locals_ : liste de dictionnaires, un par sous-domaine,
                contenant les clés "bj", "Bj", "Cj", "lu"
    - nx, J : paramètres du maillage et nombre de sous-domaines
    - Pi_operator : fonction qui applique l'opérateur Pi sur un vecteur global

    Retour :
    - g : vecteur numpy complexe, second membre global sur les interfaces
    """

    # On récupère la taille totale du vecteur interface depuis Cj (toutes Cj ont même nb colonnes)
    n_total = locals_[0]["Cj"].shape[1]

    t = np.zeros(n_total, dtype=complex)

    for j in range(J):
        Bj = locals_[j]["Bj"]      # (n_sigma_j, n_loc)
        Cj = locals_[j]["Cj"]      # (n_sigma_j, n_sigma_tot)
        lu = locals_[j]["lu"]      # factorisation de S_j
        bj = locals_[j]["bj"]      # (n_loc,)


        # Résolution locale u_j = S_j^{-1} r_j
        uj = lu.solve(bj)          # (n_loc,)

        # Calcul contribution locale t_j = B_j u_j (interface locale)
        tj = Bj @ uj               # (n_sigma_j,)

        # Assemblage global dans g via C_j^T (projection locale vers global)
        t += Cj.T @ tj             # (n_sigma_tot,)

    # 5) Application de l'opérateur Pi pour échanger l'information entre interfaces
    g = -2j * Pi_operator(t, Nx, J)

    return g
