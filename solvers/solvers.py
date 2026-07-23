import numpy as np
import scipy.sparse.linalg as spla
from operators.global_operators import S_operator, Pi_operator

def fixed_point_solver(g, locals_, nx, ny, J, omega=0.5, tol=1e-10, maxiter=200, p0=None):
    """
    Résout (I + ΠS)p = g par point fixe relaxé.
    Itération: p <- (1-ω)p + ω*(g - ΠS(p))

    Arguments:
    - g: vecteur second membre global (complexe)
    - locals_: liste des données locales (dictionnaires contenant Bj, Cj, lu, ...)
    - nx, ny, J: paramètres du maillage et sous-domaines
    - omega: relaxation (0 < omega < 1)
    - tol: tolérance de convergence sur le résidu
    - maxiter: nombre max d'itérations
    - p0: condition initiale (optionnelle)

    Retour:
    - p: solution approchée
    - res_hist: historique des résidus (normes des résidus)
    """

    n = g.size
    p = np.zeros(n, dtype=complex) if p0 is None else np.array(p0, dtype=complex, copy=True)

    res_hist = []
    it = 0

    while it < maxiter:
        # Calcul de S(p)
        Sp = S_operator(p, locals_, J)
        # Calcul de Pi(S(p))
        PiSp = Pi_operator(Sp, nx, J)
        # Application de A(p) = p + Pi(S(p))
        Ap = p + PiSp

        # Résidu r = g - A(p)
        r = g - Ap
        res = np.linalg.norm(r)
        res_hist.append(res)

        if res <= tol:break

        # Calcul nouvelle itération p_new = g - Pi(S(p))
        p_new = g - PiSp

        # Relaxation
        p = (1 - omega) * p + omega * p_new

        it += 1

    if res > tol:
        print(f"Warning: Fixed point did not converge after {maxiter} iterations, final residual = {res:.2e}")

    return p, np.array(res_hist)

def gmres_solver(g, locals_, Nx, Ny, J, eps=1e-10, iter_max=100, x0=None):
    """
    Résout le système (I + ΠS) p = g par la méthode GMRES.

    Arguments :
    - g : np.array (complexe), second membre global
    - locals_ : liste de dictionnaires, données locales par sous-domaine
    - nx, ny : int, paramètres du maillage
    - J : int, nombre de sous-domaines
    - tol : float, tolérance sur le résidu relatif
    - maxiter : int, nombre maximal d'itérations
    - x0 : np.array, condition initiale optionnelle

    Retourne :
    - p : np.array, solution approchée complexe
    - residuals : np.array, historique des normes des résidus à chaque itération
    - info : int, indicateur de convergence (0 si succès)
    """

    n = g.size  
    residuals = []

    # Opérateur linéaire matvec : x ↦ (I + Π S) x
    matvec = lambda x: x + Pi_operator(S_operator(x, locals_, J), Nx, J)

    # Création de l'opérateur linéaire compatible avec GMRES
    Aop = spla.LinearOperator(shape=(n, n), matvec=matvec, dtype=np.complex128)

    # Callback pour stocker la norme du résidu à chaque itération
    callback = lambda resnorm: residuals.append(resnorm)

    p, info = spla.gmres(
        Aop, g,
        x0=x0,
        rtol=eps,
        atol=0.0,
        maxiter=iter_max,
        callback=callback,
        callback_type='pr_norm'  # résidu préconditionné (ici simple résidu)
    )

    residuals = np.array(residuals)
    return p, residuals, info


def uj_solution(x, locals_, J):
    """
    Reconstruit les solutions locales u_j à partir de la solution
    globale d'interface x.

    Paramètres
    ----------
    x : ndarray (n_sigma_tot,)
        Solution du problème d'interface.
    locals_ : list of dict
        Données locales par sous-domaine (Bj, Tj, Cj, bj, lu).
    J : int
        Nombre de sous-domaines.

    Retour
    ------
    u_list : list of ndarray
        Liste des solutions locales u_j (une par sous-domaine).
    """

    x = np.asarray(x, dtype=complex)
    u_list = []

    for j in range(J):
        Bj = locals_[j]["Bj"]
        Tj = locals_[j]["Tj"]
        Cj = locals_[j]["Cj"]
        bj = locals_[j]["bj"]
        lu = locals_[j]["lu"]

        # restriction de x sur l'interface locale
        xj = Cj @ x

        # second membre local
        rhs = bj + Bj.conj().T @ (Tj @ xj)

        # résolution locale
        uj = lu.solve(rhs)

        u_list.append(uj)

    return u_list
