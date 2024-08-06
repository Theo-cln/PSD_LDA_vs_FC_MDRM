import numpy as np
from numpy import linalg as la

"""
differents methods to transform matrices into symmetric positive definite matrices
"""



""" method 1 """

def regularize_matrix(A):
    """
    Regularizes a matrix by ensuring all its eigenvalues are positive.

    Parameters:
        A (ndarray): Input matrix.

    Returns:
        ndarray: Regularized matrix, positive-definite.
    """
    eig_vals, eig_vecs = np.linalg.eigh(A)
    if np.all(eig_vals > 0):
        return A  # The matrix is already positive-definite.

    # Apply exponential transformation to eigenvalues to ensure positivity.
    new_eig_vals = np.exp(eig_vals)
    Lambda = np.diag(new_eig_vals)
    B = eig_vecs @ Lambda @ eig_vecs.T
    return B



""" method 2 """

def spd_matrix_correction(FC_matrix):
    """
    Corrects a functional connectivity matrix to be symmetric positive-definite (SPD).

    Parameters:
        FC_matrix (ndarray): Input functional connectivity matrix.

    Returns:
        ndarray: Symmetric positive-definite matrix.
    """
    eig_vals, eig_vecs = np.linalg.eigh(FC_matrix)
    s, p = 0, np.inf
    neg_eig_vals = []

    for eig_val in eig_vals:
        if eig_val < 0:
            s += eig_val
            neg_eig_vals.append(eig_val)
        else:
            p = min(p, eig_val)

    s *= 2
    t = (s ** 2) * 100 + 1
    for neg_val in neg_eig_vals:
        pos_val = p * (s - neg_val) ** 2 / t
        indices = np.where(eig_vals == neg_val)
        eig_vals[indices] = pos_val

    Lambda = np.diag(eig_vals)
    B = eig_vecs @ Lambda @ eig_vecs.T
    return B



"""method 3"""

def nearest_positive_definite(A, reg=1e-6):
    """
    Find the nearest positive-definite matrix to the input matrix.

    Parameters:
        A (ndarray): Input matrix.
        reg (float): Regularization factor for the smallest eigenvalue.

    Returns:
        ndarray: Nearest positive-definite matrix.
    """
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return regularize_smallest_eigenvalue(A3, reg)

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    max_iter = 100
    iter_count = 0

    while not is_positive_definite(A3) and iter_count < max_iter:
        mineig = np.min(np.real(la.eigvals(A3)))
        print(f"Iteration {k}: min eigenvalue = {mineig}")
        A3 += I * (-mineig * k ** 2 + spacing)
        A3 = (A3 + A3.T) / 2  # Ensure symmetry
        k += 1
        iter_count += 1

    if iter_count >= max_iter:
        raise RuntimeError(f"Failed to find a positive semi-definite matrix within {iter_count} iterations")

    return regularize_smallest_eigenvalue(A3, reg)


def is_positive_definite(B):
    """
    Check if a matrix is positive-definite.

    Parameters:
        B (ndarray): Input matrix.

    Returns:
        bool: True if the matrix is positive-definite, False otherwise.
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def regularize_smallest_eigenvalue(A, reg):
    """
    Regularize the smallest eigenvalue of a matrix.

    Parameters:
        A (ndarray): Input matrix.
        reg (float): Regularization factor for the smallest eigenvalue.

    Returns:
        ndarray: Regularized matrix.
    """
    eig_vals, eig_vecs = np.linalg.eigh(A)
    if np.min(eig_vals) / np.max(eig_vals) < reg:
        A = eig_vecs @ np.diag(eig_vals + reg) @ eig_vecs.T
    return A



""" method 4"""

def nearest_positive_definite_matrix(A, epsilon=1e-8):
    """
    Convert a semi-positive definite matrix to the nearest positive definite matrix.

    Parameters:
        A (ndarray): Input semi-positive definite matrix.
        epsilon (float): Tolerance for eigenvalue adjustments.

    Returns:
        ndarray: Nearest positive definite matrix.
    """
    n = A.shape[0]
    eig_vals, eig_vecs = np.linalg.eigh(A)
    min_eigval = np.min(eig_vals)

    if min_eigval > epsilon:
        return A

    A_hat = A + np.eye(n) * (-min_eigval + epsilon)
    eig_vals_hat, _ = np.linalg.eigh(A_hat)

    return A_hat + np.eye(n) * (epsilon - np.min(eig_vals_hat))
