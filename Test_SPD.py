import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

FC_matrix = np.loadtxt('FC.txt', dtype=float)
plt.imshow(FC_matrix, cmap='bwr')
plt.colorbar()
plt.show()
eig_vals, eig_vec = np.linalg.eigh(FC_matrix)
s = 0
p = 100
neg_val = []
for i in eig_vals:
    if i < 0 :
        s+=i
        neg_val.append(i)
    else :
        if i < p :
            p=i
s *= 2
t = (s**2) * 100 + 1
for n in neg_val:
    pos_val = p * (s-n) * (s-n)/t
    indices = np.where(eig_vals == n)
    eig_vals[indices] = pos_val
Lambda = np.diag(eig_vals)
# La matrice reconstruite B avec les nouvelles valeurs propres
B = eig_vec @ Lambda @ eig_vec.T

plt.imshow(B, cmap='bwr')
plt.colorbar()
plt.show()



def nearestPD(A, reg=1e-6):

    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        # Regularize
        ei, ev = np.linalg.eigh(A3)
        if np.min(ei) / np.max(ei) < reg:
            A3 = ev @ np.diag(ei + reg) @ ev.T
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    max_iter = 100
    iter_count = 0
    while not isPD(A3) and iter_count < max_iter:
        mineig = np.min(np.real(la.eigvals(A3)))
        print(f"Iteration {k}: min eigenvalue = {mineig}")
        A3 += I * (-mineig * k ** 2 + spacing)
        A3 = (A3 + A3.T) / 2  # Ensure symmetry
        k += 1
        iter_count += 1
    if iter_count >= max_iter:
        raise RuntimeError(f"Could not find a positive semi-definite matrix in {iter_count} iterations")
    # Regularize
    ei, ev = np.linalg.eigh(A3)
    if np.min(ei) / np.max(ei) < reg:
        A3 = ev @ np.diag(ei + reg) @ ev.T
    return A3


def isPD(B):

    """Returns true when input is positive-definite, via Cholesky"""

    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def isPD2(B):
    """Returns true when input is positive-definite, via eigenvalues"""
    if np.any(np.linalg.eigvals(B) < 0.0):
        return False
    else:
        return True


C = nearestPD(FC_matrix)

plt.imshow(C, cmap='bwr')
plt.colorbar()
plt.show()

print(np.corrcoef(FC_matrix.flatten(), B.flatten())[0, 1])