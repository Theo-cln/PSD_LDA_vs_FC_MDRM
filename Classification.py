import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from pyriemann.classification import FgMDM
from numpy import linalg as la
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from Visualisation import PSD_visualisation, FC_visualisation, R_squared_map_visualisation, Reorder_Rsquare, NS_visualisation
from Topomap_separate import topo_plot


"regularization of matrices"

def regularize_matrix(A):
    # Calculer les valeurs propres de la matrice
    eig_vals, eig_vec = np.linalg.eigh(A)
    if np.all(eig_vals > 0):
        return A  # La matrice est déjà définie positive, pas besoin de régularisation
    else :
        new_eig_vals = np.exp(eig_vals)
        Lambda = np.diag(new_eig_vals)
        # La matrice reconstruite B avec les nouvelles valeurs propres
        B = eig_vec @ Lambda @ eig_vec.T
        return B


def SPD (FC_matrix):
    eig_vals, eig_vec = np.linalg.eigh(FC_matrix)
    s = 0
    p = 100
    neg_val = []
    for i in eig_vals:
        if i < 0:
            s += i
            neg_val.append(i)
        else:
            if i < p:
                p = i
    s *= 2
    t = (s ** 2) * 100 + 1
    for n in neg_val:
        pos_val = p * (s - n) * (s - n) / t
        indices = np.where(eig_vals == n)
        eig_vals[indices] = pos_val
    Lambda = np.diag(eig_vals)
    B = eig_vec @ Lambda @ eig_vec.T
    return B


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


def nearest_pos_def_matrix(A, epsilon=1e-8):

    """
    Convert a semi-positive definite matrix to the nearest positive definite matrix.

    Parameters:
        A (ndarray): The input semi-positive definite matrix.
        epsilon (float): Tolerance for eigenvalue checks.

    Returns:
        ndarray: The nearest positive definite matrix.
    """

    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eigh(A)
    min_eigval = np.min(eigvals)
    if min_eigval > epsilon:
        return A
    else:
        A_hat = A + np.eye(n) * (-min_eigval + epsilon)
        eigvals_hat, _ = np.linalg.eigh(A_hat)
        return A_hat + np.eye(n) * (epsilon - np.min(eigvals_hat))


def K_fold(X, y, model):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Liste pour stocker les scores de performance
    scores = []
    # Boucle sur les différentes itérations de la validation croisée
    for train_index, test_index in kf.split(X):
        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Entraînement du modèle sur l'ensemble d'entraînement
        model.fit(X_train, y_train)
        # Évaluation de la performance sur l'ensemble de test
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        # Stockage du score
        scores.append(score)
    # Calcul de la moyenne des scores de performance
    mean_score = np.mean(scores)
    print("Performances : ", [round(score, 2) for score in scores])
    print("Mean performance :", mean_score)


def features_separability(MI, rest, previous_MI, previous_rest, target, electrodes):
    x_MI = MI[:, target[0], target[1]]
    y_MI = MI[:, target[2], target[3]]
    x_rest = rest[:, target[0], target[1]]
    y_rest = rest[:, target[2], target[3]]
    previous_x_MI = previous_MI[:, target[0], target[1]]
    previous_y_MI = previous_MI[:, target[2], target[3]]
    previous_x_rest = previous_rest[:, target[0], target[1]]
    previous_y_rest = previous_rest[:, target[2], target[3]]
    plt.figure(figsize=(10, 10))
    plt.scatter(x_MI, y_MI, c='red', label='MI')
    plt.scatter(x_rest, y_rest, c='blue', label='rest')
    plt.scatter(previous_x_MI, previous_y_MI, c='orange', label='previous MI')
    plt.scatter(previous_x_rest, previous_y_rest, c='green', label='previous rest')
    plt.xlabel(f'électrode {electrodes[target[0]]}, fréquence {target[1]+4}')
    plt.ylabel(f'électrode {electrodes[target[2]]}, fréquence {target[3]+4}')
    plt.title("Feature separability")
    plt.legend()
    plt.show()


def LDA_train(data_MI, data_rest, freqs_MI, freqs_rest, Ordered_R_squared_map, Ordered_channel_name, electrodes, phase_name, lowLimit=0, lda_classifier=LinearDiscriminantAnalysis(), k_fold=False, feature_name="PSD"):
    if feature_name == "PSD":
        R_squared_map_visualisation(Ordered_R_squared_map, freqs_MI, Ordered_channel_name, phase_name, feature_name="PSD")
    if feature_name == "NS":
        R_squared_map_visualisation(Ordered_R_squared_map, freqs_MI, Ordered_channel_name, phase_name, feature_name="NS")

    retry = "y"
    while retry == "y":
        target = input("electrode and frequency bin for the training : (format: electrode 1;frequency bin 1;"
                       "electrode 2;frequency bin 2) ")
        target = target.split(";")
        target[0] = electrodes.index(target[0])
        target[2] = electrodes.index(target[2])
        target[1] = int(target[1]) - lowLimit
        target[3] = int(target[3]) - lowLimit
        data_mean_MI = np.mean(data_MI, axis=0)
        data_mean_rest = np.mean(data_rest, axis=0)
        if feature_name == "PSD":
            PSD_visualisation(freqs_MI, freqs_rest, data_mean_MI[target[0]], data_mean_rest[target[0]], electrodes[target[0]], phase_name)
            PSD_visualisation(freqs_MI, freqs_rest, data_mean_MI[target[2]], data_mean_rest[target[2]], electrodes[target[2]], phase_name)
        if feature_name == "NS":
            NS_visualisation(freqs_MI, data_mean_MI[target[0]], data_mean_rest[target[0]], electrodes[target[0]], phase_name)
            NS_visualisation(freqs_MI, data_mean_MI[target[2]], data_mean_rest[target[2]], electrodes[target[2]], phase_name)
        X_train = np.vstack((np.hstack((data_MI[:, target[0], target[1]][:, None],data_rest[:, target[0], target[1]][:, None])),
                             np.hstack((data_MI[:, target[2], target[3]][:, None], data_rest[:, target[2], target[3]][:, None]))))
        print(f"LDA train shape : {X_train.shape}")
        y_train = np.hstack((np.ones((data_MI.shape[0])), np.zeros((data_rest.shape[0]))))
        """np.zeros((data_rest.shape[0]))[:, None], np.zeros((data_rest.shape[0]))[:, None]))"""
        print(f"LDA train shape :{y_train.shape}")
        lda_classifier_2 = lda_classifier
        if k_fold:
            K_fold(X_train, y_train, lda_classifier_2)
        retry = input("Do you want to retry ? (y/n)")
    print('LDA shapes : ', X_train.shape, y_train.shape)
    lda_classifier.fit(X_train, y_train)
    return lda_classifier, target


def LDA_test(data_MI, data_rest, previous_data_MI, previous_data_rest, lda_classifier, target, electrodes):
    X_test = np.vstack((np.hstack((data_MI[:, target[0], target[1]][:, None],data_rest[:, target[0], target[1]][:, None])),
                         np.hstack((data_MI[:, target[2], target[3]][:, None], data_rest[:, target[2], target[3]][:, None]))))
    y_test = np.hstack((np.ones((data_MI.shape[0])), np.zeros((data_rest.shape[0]))))
    print(f"LDA test shape : {X_test.shape}")
    print(f"LDA test shape : {y_test.shape}")
    features_separability(data_MI, data_rest, previous_data_MI, previous_data_rest, target, electrodes)
    lda_predictions = lda_classifier.predict(X_test)
    acc = accuracy_score(y_test, lda_predictions)
    lda_accuracy = classification_report(y_test, lda_predictions)
    return lda_predictions, lda_accuracy, acc


def mdm_train(fc_MI, fc_rest, Ordered_R_squared_map_NS, freqs_MI,Ordered_channel_name_NS, NS_MI, NS_Rest, NS_diff,
              electrodes, Hz, phase_name, mdm=FgMDM(), lowLimit=0, k_fold=False):
    R_squared_map_visualisation(Ordered_R_squared_map_NS, freqs_MI,Ordered_channel_name_NS, phase_name,
                                feature_name="Node Strength")
    retry = "y"
    while retry == "y":
        target = int(input("Frequency for the RMDM training : "))
        """
        NS_curves = input("electrodes for NS curves visualisation (format : electrode 1;electrode 2")
        NS_curves = NS_curves.split(";")
        channels = [electrodes.index(NS_curves[0]), electrodes.index(NS_curves[1])]
        for cpt, channel in enumerate(channels):
            NS_visualisation(freqs_MI, np.mean(NS_MI, axis=0)[channel], np.mean(NS_Rest, axis=0)[channel], NS_curves[cpt], phase_name)
        topo_plot(node_strength_diff, target - lowLimit, electrodes, Hz, 'R_square signed', vmin=np.min(node_strength_diff[:, target]),vmax=np.max(node_strength_diff[:, target]), phase_name=phase_name, frequency_name=target)
        """

        Channel_of_interest = ["C1", "C3", "C5", "CP1", "CP3", "CP5", "C2", "CP2"]
        indices_Channels_of_interest = [Ordered_channel_name_NS.index(i) for i in Channel_of_interest]
        """fc_MI = fc_MI[:, indices_Channels_of_interest, :, :]
        fc_rest = fc_rest[:, indices_Channels_of_interest, :, :]
        fc_MI = fc_MI[:, :, indices_Channels_of_interest, :]
        fc_rest = fc_rest[:, :, indices_Channels_of_interest, :]"""
        FC_visualisation(fc_MI, fc_rest, target, electrodes,phase_name)

        for i in range(fc_MI.shape[0]):
            fc_MI[i, :, :, target] = SPD(fc_MI[i, :, :, target])
            fc_rest[i, :, :, target] = SPD(fc_rest[i, :, :, target])
        FC_visualisation(fc_MI, fc_rest, target, electrodes, phase_name)

        X_train = np.vstack((fc_MI[:, :, :, target], fc_rest[:, :, :, target]))
        y_train = np.hstack((np.ones((fc_MI.shape[0])), np.zeros((fc_rest.shape[0]))))
        for i in range(X_train.shape[0]):
            X_train[i] = SPD(X_train[i])
        print(f"MDM X_train shape : {X_train.shape}")
        print(f"MDM y_train shape : {y_train.shape}")
        mdm_2 = mdm
        if k_fold:
            K_fold(X_train, y_train, mdm_2)
        retry = input("Do you want to retry ? (y/n)")
    mdm.fit(X_train, y_train)
    return mdm, target


def mdm_test(fc_MI, fc_rest, mdm, target, Ordered_channel_name_NS):
    Channel_of_interest = ["C1", "C3", "C5", "CP1", "CP3", "CP5", "C2", "CP2"]
    indices_Channels_of_interest = [Ordered_channel_name_NS.index(i) for i in Channel_of_interest]
    """fc_MI = fc_MI[:, indices_Channels_of_interest, :, :]
    fc_rest = fc_rest[:, indices_Channels_of_interest, :, :]
    fc_MI = fc_MI[:, :, indices_Channels_of_interest, :]
    fc_rest = fc_rest[:, :, indices_Channels_of_interest, :]"""
    X_test = np.vstack((fc_MI[:, :, :, target], fc_rest[:, :, :, target]))
    y_test = np.hstack((np.ones((fc_MI.shape[0])), np.zeros((fc_rest.shape[0]))))
    for i in range(X_test.shape[0]):
        X_test[i] = SPD(X_test[i])
    print(f"MDM test shape : {X_test.shape}")
    print(f"MDM test shape : {y_test.shape}")
    pred = mdm.predict(X_test)
    return classification_report(y_test, pred), accuracy_score(y_test, pred)
