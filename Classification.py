import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from pyriemann.classification import FgMDM
from Visualisation import (
    PSD_visualisation,
    FC_visualisation,
    R_squared_map_visualisation,
    NS_visualisation
)
from Topomap_separate import topo_plot
from SPD_matrices import nearest_positive_definite

def k_fold_validation(X, y, model, n_splits=10, random_state=42):
    """
    Perform K-Fold Cross-Validation on the given model.

    Parameters:
    - X: Features array.
    - y: Labels array.
    - model: Model to be trained and validated.
    - n_splits: Number of folds (default is 10).
    - random_state: Seed for random shuffling (default is 42).

    Returns:
    - mean_score: Mean accuracy score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        scores.append(score)

    mean_score = np.mean(scores)
    print(f"Performance per fold: {[round(score, 2) for score in scores]}")
    print(f"Mean performance: {mean_score:.4f}")
    return mean_score


def plot_feature_separability(MI, rest, prev_MI, prev_rest, target, electrodes):
    """
    Visualize the separability of features between Motor Imagery (MI) and rest states.

    Parameters:
    - MI, rest: Current data for MI and rest.
    - prev_MI, prev_rest: Previous data for MI and rest.
    - target: Selected electrode and frequency bins for comparison.
    - electrodes: List of electrode names.
    """
    x_MI, y_MI = MI[:, target[0], target[1]], MI[:, target[2], target[3]]
    x_rest, y_rest = rest[:, target[0], target[1]], rest[:, target[2], target[3]]
    prev_x_MI, prev_y_MI = prev_MI[:, target[0], target[1]], prev_MI[:, target[2], target[3]]
    prev_x_rest, prev_y_rest = prev_rest[:, target[0], target[1]], prev_rest[:, target[2], target[3]]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_MI, y_MI, c='red', label='MI')
    plt.scatter(x_rest, y_rest, c='blue', label='Rest')
    plt.scatter(prev_x_MI, prev_y_MI, c='orange', label='Previous MI')
    plt.scatter(prev_x_rest, prev_y_rest, c='green', label='Previous Rest')
    plt.xlabel(f'Electrode {electrodes[target[0]]}, Frequency {target[1] + 4}')
    plt.ylabel(f'Electrode {electrodes[target[2]]}, Frequency {target[3] + 4}')
    plt.title("Feature Separability")
    plt.legend()
    plt.show()


def train_lda(data_MI, data_rest, freqs_MI, freqs_rest, r_squared_map, channel_names, electrodes, phase_name,
              low_limit=0, lda_classifier=LinearDiscriminantAnalysis(), use_k_fold=False, feature_name="PSD"):
    """
    Train an LDA classifier using selected features.

    Parameters:
    - data_MI, data_rest: Data for MI and rest states.
    - freqs_MI, freqs_rest: Frequency arrays for MI and rest states.
    - r_squared_map: Ordered R-squared map for feature selection.
    - channel_names: List of channel names.
    - electrodes: List of electrode names.
    - phase_name: Name of the phase (e.g., training, testing).
    - low_limit: Lower frequency limit for feature selection (default is 0).
    - lda_classifier: LDA classifier instance (default is LinearDiscriminantAnalysis).
    - use_k_fold: Whether to use K-Fold cross-validation (default is False).
    - feature_name: Name of the feature type (default is "PSD").

    Returns:
    - lda_classifier: Trained LDA classifier.
    - target: Selected target electrodes and frequency bins.
    """
    R_squared_map_visualisation(r_squared_map, freqs_MI, channel_names, phase_name, feature_name=feature_name)

    while True:
        target_input = input("Select electrode and frequency bin for training (format: electrode1;freq1;electrode2;freq2): ")
        target = [int(x) if i % 2 else electrodes.index(x) for i, x in enumerate(target_input.split(";"))]
        target[1] -= low_limit
        target[3] -= low_limit

        data_mean_MI, data_mean_rest = np.mean(data_MI, axis=0), np.mean(data_rest, axis=0)

        if feature_name == "PSD":
            PSD_visualisation(freqs_MI, freqs_rest, data_mean_MI[target[0]], data_mean_rest[target[0]], electrodes[target[0]], phase_name)
            PSD_visualisation(freqs_MI, freqs_rest, data_mean_MI[target[2]], data_mean_rest[target[2]], electrodes[target[2]], phase_name)
        elif feature_name == "NS":
            NS_visualisation(freqs_MI, data_mean_MI[target[0]], data_mean_rest[target[0]], electrodes[target[0]], phase_name)
            NS_visualisation(freqs_MI, data_mean_MI[target[2]], data_mean_rest[target[2]], electrodes[target[2]], phase_name)

        X_train = np.vstack([
            np.hstack([data_MI[:, target[0], target[1]][:, None], data_MI[:, target[2], target[3]][:, None]]),
            np.hstack([data_rest[:, target[0], target[1]][:, None], data_rest[:, target[2], target[3]][:, None]])
        ])
        y_train = np.hstack([np.ones(data_MI.shape[0]), np.zeros(data_rest.shape[0])])

        print(f"LDA Training Data Shape: {X_train.shape}")
        print(f"LDA Labels Shape: {y_train.shape}")

        if use_k_fold:
            k_fold_validation(X_train, y_train, lda_classifier)

        retry = input("Do you want to retry with different targets? (y/n): ")
        if retry.lower() != 'y':
            break

    lda_classifier.fit(X_train, y_train)
    return lda_classifier, target


def test_lda(data_MI, data_rest, prev_data_MI, prev_data_rest, lda_classifier, target, electrodes):
    """
    Test the LDA classifier on new data.

    Parameters:
    - data_MI, data_rest: Test data for MI and rest states.
    - prev_data_MI, prev_data_rest: Previous data for comparison.
    - lda_classifier: Trained LDA classifier.
    - target: Selected target electrodes and frequency bins.
    - electrodes: List of electrode names.

    Returns:
    - lda_predictions: Predictions made by the LDA classifier.
    - lda_accuracy: Classification report.
    - accuracy: Accuracy score of the LDA classifier.
    """
    X_test = np.vstack([
        np.hstack([data_MI[:, target[0], target[1]][:, None], data_MI[:, target[2], target[3]][:, None]]),
        np.hstack([data_rest[:, target[0], target[1]][:, None], data_rest[:, target[2], target[3]][:, None]])
    ])
    y_test = np.hstack([np.ones(data_MI.shape[0]), np.zeros(data_rest.shape[0])])

    print(f"LDA Test Data Shape: {X_test.shape}")
    print(f"LDA Labels Shape: {y_test.shape}")

    plot_feature_separability(data_MI, data_rest, prev_data_MI, prev_data_rest, target, electrodes)

    lda_predictions = lda_classifier.predict(X_test)
    lda_accuracy = classification_report(y_test, lda_predictions)
    accuracy = accuracy_score(y_test, lda_predictions)

    return lda_predictions, lda_accuracy, accuracy


def train_mdrm(fc_MI, fc_rest, r_squared_map_ns, freqs_MI, channel_names_ns, ns_diff, electrodes, Hz, phase_name,
              low_limit=0, mdrm_classifier=FgMDM(), use_k_fold=False):
    """
    Train an MDRM classifier using functional connectivity features.

    Parameters:
    - fc_MI, fc_rest: Functional connectivity data for MI and rest states.
    - r_squared_map_ns: Ordered R-squared map for Node Strength.
    - freqs_MI: Frequency array for MI states.
    - channel_names_ns: List of channel names for Node Strength.
    - ns_diff: Node Strength difference map.
    - electrodes: List of electrode names.
    - Hz: Sampling frequency.
    - phase_name: Name of the phase (e.g., training, testing).
    - low_limit: Lower frequency limit for feature selection (default is 0).
    - mdrm_classifier: MDRM classifier instance (default is FgMDM).
    - use_k_fold: Whether to use K-Fold cross-validation (default is False).

    Returns:
    - mdrm_classifier: Trained MDRM classifier.
    - target: Selected frequency bin for training.
    """
    R_squared_map_visualisation(r_squared_map_ns, freqs_MI, channel_names_ns, phase_name, feature_name="Node Strength")

    while True:
        target = int(input("Select frequency bin for MDRM training: ")) - 4

        topo_plot(ns_diff, target - low_limit, electrodes, Hz, 'R_square signed',
                  vmin=np.min(ns_diff[:, target]), vmax=np.max(ns_diff[:, target]),
                  phase_name=phase_name, frequency_name=target)
        FC_visualisation(fc_MI, fc_rest, target, electrodes, phase_name)

        for i in range(fc_MI.shape[0]):
            fc_MI[i, :, :, target] = nearest_positive_definite(fc_MI[i, :, :, target])
            fc_rest[i, :, :, target] = nearest_positive_definite(fc_rest[i, :, :, target])

        FC_visualisation(fc_MI, fc_rest, target, electrodes, phase_name)

        X_train = np.vstack([fc_MI[:, :, :, target], fc_rest[:, :, :, target]])
        y_train = np.hstack([np.ones(fc_MI.shape[0]), np.zeros(fc_rest.shape[0])])

        for i in range(X_train.shape[0]):
            X_train[i] = nearest_positive_definite(X_train[i])

        print(f"MDRM Training Data Shape: {X_train.shape}")
        print(f"MDRM Labels Shape: {y_train.shape}")

        if use_k_fold:
            k_fold_validation(X_train, y_train, mdrm_classifier)

        retry = input("Do you want to retry with a different frequency bin? (y/n): ")
        if retry.lower() != 'y':
            break

    mdrm_classifier.fit(X_train, y_train)
    return mdrm_classifier, target


def test_mdrm(fc_MI, fc_rest, mdrm_classifier, target, channel_names_ns):
    """
    Test the MDRM classifier on new data.

    Parameters:
    - fc_MI, fc_rest: Test data for MI and rest states.
    - mdrm_classifier: Trained MDRM classifier.
    - target: Selected frequency bin for testing.
    - channel_names_ns: List of channel names for Node Strength.

    Returns:
    - classification_report: Classification report for the MDRM classifier.
    - accuracy: Accuracy score of the MDRM classifier.
    """
    X_test = np.vstack([fc_MI[:, :, :, target], fc_rest[:, :, :, target]])
    y_test = np.hstack([np.ones(fc_MI.shape[0]), np.zeros(fc_rest.shape[0])])

    for i in range(X_test.shape[0]):
        X_test[i] = nearest_positive_definite(X_test[i])

    print(f"MDRM Test Data Shape: {X_test.shape}")
    print(f"MDRM Labels Shape: {y_test.shape}")

    predictions = mdrm_classifier.predict(X_test)
    return classification_report(y_test, predictions), accuracy_score(y_test, predictions)
