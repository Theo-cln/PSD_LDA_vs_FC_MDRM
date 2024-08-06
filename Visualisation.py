import matplotlib.pyplot as plt
import numpy as np


def PSD_visualisation(Freqs_MI, Freqs_rest, Psd_mean_MI, Psd_mean_rest, electrode_name, phase_name):
    """
    Visualize the Power Spectral Density (PSD) for Motor Imagery (MI) and rest phases.

    Parameters:
        Freqs_MI (ndarray): Frequencies corresponding to the MI phase.
        Freqs_rest (ndarray): Frequencies corresponding to the rest phase.
        Psd_mean_MI (ndarray): PSD values for the MI phase.
        Psd_mean_rest (ndarray): PSD values for the rest phase.
        electrode_name (str): Name of the electrode being analyzed.
        phase_name (str): Name of the phase being visualized.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(Freqs_MI, Psd_mean_MI, color='red', label='MI')
    plt.plot(Freqs_rest, Psd_mean_rest, color='blue', label='Rest')
    plt.title(f'Average Power Spectral Density (PSD) of MI vs Rest ({phase_name} {electrode_name})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectrum (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()


def FC_visualisation(FC_MI, FC_rest, frequency, channel_names, phase_name):
    """
    Visualize the Functional Connectivity (FC) for MI and rest phases.

    Parameters:
        FC_MI (ndarray): Functional connectivity matrix for the MI phase.
        FC_rest (ndarray): Functional connectivity matrix for the rest phase.
        frequency (int): The specific frequency to visualize.
        channel_names (list): List of channel names.
        phase_name (str): Name of the phase being visualized.

    Returns:
        None
    """
    FC_mean_MI = np.mean(FC_MI, axis=0)
    FC_mean_rest = np.mean(FC_rest, axis=0)
    channels = range(len(channel_names))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    im_MI = axes[0].imshow(FC_mean_MI[:, :, frequency], cmap='bwr')
    axes[0].set_title(f'Functional Connectivity MI {phase_name} {frequency} Hz')
    axes[0].set_xticks(channels)
    axes[0].set_xticklabels(channel_names, rotation=90, fontsize=5)
    axes[0].set_yticks(channels)
    axes[0].set_yticklabels(channel_names, fontsize=5)

    im_rest = axes[1].imshow(FC_mean_rest[:, :, frequency], cmap='bwr')
    axes[1].set_title(f'Functional Connectivity Rest {phase_name} {frequency} Hz')
    axes[1].set_xticks(channels)
    axes[1].set_xticklabels(channel_names, rotation=90, fontsize=5)
    axes[1].set_yticks(channels)
    axes[1].set_yticklabels(channel_names, fontsize=5)

    fig.colorbar(im_MI, ax=axes, orientation='vertical')
    plt.show()


def R_squared_map_visualisation(R_squared_map, frequency, channel_names=None, phase_name="", feature_name=""):
    """
    Visualize the R-squared map for a specific feature across channels and frequencies.

    Parameters:
        R_squared_map (ndarray): The R-squared values matrix.
        frequency (list): List of frequency values.
        channel_names (list): List of channel names.
        phase_name (str): Name of the phase being visualized.
        feature_name (str): Name of the feature being visualized.

    Returns:
        None
    """
    if channel_names is None:
        channel_names = []

    channels = range(len(channel_names))
    frequency_indices = range(len(frequency))

    plt.figure(figsize=(16, 16))
    plt.imshow(R_squared_map, cmap='jet', aspect='auto')
    plt.title(f'{phase_name} R-squared Map of {feature_name}')

    if channel_names:
        plt.yticks(channels, channel_names)
    plt.xticks(frequency_indices, frequency)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channels')
    plt.colorbar()
    plt.show()


def reorder_r_squared(Rsquare, electrodes_orig):
    """
    Reorder R-squared values based on a standard electrode layout.

    Parameters:
        Rsquare (ndarray): R-squared values to reorder.
        electrodes_orig (list): Original list of electrode names.

    Returns:
        tuple: Reordered R-squared values and target electrode names.
    """
    if len(electrodes_orig) >= 64:
        electrodes_target = [
            'Fp1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
            'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1',
            'PO7', 'PO3', 'O1', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz',
            'Iz', 'Fp2', 'AF8', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT10', 'FT8', 'FC6', 'FC4',
            'FC2', 'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4',
            'P2', 'PO8', 'PO4', 'O2'
        ]
    else:
        electrodes_target = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
            'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
            'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
        ]

    index_elec = [electrodes_orig.index(elec) for elec in electrodes_target if elec in electrodes_orig]

    Rsquare_final = np.zeros_like(Rsquare)
    for i, idx in enumerate(index_elec):
        Rsquare_final[i, :] = Rsquare[idx, :]

    return Rsquare_final, electrodes_target


def NS_visualisation(freq, NS_MI, NS_Rest, channel, phase_name):
    """
    Visualize the Node Strength (NS) for MI and rest phases.

    Parameters:
        freq (ndarray): Frequency values.
        NS_MI (ndarray): Node strength values for the MI phase.
        NS_Rest (ndarray): Node strength values for the rest phase.
        channel (str): Name of the channel being analyzed.
        phase_name (str): Name of the phase being visualized.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freq, NS_MI, label='NS MI', color='red')
    plt.plot(freq, NS_Rest, label='NS Rest', color='blue')
    plt.title(f'Node Strength of MI vs Rest for {phase_name} {channel}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Node Strength')
    plt.legend()
    plt.grid(True)
    plt.show()
