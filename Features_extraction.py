import numpy as np
from scipy.signal import welch, coherence, csd
from mne import viz
from tqdm import tqdm
from pyriemann.estimation import Coherences


def PSD(signal, fs, lowLimit, highLimit):

    """
    calculate the Power Spectral Density (PSD) mean of the signal

    parameters : signal : numpy array representing the signal
                 sampling_rate : the sampling rate of the signal
    """

    window = int(0.5 * fs)  # 0.5 seconde
    overlap = int(0.25 * fs)  # 0.25 seconde
    PSD_all_trials = []
    for trial in signal:
        PSD_trial = []
        for channel in trial:
            f, Pxx = welch(channel, fs=fs, window='hamming', nperseg=window, noverlap=overlap, scaling='density', nfft=500, detrend='constant')
            PSD_trial.append(Pxx)
        PSD_all_trials.append(PSD_trial)
    PSD_all_trials = np.array(PSD_all_trials)
    PSD_dB = 10 * np.log10(PSD_all_trials)
    lowFreqLimits = np.where(f == lowLimit)[0][0]
    highFreqLimits = np.where(f == highLimit)[0][0]
    f = f[lowFreqLimits:highFreqLimits]
    PSD_dB = PSD_dB[:, :, lowFreqLimits:highFreqLimits]

    return f, PSD_dB


def Compute_Rsquare_Map_Welch(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape
    Rsquare_tab = np.zeros([a[1],a[2]])
    for k in range(b[1]):
        for l in range(b[2]):
            concat_tab_MI = []
            concat_tab_Rest = []
            for i in range(a[0]):
                concat_tab_MI.append(Power_of_trials_1[i,k,l])
                concat_tab_Rest.append(Power_of_trials_2[i,k,l])
            #correlation_matrix = np.corrcoef(concat_tab_MI, concat_tab_Rest)
            Sum_q = sum(concat_tab_MI)
            Sum_r = sum(concat_tab_Rest)
            n1 = len(concat_tab_MI)
            n2 = len(concat_tab_Rest)
            sumsqu1 = sum(np.multiply(concat_tab_MI,concat_tab_MI))
            sumsqu2 = sum(np.multiply(concat_tab_Rest,concat_tab_Rest))
            G=((Sum_q+Sum_r)**2)/(n1+n2)
            Rsquare_tab[k,l] = (Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G)

    diff = Power_of_trials_2-Power_of_trials_1
    diff = np.mean(diff, axis=0)
    for j in range(diff.shape[0]):
        for k in range(diff.shape[1]):
            if diff[j,k] < 0:
                Rsquare_tab[j,k] = 0



    return Rsquare_tab


def FC(data, freq, lowLimit, highLimit):

    """
    Calculate the functionnal connectivity between every signal in the dataset

    parameters : signal : numpy array representing the signal
            fs : sampling rate of the signal
            lowcut : lower bound of the data taken in account
            highcut : upper bound of the data taken in account
    """

    n_segment = data.shape[0]
    n_channels = data.shape[1]
    # Initialiser la matrice de cohérence
    coherence_matrix = np.zeros((n_segment, n_channels, n_channels, 251))
    # Calculer la cohérence entre les paires de canaux
    for i in range(n_segment):
        progress_bar = tqdm(total=n_channels, desc=f'Processing trial n°{i} : ', unit='%')
        for j in range(n_channels):
            for k in range(j+1, n_channels):
                frequencies, coherence_values = coherence(data[i, j], data[i, k], fs=freq, nfft=500)
                coherence_matrix[i, j, k] = coherence_values
                coherence_matrix[i, k, j] = coherence_values
            progress_bar.update(1)
            progress_bar.set_postfix({"progress": f"{progress_bar.n/progress_bar.total:.0%}"})
        progress_bar.close()
    lowFreqLimits = np.where(frequencies == lowLimit)[0][0]
    highFreqLimits = np.where(frequencies == highLimit)[0][0]
    coherence_matrix = coherence_matrix[:, :, :, lowFreqLimits:highFreqLimits]

    return coherence_matrix





def imaginary_coherence(data, freq, lowLimit, highLimit):

    """
    Compute the imaginary coherence of a signal.

    Args:
    - data : array representing the signal (shape(nb_trials, nb_channels, nb_data_points))
    - freq : frequency of the signal
    - lowlimit : lower bound of the data taken in account (in Hz)
    - highlimit : upper bound of the data taken in account (in Hz)

    Returns:
    - imaginary_coherence (shape(nb_trials, nb_channels, nb_channels, HighLimit - LowLimit))
    """

    window = int(0.5 * freq)  # 0.5 seconds
    overlap = int(0.25 * freq)  # 0.25 seconds
    n_segment = data.shape[0]
    n_channels = data.shape[1]
    im_coherence = np.zeros((n_segment, n_channels, n_channels, 251))
    for i in range(n_segment):
        progress_bar = tqdm(total=n_channels, desc=f'Processing trial n°{i} : ', unit='%')
        for j in range(n_channels):
            for k in range(j+1, n_channels):
                sfarrA, Pxx = welch(data[i,j], fs=freq, window='hamming', nperseg=window, noverlap=overlap, scaling='density', nfft=500)
                sfarrB, Pyy = welch(data[i,k], fs=freq, window='hamming', nperseg=window, noverlap=overlap, scaling='density', nfft=500)
                sfarrAB, Pxy = csd(data[i,j], data[i,k], fs=freq, window='hamming', nperseg=window, noverlap=overlap, scaling='density', nfft=500)
                Icoh = np.abs(np.imag(Pxy)) / Pxx * Pyy
                im_coherence[i, j, k] = Icoh
                im_coherence[i, k, j] = Icoh
            progress_bar.update(1)
            progress_bar.set_postfix({"progress": f"{progress_bar.n/progress_bar.total:.0%}"})
        progress_bar.close()
    lowFreqLimits = np.where(sfarrAB == lowLimit)[0][0]
    highFreqLimits = np.where(sfarrAB == highLimit)[0][0]
    im_coherence = im_coherence[:, :, :, lowFreqLimits:highFreqLimits]

    return im_coherence


def node_strength(coherence_matrix):

    """
    Calculate the node_strength of the coherence matrix by summing up the column and line corresponding to each electrode
    summing on all the frequencies

    parameter : coherence_matrix, the coherence matrix to calculate the node strength
    """

    num_trial = coherence_matrix.shape[0]
    num_channels = coherence_matrix.shape[1]
    num_freqs = coherence_matrix.shape[-1]
    node_strengths = np.zeros((num_trial, num_channels, num_freqs))  # Initialisation des node strengths à zéro
    # Calcul de la somme des cohérences absolues pour chaque canal à chaque fréquence
    for trial in range(num_trial):
        for channel in range(num_channels):
            for freq in range(num_freqs):
                node_strengths[trial, channel, freq] += np.sum(coherence_matrix[trial, channel, :, freq])
                node_strengths[trial, channel, freq] += np.sum(coherence_matrix[trial, :, channel, freq])

    return node_strengths


def NSdiff (node_strength_MI, node_strength_rest):
    node_strength_diff = np.zeros((node_strength_MI.shape[1], node_strength_MI.shape[2]))
    for freq in range (node_strength_MI.shape[2]):
        for channel in range(node_strength_MI.shape[1]):
            diff = node_strength_MI[:, channel, freq] - node_strength_rest[:, channel, freq]
            node_strength_diff[channel, freq] = np.mean(diff)
    print(node_strength_rest[:, 13, 8], "\n --- \n", node_strength_MI[:, 13, 8],  "\n --- \n", node_strength_MI.shape[2])

    return node_strength_diff


def FC_Fucon(data, freq, lowLimit, highLimit):

    n_segment = data.shape[0]
    n_channels = data.shape[1]

    window = int(0.5 * freq)  # 0.5 seconde
    overlap = 0.25  # 0.25 seconde

    # Calculer la transformée de Fourier pour chaque canal
    fft_data = np.fft.fft(data, axis=2)

    # Initialiser la matrice de cohérence
    coherence_matrix = np.ones((n_segment, n_channels, n_channels, 251))
    coh = Coherences(window, overlap, lowLimit, highLimit,freq)
    # Calculer la cohérence entre les paires de canaux
    for i in range(n_segment):
        progress_bar = tqdm(total=n_channels, desc=f'Processing trial n°{i} : ', unit='%')
        r = coh.fit_transform(data[i])
        print(r.shape)

        progress_bar.update(1)
        progress_bar.set_postfix({"progress": f"{progress_bar.n / progress_bar.total:.0%}"})
        progress_bar.close()

    return r

