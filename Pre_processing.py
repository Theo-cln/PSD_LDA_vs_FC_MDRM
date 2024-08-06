import numpy as np
from scipy.signal import butter, filtfilt

def pass_band_filtering(data, low_cutoff, high_cutoff, sampling_freq):
    """
    Apply a pass-band filter to the data using a Butterworth filter design.

    Parameters:
    - data (numpy array): The input data to be filtered. Expected shape: (n_epochs, n_channels, n_samples).
    - low_cutoff (float): The lower cutoff frequency for the Butterworth filter.
    - high_cutoff (float): The upper cutoff frequency for the Butterworth filter.
    - sampling_freq (float): The sampling frequency of the data.

    Returns:
    - filtered_data (numpy array): The filtered data with the same shape as the input data.
    """
    # Normalize the cutoff frequencies with respect to the Nyquist frequency
    nyquist_freq = 0.5 * sampling_freq
    Wn = [low_cutoff / nyquist_freq, high_cutoff / nyquist_freq]

    # Design a 5th order Butterworth bandpass filter
    b, a = butter(5, Wn, btype='band', analog=False)

    # Apply the filter to each epoch and channel
    filtered_data = np.zeros_like(data)
    for epoch_idx in range(data.shape[0]):
        for channel_idx in range(data.shape[1]):
            filtered_data[epoch_idx, channel_idx, :] = filtfilt(b, a, data[epoch_idx, channel_idx, :])

    return filtered_data

def CAR(data):
    """
    Compute the Common Average Reference (CAR) for the data.

    Parameters:
    - data (numpy array): The input data. Expected shape: (n_epochs, n_channels, n_samples).

    Returns:
    - car_data (numpy array): The data after applying the Common Average Reference (CAR).
    """
    # Compute the mean across all channels for each epoch
    channel_mean = np.mean(data, axis=1, keepdims=True)

    # Subtract the mean from each channel in the epoch
    car_data = data - channel_mean

    return car_data
