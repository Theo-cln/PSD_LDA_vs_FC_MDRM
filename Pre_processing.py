import numpy as np
from scipy.signal import butter, filtfilt


def pass_band_filtering(data, lowlimit, highlimit, freq):

    """
    Apply a pass-band filtering to the data between lowlimit and highlimit using a butterworth filter design

    parameters : data : numpy array corresponding to the data to filter
                 low_limit : lower bound of the butterworth
                 high_limit : upper bound of the butterworth
    """

    Wn = [lowlimit / (0.5 * freq), highlimit / (0.5 * freq)]
    b, a = butter(5, Wn, btype='band', analog=False)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])

    return filtered_data


def CAR(data):

    """
    Calculate the common average reference of the signal

    parameters : data : numpy array corresponding to the data

    return: car_data : numpy array corresponding to the common average reference of the signal
    """

    car_data = np.zeros_like(data)
    mean = np.mean(data, axis=1)
    for i in range(data.shape[1]):
        car_data[:, i] = data[:, i] - mean

    return car_data