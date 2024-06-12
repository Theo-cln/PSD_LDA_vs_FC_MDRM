import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, filtfilt, welch, coherence
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



from Data_importation import edf_file_importation, dataset_creator
from Pre_processing import pass_band_filtering, CAR
from Features_extraction import PSD, FC

print("importation done")



"""Variables"""


Hz = 500
tmin = 1
tmax = 4  #in seconds
"""segment_length = tmax*Hz
overlap = 1 #in seconds
overlap = overlap * Hz"""
lowLimit = 6
highLimit = 36

folder = "/Users/theo.coulon/data/sub-02/ses-01/"
event = ['OVTK_GDF_Left', 'OVTK_GDF_Right']




"""data importation"""

train_file, test_file_1, test_file_2 = edf_file_importation(folder)



"""train and test sets creation"""

MI_train, rest_train = dataset_creator(train_file, event,  tmin=tmin, tmax=tmax, freq=Hz)

MI_test_1, rest_test_1 = dataset_creator(test_file_1, event,  tmin=tmin, tmax=tmax, freq=Hz)

MI_test_2, rest_test_2 = dataset_creator(test_file_2, event,  tmin=tmin, tmax=tmax, freq=Hz)

print('datasets shapes : '
      '\n Train :', MI_train.shape, rest_train.shape,
      '\n Test_1 : ', MI_test_1.shape, rest_test_1.shape,
      '\n Test_2 : ', MI_test_2.shape, rest_test_2.shape)




"""data preprocessing"""

filtered_MI_train = pass_band_filtering(MI_train, lowLimit, highLimit, Hz)
filtered_rest_train = pass_band_filtering(rest_train, lowLimit, highLimit, Hz)
filtered_MI_test_1 = pass_band_filtering(MI_test_1, lowLimit, highLimit, Hz)
filtered_rest_test_1 = pass_band_filtering(rest_test_1, lowLimit, highLimit, Hz)
filtered_MI_test_2 = pass_band_filtering(MI_test_2, lowLimit, highLimit, Hz)
filtered_rest_test_2 = pass_band_filtering(rest_test_2, lowLimit, highLimit, Hz)


car_MI_train = CAR(filtered_MI_train)
car_rest_train = CAR(filtered_rest_train)
car_MI_test_1 = CAR(filtered_MI_test_1)
car_rest_test_1 = CAR(filtered_rest_test_1)
car_MI_test_2 = CAR(filtered_MI_test_2)
car_rest_test_2 = CAR(filtered_rest_test_2)




"""Features exctraction"""

Freqs_MI = []
Freqs_rest = []
Psd_mean_MI = []
Psd_mean_rest = []



print('Shapes PSD : ')

freqs_MI_train, psd_mean_MI_train = PSD(car_MI_train, Hz, lowLimit, highLimit)
freqs_rest_train, psd_mean_rest_train = PSD(car_rest_train, Hz, lowLimit, highLimit)

print('Shape PSD train : ', psd_mean_MI_train.shape, psd_mean_rest_train.shape)


Freqs_MI.append(freqs_MI_train)
Freqs_rest.append(freqs_rest_train)
Psd_mean_MI.append(psd_mean_MI_train)
Psd_mean_rest.append(psd_mean_rest_train)



freqs_MI_test_1, psd_mean_MI_test_1 = PSD(car_MI_test_1, Hz)
freqs_rest_test_1, psd_mean_rest_test_1 = PSD(car_rest_test_1, Hz)

print('Shape PSD test_1 : ', psd_mean_MI_test_1.shape, psd_mean_rest_test_1.shape)



Freqs_MI.append(freqs_MI_test_1)
Freqs_rest.append(freqs_rest_test_1)
Psd_mean_MI.append(psd_mean_MI_test_1)
Psd_mean_rest.append(psd_mean_rest_test_1)



freqs_MI_test_2, psd_mean_MI_test_2 = PSD(car_MI_test_2, Hz)
freqs_rest_test_2, psd_mean_rest_test_2 = PSD(car_rest_test_2, Hz)

print('Shape PSD test_2 : ', psd_MI_test_2.shape, psd_rest_test_2.shape)

Freqs_MI.append(freqs_MI_test_2)
Freqs_rest.append(freqs_rest_test_2)
Psd_mean_MI.append(psd_mean_MI_test_2)
Psd_mean_rest.append(psd_mean_rest_test_2)





FC_MI_train = FC(filtered_MI_train, Hz)
FC_rest_train = FC(filtered_rest_train, Hz)
FC_MI_test_1 = FC(filtered_MI_test_1, Hz)
FC_rest_test_1 = FC(filtered_rest_test_1, Hz)
FC_MI_test_2 = FC(filtered_MI_test_2, Hz)
FC_rest_test_2 = FC(filtered_rest_test_2, Hz)


print('Shape FC : ',
      ' \n Train : ', FC_MI_train.shape, FC_rest_train.shape,
      ' \n Test_1 : ', FC_MI_test_1.shape, FC_rest_test_1.shape,
      ' \n Test_2 : ', FC_MI_test_2.shape, FC_rest_test_2.shape)


FC_mean_MI_train = np.mean(FC_MI_train, axis=0)
FC_mean_rest_train = np.mean(FC_rest_train, axis=0)
FC_mean_MI_test_1 = np.mean(FC_MI_test_1, axis=0)
FC_mean_rest_test_1 = np.mean(FC_rest_test_1, axis=0)
FC_mean_MI_test_2 = np.mean(FC_MI_test_2, axis=0)
FC_mean_rest_test_2 = np.mean(FC_rest_test_2, axis=0)



"""Classification"""



"""Plots"""





"""
cross_spectrum = fft_data[i, j] * np.conj(fft_data[i, k]) # scpipy coherence
                auto_spectrum_j = np.abs(fft_data[i, j]) ** 2
                auto_spectrum_k = np.abs(fft_data[i, k]) ** 2
                coherence_value = np.abs(cross_spectrum) ** 2 / (auto_spectrum_j * auto_spectrum_k)
                coherence_matrix[i, j, k] = coherence_value
                coherence_matrix[i, k, j] = coherence_value




"""


