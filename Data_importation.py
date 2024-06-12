import os
import mne
import numpy as np


def edf_file_importation(folder):

    """
     Retrieves all files with an .edf extension and sorts them into train and test.

     parameter : folder : path to the folder containing the .edf files
    """

    file_train = []
    file_test_1= []
    if "Braccio_Connect" in folder.split("/") :
        train = "mi"
    else :
        train = "train"
    # Browse all files in the specified folder
    for file in os.listdir(folder):
        if file.endswith('.edf'):
            # Retrieves the first word of the file's name
            first_word = file.split('-')[0]
            # Check whether the first word is "train" or "test" and add to the appropriate list.
            if first_word.lower() == train:
                file_train.append(os.path.join(folder, file))
            elif first_word.lower() == 'test':
                file_test_1.append(os.path.join(folder, file))
    file_test_1 = sorted(file_test_1)
    file_test_2 = file_test_1[-2:]
    del file_test_1[-2:]

    return [file_train, file_test_1, file_test_2]




def dataset_creator(files, event, tmin=1, tmax=4, freq=500):

    """
    Take the data from every file and collect only those corresponding to GDF-left or GDF-right in the interest_data list
    Those events last for a number of data corresponding to segment_length
    Create a label array, in which we found 0 at the positions of GDF-right segments in the interest_data list and 1
    at the positions of GDF-left segments in the interest_data list

    parameters: files: string list corresponding to the data folders paths
                events: int list corresponding to the numbers representatives of the events of interest
                segment_length: int corresponding to the length of the data segment we will keep
    """

    MI = []
    rest = []
    tmin, tmax = tmin * freq, tmax * freq
    for file in files:
        data = mne.io.read_raw_edf(file, preload=True)
        raw_data = data.get_data()
        total_events, dict = mne.events_from_annotations(data)
        index_events = [dict[event[i]] for i in range(len(event))]  #[dict[event[0]], dict[event[1]]]
        index = [i for i in range(total_events.shape[0]) if total_events[i, 2] in index_events]
        interest_events = total_events[index, 0]
        interest_events_label = total_events[index, 2]
        for i in range(interest_events.shape[0]):
            if interest_events_label[i] == index_events[0]:
                MI.append(raw_data[:, interest_events[i] + tmin:interest_events[i] + tmax])
            elif interest_events_label[i] == index_events[1]:
                rest.append(raw_data[:, interest_events[i] + tmin:interest_events[i] + tmax])

    return np.array(MI), np.array(rest)