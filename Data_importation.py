import os
import mne
import numpy as np


def edf_file_importation(folder):
    """
    Import and categorize .edf files from a specified folder into training and test sets.

    Parameters:
    - folder (str): Path to the folder containing the .edf files.

    Returns:
    - list: A list containing three lists: [training_files, test_files_1, test_files_2].
    """

    # Initialize lists to store file paths
    train_files = []
    test_files_1 = []

    # Determine the prefix used for training files based on the folder structure
    train_prefix = "mi" if "Braccio_Connect" in folder.split(os.sep) else "train"

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder):
        if filename.endswith('.edf'):
            # Extract the first word of the file name to determine its category
            first_word = filename.split('-')[0].lower()

            # Categorize files into training or test based on the first word
            if first_word == train_prefix:
                train_files.append(os.path.join(folder, filename))
            elif first_word == 'test':
                test_files_1.append(os.path.join(folder, filename))

    # Sort test files to ensure consistent ordering
    test_files_1 = sorted(test_files_1)

    """
    Distribute test files between test set 1 and test set 2 based on the number of test files that can be different 
    for Braccio connect dataset
    """

    if len(test_files_1) == 3:
        return [train_files, test_files_1]

    try:
        if len(test_files_1) == 5:
            test_files_2 = test_files_1[-2:]
            test_files_1 = test_files_1[:-2]
        elif len(test_files_1) == 10:
            test_files_2 = test_files_1[-4:]
            test_files_1 = test_files_1[:-4]
        else:
            raise ValueError(f"Unexpected number of test files found. Expected 5 or 10 test files, got {len(test_files_1)}")
    except Exception as e:
        print("Error in distributing test files:", e)
        raise

    return [train_files, test_files_1, test_files_2]


def dataset_creator(files, event_labels, tmin=1, tmax=4, sampling_freq=500):
    """
    Create datasets for Motor Imagery (MI) and Rest conditions based on event markers in EEG recordings.

    Parameters:
    - files (list of str): List of paths to the .edf files.
    - event_labels (list of str): List of event annotations of interest (e.g., 'OVTK_GDF_Left', 'OVTK_GDF_Right').
    - tmin (int): Start time in seconds relative to event onset.
    - tmax (int): End time in seconds relative to event onset.
    - sampling_freq (int): Sampling frequency of the EEG recordings.

    Returns:
    - tuple: Two numpy arrays containing the segmented data for MI and Rest conditions respectively.
    """

    mi_data = []
    rest_data = []

    # Convert time in seconds to sample indices
    tmin_samples, tmax_samples = int(tmin * sampling_freq), int(tmax * sampling_freq)

    for file in files:
        # Load the EEG data from the .edf file
        raw_data = mne.io.read_raw_edf(file, preload=True)
        data = raw_data.get_data()

        # Extract events from annotations
        events, event_id = mne.events_from_annotations(raw_data)

        print(event_id)

        # Find indices of events of interest
        event_indices = [event_id[label] for label in event_labels]
        event_positions = [i for i in range(events.shape[0]) if events[i, 2] in event_indices]

        # Extract the times and labels of the relevant events
        interest_events = events[event_positions, 0]
        interest_labels = events[event_positions, 2]

        # Remove events that are too close to each other
        close_event_indices = [i for i in range(1, len(interest_events)) if
                               interest_events[i] - interest_events[i - 1] <= 2]
        interest_events = np.delete(interest_events, close_event_indices)
        interest_labels = np.delete(interest_labels, close_event_indices)

        # Segment data based on event type
        for i in range(len(interest_events)):
            start_idx = interest_events[i] + tmin_samples
            end_idx = interest_events[i] + tmax_samples
            if interest_labels[i] == event_indices[0]:  # MI event
                mi_data.append(data[:, start_idx:end_idx])
            elif interest_labels[i] == event_indices[1]:  # Rest event
                rest_data.append(data[:, start_idx:end_idx])

    mi_data = np.array(mi_data)
    rest_data = np.array(rest_data)

    if mi_data.shape[0] < rest_data.shape[0]:
        diff = mi_data.shape[0] - rest_data.shape[0]
        rest_data = rest_data[:diff]

    elif rest_data.shape[0] < mi_data.shape[0]:
        diff = rest_data.shape[0] - mi_data.shape[0]
        mi_data = mi_data[:diff]

    return mi_data, rest_data
