import mne
import numpy as np
from Data_importation import edf_file_importation, dataset_creator
from Pre_processing import pass_band_filtering, CAR
from Features_extraction import PSD, compute_rsquare_map_welch, FC, imaginary_coherence, node_strength, NSdiff
from Visualisation import reorder_r_squared
from Classification import train_lda, test_lda, train_mdrm, test_mdrm

print("All modules imported successfully.")

# Constants and parameters
SAMPLING_FREQ = 500  # Hz
TIME_WINDOW_START = 1  # seconds
TIME_WINDOW_END = 4  # seconds
LOW_CUTOFF_FREQ = 4  # Hz
HIGH_CUTOFF_FREQ = 40  # Hz

# File paths
BRACCIO_DATA_PATH = "/Users/theo.coulon/data/sub-02/ses-03/"
BRACCIO_CONNECT_DATA_PATH = "/Users/theo.coulon/data/Braccio_Connect/Sub06/PSD"
EVENT_LABELS = ['OVTK_GDF_Left', 'OVTK_GDF_Right']
PHASE_NAMES = ['Train', 'Test_1', 'Test_2']

# Data Importation
phases_data = edf_file_importation(BRACCIO_CONNECT_DATA_PATH)
phases_data = dict(zip(PHASE_NAMES, phases_data))

# Process each phase
for phase_name, phase_data in phases_data.items():
    print(f'Processing phase: {phase_name}')

    # Create datasets for Motor Imagery (MI) and Rest
    MI_data, rest_data = dataset_creator(phase_data, EVENT_LABELS, tmin=TIME_WINDOW_START, tmax=TIME_WINDOW_END,
                                         sampling_freq=SAMPLING_FREQ)
    print(f'Dataset shapes: MI: {MI_data.shape}, Rest: {rest_data.shape}')

    # Preprocess data: Band-pass filtering and Common Average Referencing (CAR)
    filtered_MI_data = pass_band_filtering(MI_data, LOW_CUTOFF_FREQ, HIGH_CUTOFF_FREQ, SAMPLING_FREQ)
    filtered_rest_data = pass_band_filtering(rest_data, LOW_CUTOFF_FREQ, HIGH_CUTOFF_FREQ, SAMPLING_FREQ)

    car_MI_data = CAR(filtered_MI_data)
    car_rest_data = CAR(filtered_rest_data)

    # PSD Extraction
    freqs_MI, psd_MI = PSD(car_MI_data, SAMPLING_FREQ, LOW_CUTOFF_FREQ,
                                                              HIGH_CUTOFF_FREQ)
    freqs_rest, psd_rest = PSD(car_rest_data, SAMPLING_FREQ, LOW_CUTOFF_FREQ, HIGH_CUTOFF_FREQ)
    R_squared_map = compute_rsquare_map_welch(psd_MI, psd_rest)
    print(f'PSD Shape for {phase_name}: MI: {psd_MI.shape}, Rest: {psd_rest.shape}, R-squared: {R_squared_map.shape}')

    # Functional Connectivity (FC) Extraction
    FC_MI = FC(filtered_MI_data, SAMPLING_FREQ, LOW_CUTOFF_FREQ, HIGH_CUTOFF_FREQ)
    FC_rest = FC(filtered_rest_data, SAMPLING_FREQ, LOW_CUTOFF_FREQ, HIGH_CUTOFF_FREQ)
    print(f'FC Shape for {phase_name}: MI: {FC_MI.shape}, Rest: {FC_rest.shape}')

    # Node Strength Calculation
    node_strength_MI = node_strength(FC_MI)
    node_strength_rest = node_strength(FC_rest)
    R_squared_map_NS = compute_rsquare_map_welch(node_strength_MI, node_strength_rest, feature_name="NS")
    node_strength_diff = NSdiff(node_strength_MI, node_strength_rest)

    # Prepare for Classification
    raw_data = mne.io.read_raw_edf(phase_data[0], preload=False)
    channel_names = raw_data.info['ch_names']

    # Reorder features based on R-squared map
    Ordered_PSD_MI = np.array([reorder_r_squared(psd_MI[i], channel_names)[0] for i in range(psd_MI.shape[0])])
    Ordered_PSD_rest = np.array([reorder_r_squared(psd_rest[i], channel_names)[0] for i in range(psd_rest.shape[0])])
    Ordered_NS_MI = np.array([reorder_r_squared(node_strength_MI[i], channel_names)[0] for i in
                       range(node_strength_MI.shape[0])])
    Ordered_NS_rest = np.array([reorder_r_squared(node_strength_rest[i], channel_names)[0] for i in
                       range(node_strength_rest.shape[0])])

    Ordered_R_squared_map, Ordered_channel_names = reorder_r_squared(R_squared_map, channel_names)
    Ordered_R_squared_map_NS, Ordered_channel_names_NS = reorder_r_squared(R_squared_map_NS, channel_names)

    # List of electrode names
    electrodes = [
        'Fp1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'T7', 'C5',
        'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO3', 'O1', 'Fpz',
        'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Iz', 'Fp2', 'AF8', 'AF4', 'F8', 'F6',
        'F4', 'F2', 'FT10', 'FT8', 'FC6', 'FC4', 'FC2', 'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4',
        'CP2', 'P8', 'P6', 'P4', 'P2', 'PO8', 'PO4', 'O2'
    ]

    print(Ordered_PSD_MI.shape)
    print(Ordered_PSD_rest.shape)

    # Classification Process
    if phase_name == 'Train':
        # Train LDA and MDRM classifiers on calibration data
        lda_classifier, target_PSD = train_lda(Ordered_PSD_MI, Ordered_PSD_rest, freqs_MI, freqs_rest,
                                               Ordered_R_squared_map, Ordered_channel_names, electrodes, phase_name,
                                               LOW_CUTOFF_FREQ, use_k_fold=False)
        lda_NS_classifier, target_NS = train_lda(Ordered_NS_MI, Ordered_NS_rest, freqs_MI, freqs_rest,
                                                 Ordered_R_squared_map_NS, Ordered_channel_names_NS, electrodes,
                                                 phase_name, LOW_CUTOFF_FREQ, use_k_fold=False, feature_name="NS")
        mdrm_classifier, target_mdrm = train_mdrm(FC_MI, FC_rest, Ordered_R_squared_map_NS, freqs_MI,
                                               Ordered_channel_names_NS, node_strength_diff, electrodes, SAMPLING_FREQ,
                                               phase_name,LOW_CUTOFF_FREQ, use_k_fold=False)


        # Save the previous phase data for subsequent testing
        previous_psd_MI = Ordered_PSD_MI
        previous_psd_rest = Ordered_PSD_rest
        previous_node_strength_MI = Ordered_NS_MI
        previous_node_strength_rest = Ordered_NS_rest

    elif phase_name in ['Test_1', 'Test_2']:
        # Test the classifiers on testing data
        lda_predictions, lda_report, lda_acc = test_lda(Ordered_PSD_MI, Ordered_PSD_rest, previous_psd_MI,
                                                          previous_psd_rest, lda_classifier, target_PSD, electrodes)
        lda_predictions_NS, lda_NS_report, lda_acc_NS = test_lda(Ordered_NS_MI, Ordered_NS_rest,
                                                                   previous_node_strength_MI,
                                                                   previous_node_strength_rest, lda_NS_classifier,
                                                                   target_NS, electrodes)
        mdrm_report, mdrm_acc = test_mdrm(FC_MI, FC_rest, mdrm_classifier, target_mdrm, Ordered_channel_names_NS)

        print(f"Phase {phase_name} results:")
        print(f" - LDA (PSD) accuracy: {lda_predictions}")
        print(f" - LDA (NS) accuracy: {lda_predictions_NS}")
        print(f" - MDRM accuracy: {mdrm_report}")
        print(f" - LDA (PSD) report : \n {lda_acc}")
        print(f" - LDA (NS) accuracy: \n {lda_acc_NS}")
        print(f" - MDRM accuracy: \n {mdrm_acc}")

        if phase_name == 'Test_1':
            # Re-train the classifiers with current test data
            lda_classifier, target_PSD = train_lda(Ordered_PSD_MI, Ordered_PSD_rest, freqs_MI, freqs_rest,
                                                   Ordered_R_squared_map, Ordered_channel_names, electrodes, phase_name,
                                                   LOW_CUTOFF_FREQ, lda_classifier, use_k_fold=False)
            lda_NS_classifier, target_NS = train_lda(Ordered_NS_MI, Ordered_NS_rest, freqs_MI, freqs_rest,
                                                     Ordered_R_squared_map_NS, Ordered_channel_names_NS, electrodes,
                                                     phase_name, LOW_CUTOFF_FREQ, lda_NS_classifier, use_k_fold=False,
                                                     feature_name="NS")
            mdrm_classifier, target_mdrm = train_mdrm(FC_MI, FC_rest, Ordered_R_squared_map_NS, freqs_MI,
                                                   Ordered_channel_names_NS, node_strength_diff, electrodes,
                                                   SAMPLING_FREQ, phase_name, LOW_CUTOFF_FREQ, mdrm_classifier,
                                                   use_k_fold=False)

            # Update previous phase data for the next test
            previous_psd_MI = Ordered_PSD_MI
            previous_psd_rest = Ordered_PSD_rest
            previous_node_strength_MI = Ordered_NS_MI
            previous_node_strength_rest = Ordered_NS_rest
