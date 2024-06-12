import mne
import numpy as np
from Data_importation import edf_file_importation, dataset_creator
from Pre_processing import pass_band_filtering, CAR
from Features_extraction import PSD, Compute_Rsquare_Map_Welch, FC, imaginary_coherence, node_strength, NSdiff, FC_Fucon
from Visualisation import PSD_visualisation, FC_visualisation, R_squared_map_visualisation, Reorder_Rsquare, NS_visualisation
from Classification import LDA_train, LDA_test, mdm_train, mdm_test

print("importation done")


"""Variables"""

Hz = 500
tmin = 1
tmax = 4  # in seconds
lowLimit = 4
highLimit = 40
folder = "/Users/theo.coulon/data/sub-02/ses-01/"
#folder = "/Users/theo.coulon/data/Braccio_Connect/Sub02/PSD+NS"
event = ['OVTK_GDF_Left', 'OVTK_GDF_Right']
phases_names = ['Train', 'Test_1', 'Test_2']


"""data importation"""

phases = edf_file_importation(folder)
phases = dict(zip(phases_names, phases))
for phase_name, phase in phases.items():
    print('---------------------------------------------------')
    print('Current phase :', phase_name)
    MI, rest = dataset_creator(phase, event, tmin=tmin, tmax=tmax, freq=Hz)
    print('datasets shapes:', MI.shape, rest.shape)


    """data preprocessing"""


    filtered_MI = pass_band_filtering(MI, lowLimit, highLimit, Hz)
    filtered_rest = pass_band_filtering(rest, lowLimit, highLimit, Hz)

    car_MI = CAR(MI)
    car_rest = CAR(rest)


    """Features extraction"""

    freqs_MI, psd_MI = PSD(car_MI, Hz, lowLimit, highLimit)
    freqs_rest, psd_rest = PSD(car_rest, Hz, lowLimit, highLimit)
    R_squared_map = Compute_Rsquare_Map_Welch(psd_MI, psd_rest)
    print('Shape PSD', phase_name, ':', psd_MI.shape, psd_rest.shape, R_squared_map.shape)

    FC_MI = imaginary_coherence(filtered_MI, Hz, lowLimit, highLimit)
    FC_rest = imaginary_coherence(filtered_rest, Hz, lowLimit, highLimit)
    print('Shape FC', phase_name, ':', FC_MI.shape, FC_rest.shape)

    #np.savetxt("FC.txt", FC_MI[5, : , : , 13], fmt='%f')

    node_strength_MI = node_strength(FC_MI)
    node_strength_rest = node_strength(FC_rest)
    print('Shape NS', phase_name, ':', node_strength_MI.shape, node_strength_rest.shape)
    R_squared_map_NS = Compute_Rsquare_Map_Welch(node_strength_MI, node_strength_rest)
    node_strength_diff = NSdiff(node_strength_MI, node_strength_rest)


    """Classification"""

    raw = mne.io.read_raw_edf(phase[0], preload=False)
    channel_names = raw.info['ch_names']
    Ordered_PSD_MI = np.zeros_like(psd_MI)
    Ordered_PSD_rest = np.zeros_like(psd_rest)
    Ordered_NS_MI = np.zeros_like(node_strength_MI)
    Ordered_NS_rest = np.zeros_like(node_strength_rest)
    Ordered_R_squared_map, Ordered_channel_name = Reorder_Rsquare(R_squared_map, channel_names)
    Ordered_R_squared_map_NS, Ordered_channel_name_NS = Reorder_Rsquare(R_squared_map_NS, channel_names)

    for i in range(psd_MI.shape[0]):
        Ordered_PSD_MI[i], Ordered_channel_name_PSD = Reorder_Rsquare(psd_MI[i], channel_names)
        Ordered_PSD_rest[i], _ = Reorder_Rsquare(psd_rest[i], channel_names)
        Ordered_NS_MI[i], _ = Reorder_Rsquare(node_strength_MI[i], channel_names)
        Ordered_NS_rest[i], _ = Reorder_Rsquare(node_strength_rest[i], channel_names)

    electrodes = ['Fp1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'T7', 'C5',
                         'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO3', 'O1', 'Fpz',
                         'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Iz', 'Fp2', 'AF8', 'AF4', 'F8', 'F6',
                         'F4', 'F2', 'FT10', 'FT8', 'FC6', 'FC4', 'FC2', 'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4',
                         'CP2', 'P8', 'P6', 'P4', 'P2', 'PO8', 'PO4', 'O2']

    if (phase_name == 'Train'):
        lda_classifier, target_PSD = LDA_train(Ordered_PSD_MI, Ordered_PSD_rest, freqs_MI, freqs_rest,
                                           Ordered_R_squared_map, Ordered_channel_name, electrodes, phase_name,
                                           lowLimit, k_fold=False)

        lda_NS_classifier, target_NS = LDA_train(Ordered_NS_MI, Ordered_NS_rest, freqs_MI, freqs_rest,
                                           Ordered_R_squared_map_NS, Ordered_channel_name, electrodes, phase_name,
                                           lowLimit, k_fold=False)

        mdm, target_mdm = mdm_train(FC_MI, FC_rest,Ordered_R_squared_map_NS, freqs_MI,
                                                    Ordered_channel_name_NS, Ordered_NS_MI, Ordered_NS_rest,
                                                    node_strength_diff, electrodes, Hz, phase_name, lowLimit=lowLimit,
                                                    k_fold=False)

        previous_psd_MI = Ordered_PSD_MI
        previous_psd_rest = Ordered_PSD_rest
        previous_node_strength_MI = Ordered_NS_MI
        previous_node_strength_rest = Ordered_NS_rest


    elif (phase_name == 'Test_1'):

        lda_predictions, lda_accuracy, acc = LDA_test(Ordered_PSD_MI, Ordered_PSD_rest, previous_psd_MI,
                                                      previous_psd_rest, lda_classifier, target_PSD, electrodes)
        lda_predictions_NS, lda_accuracy_NS, acc_NS = LDA_test(Ordered_NS_MI, Ordered_NS_rest, previous_node_strength_MI,
                                                               previous_node_strength_rest, lda_NS_classifier, target_NS,
                                                               electrodes)
        mdm_report = mdm_test(FC_MI, FC_rest, mdm, target_mdm, Ordered_channel_name_NS)
        """print(f"Pipeline PSD - LDA results for phase {phase_name}: \n", lda_accuracy)
        print(f"Pipeline NS - LDA results for phase {phase_name}: \n", lda_accuracy_NS)
        print(f"Pipeline FC - MDM results for phase {phase_name}: \n", mdm_report)"""
        print("acc PSD : ", acc)
        print("acc NS : ", acc_NS)
        print("acc FC : ", mdm_report)

        lda_classifier, target_PSD = LDA_train(Ordered_PSD_MI, Ordered_PSD_rest, freqs_MI, freqs_rest,
                                           Ordered_R_squared_map, Ordered_channel_name, electrodes, phase_name,
                                           lowLimit, k_fold=False)

        lda_NS_classifier, target_NS = LDA_train(Ordered_NS_MI, Ordered_NS_rest, freqs_MI, freqs_rest,
                                           Ordered_R_squared_map_NS, Ordered_channel_name, electrodes, phase_name,
                                           lowLimit, k_fold=False)

        mdm, target_mdm = mdm_train(FC_MI, FC_rest, Ordered_R_squared_map_NS, freqs_MI,
                                    Ordered_channel_name_NS, Ordered_NS_MI, Ordered_NS_rest, node_strength_diff,
                                    electrodes, Hz, phase_name, lowLimit=lowLimit, k_fold=False)

        previous_psd_MI = Ordered_PSD_MI
        previous_psd_rest = Ordered_PSD_rest
        previous_node_strength_MI = Ordered_NS_MI
        previous_node_strength_rest = Ordered_NS_rest

    else :
        lda_predictions, lda_accuracy, acc = LDA_test(Ordered_PSD_MI, Ordered_PSD_rest, previous_psd_MI,
                                                      previous_psd_rest, lda_classifier, target_PSD, electrodes)
        lda_predictions_NS, lda_accuracy_NS, acc_NS = LDA_test(Ordered_NS_MI, Ordered_NS_rest,
                                                               previous_node_strength_MI,
                                                               previous_node_strength_rest, lda_NS_classifier,
                                                               target_NS,
                                                               electrodes)
        mdm_report = mdm_test(FC_MI, FC_rest, mdm, target_mdm, Ordered_channel_name_NS)
        """print(f"Pipeline PSD - LDA results for phase {phase_name}: \n", lda_accuracy)
        print(f"Pipeline NS - LDA results for phase {phase_name}: \n", lda_accuracy_NS)
        print(f"Pipeline FC - MDM results for phase {phase_name}: \n", mdm_report)"""
        print("acc PSD : ", acc)
        print("acc NS : ", acc_NS)
        print("acc FC : ", mdm_report)

