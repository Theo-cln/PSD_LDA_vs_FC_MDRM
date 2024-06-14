import matplotlib.pyplot as plt
import mne
import numpy as np


def PSD_visualisation (Freqs_MI, Freqs_rest, Psd_mean_MI, Psd_mean_rest, electrode_name, phase_name):
    plt.figure(figsize=(10, 6))
    plt.plot(Freqs_MI, Psd_mean_MI, color='red', label='MI')
    plt.plot(Freqs_rest,Psd_mean_rest, color='blue', label='rest')
    plt.title(f'Average Power Spectral Density (PSD) of MI vs rest ({phase_name} {electrode_name})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectrum (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()


def FC_visualisation(FC_MI, FC_rest, frequency, channel_names, phase_name):
    FC_mean_MI = np.mean(FC_MI, axis=0)
    FC_mean_rest = np.mean(FC_rest, axis=0)
    """FC_mean_MI = FC_MI
    FC_mean_rest = FC_rest"""
    channels = range(len(channel_names))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    plot = axes[0].imshow(FC_mean_MI[:, :, frequency], cmap='bwr')
    axes[0].set_title(f'Functional connectivity MI {phase_name} {frequency} Hz')
    axes[0].set_xticks(channels, channel_names, rotation=90, fontsize=5)
    axes[0].set_yticks(channels, channel_names, fontsize=5)
    axes[1].imshow(FC_mean_rest[:, :, frequency], cmap='bwr')
    axes[1].set_title(f'Functional connectivity rest {phase_name} {frequency} Hz')
    axes[1].set_xticks(channels, channel_names, rotation=90, fontsize=5)
    axes[1].set_yticks(channels, channel_names, fontsize=5)
    plt.colorbar(plot, ax=axes, orientation='vertical')
    plt.show()


def R_squared_map_visualisation(R_squared_map, frequency, channel_name="", phase_name="", feature_name=""):
    channels = range(len(channel_name))
    frequency_length = range(len(frequency))
    plt.figure(figsize=(16, 16))
    plt.imshow(R_squared_map, cmap='jet', aspect='auto')
    plt.title(f'{phase_name} R_squared map of {feature_name}')
    if len(channel_name) > 0:
        plt.yticks(channels, channel_name)
    plt.xticks(frequency_length, frequency)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channels')
    plt.colorbar()
    plt.show()


def Reorder_Rsquare(Rsquare,electrodes_orig):
    if (len(electrodes_orig)>=64):
        electrodes_target = ['Fp1','AF7','AF3','F7','F5','F3','F1','FT9','FT7','FC5','FC3','FC1','T7','C5','C3','C1',
                             'TP7','CP5','CP3','CP1','P7','P5','P3','P1','PO7','PO3','O1','Fpz','AFz','Fz','FCz','Cz',
                             'CPz','Pz','POz','Oz','Iz','Fp2','AF8','AF4','F8','F6','F4','F2','FT10','FT8','FC6','FC4',
                             'FC2','T8','C6','C4','C2','TP8','CP6','CP4','CP2','P8','P6','P4','P2','PO8','PO4','O2']
    else:
        electrodes_target = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9'
                            ,'CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
    index_elec = []
    for k in range(len(electrodes_target)):
        for i in range(len(electrodes_orig)):
            if (electrodes_orig[i]==electrodes_target[k]):
                index_elec.append(i)
                break
    Rsquare_final = np.zeros([Rsquare.shape[0], Rsquare.shape[1]])
    electrode_test = []
    for l in range(len(index_elec)):
        electrode_test.append(index_elec[l])
        Rsquare_final[l, :] = Rsquare[index_elec[l], :]

    return Rsquare_final, electrodes_target




def NS_visualisation(freq, NS_MI, NS_Rest, channel, phase_name):
    plt.figure(figsize=(10, 6))
    plt.plot(freq, NS_MI, label='NS MI', color='red')
    plt.plot(freq, NS_Rest, label='NS Rest', color='blue')
    plt.title(f'Node strength of MI, rest for {phase_name} {channel}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Node strength')
    plt.legend()
    plt.grid(True)
    plt.show()



