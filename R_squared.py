#import numpy as np

def Compute_Rsquare_Map_Welch(Power_of_trials_1,Power_of_trials_2):
    b = Power_of_trials_1.shape
    a = Power_of_trials_2.shape

    #print(a)
    #print(b)
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

            #correlation_xy = correlation_matrix[0,1]
            #Rsquare_tab[k,l] = correlation_xy**2
            Rsquare_tab[k,l] = (Sum_q**2/n1+Sum_r**2/n2-G)/(sumsqu1+sumsqu2-G)

    return Rsquare_tab




def load_file_eeg(sample_data_folder,filename):
    sample_Training_EDF = os.path.join(sample_data_folder, filename)
    raw_Training_EEG = mne.io.read_raw_nihon(sample_Training_EDF, preload=True, verbose=False)
    events_from_annot_1,event_id_1 = mne.events_from_annotations(raw_Training_EEG,event_id='auto')
    return raw_Training_EEG, events_from_annot_1,event_id_1




def select_Event(event_name,RAW_data,events_from_annot,event_id,t_min,t_max,number_electrodes):

    epochs_training = mne.Epochs(RAW_data, events_from_annot, event_id,tmin=t_min, tmax=t_max,preload=True,event_repeated='merge',baseline = None,picks = np.arange(0,number_electrodes))

    #epochs_training = mne.Epochs(RAW_data, events_from_annot, event_id,tmin = t_min, tmax=t_max,preload=True,event_repeated='merge')
    return epochs_training[event_name]




def Reorder_Rsquare(Rsquare,electrodes_orig):
    if (len(electrodes_orig)>=64):

        electrodes_target = ['Fp1','AF7','AF3','F7','F5','F3','F1','FT9','FT7','FC5','FC3','FC1','T7','C5','C3','C1','TP7','CP5','CP3','CP1','P7','P5','P3','P1','PO7','PO3','O1','Fpz','AFz','Fz','FCz','Cz','CPz','Pz','POz','Oz','Iz','Fp2','AF8','AF4','F8','F6','F4','F2','FT10','FT8','FC6','FC4','FC2','T8','C6','C4','C2','TP8','CP6','CP4','CP2','P8','P6','P4','P2','PO8','PO4','O2']
    else:
        electrodes_target = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
    index_elec = []
    electrod_final=[]
    for k in range(len(electrodes_target)):
        for i in range(len(electrodes_orig)):
            if (electrodes_orig[i]==electrodes_target[k]):
                index_elec.append(i)
                break

    print(index_elec)

    Rsquare_final = np.zeros([Rsquare.shape[0], Rsquare.shape[1]])

    electrode_test = []
    for l in range(len(index_elec)):
        electrode_test.append(index_elec[l])
        Rsquare_final[l, :] = Rsquare[index_elec[l], :]

    return Rsquare_final, electrodes_target
    electrodes = channel_generator(64, 'TP9', 'TP10')




def channel_generator(number_of_channel, Ground, Ref):
    global index_gnd, index_ref
    if number_of_channel == 32:
        electrodes = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
        for i in range(len(electrodes)):
            if (electrodes[i] == Ground):
                index_gnd = i
            if (electrodes[i] == Ref):
                index_ref = i
        electrodes[index_gnd] = 'AFz'
        electrodes[index_ref] = 'FCz'

    elif number_of_channel == 64:
        #electrodes = ['FP1','FP2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10','C5','C1','C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7','PO3','POz','PO4','PO8']
        electrodes = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','Iz']
        for i in range(len(electrodes)):
            if (electrodes[i] == Ground):
                index_gnd = i
            if (electrodes[i] == Ref):
                index_ref = i
        electrodes[index_gnd] = 'Fpz'
        electrodes[index_ref] = 'FCz'

    return electrodes




def plot_Rsquare_calcul_welch(Rsquare,channel_array,freq,fmin,fmax):
    fig,ax = plt.subplots(figsize=(15, 15))
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }
    frequence = []
    index_fmin = 0
    index_fmax = 40
    for i in range(len(freq)):
        if freq[i]==fmin:
            index_fmin = i

    for i in range(len(freq)):
        if freq[i]==fmax:
            index_fmax = i
    Rsquare_reshape = Rsquare[0:64,index_fmin:index_fmax+1]


    plt.imshow(Rsquare_reshape,cmap='jet',aspect='auto')
    cm.get_cmap('jet')
    #plt.jet()
    cbar = plt.colorbar()
    cbar.set_label('R^2', rotation=270,labelpad = 10,fontsize = 20)
    cbar.ax.tick_params(labelsize=20)
    plt.yticks(range(len(channel_array)),channel_array)
    plt.xlabel('Frequency (Hz)', fontdict=font)
    plt.ylabel('Sensors', fontdict=font)