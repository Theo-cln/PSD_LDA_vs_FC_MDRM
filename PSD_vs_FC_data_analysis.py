import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

#accuracies obtain from the Braccio dataset
data_sub2 = {
    'PSD-LDA': [0.58, 0.55, 1, 0.625, 0.58, 1],
    'FC-MDRM': [0.5, 0.825, 0.58, 0.75, 0.73, 0.75],
    'NS-LDA': [0.65, 0.525, 0.55, 0.575, 0.52, 0.6]
}

data_sub14 = {
    'PSD-LDA': [0.65, 0.6, 0.6, 0.7, 0.55, 0.675],
    'FC-MDRM': [0.52, 0.675, 0.58, 0.5, 0.85, 0.775],
    'NS-LDA': [0.58, 0.6, 0.5, 0.625, 0.5, 0.575]
}

#accuracies obtain from the Braccio_Connect dataset

data_sub02 = {
    'PSD-LDA': [0.64, 0.55, 0.72, 0.65],
    'FC-MDRM': [0.57, 0.53, 0.64, 0.8],
    'NS-LDA': [0.64, 0.53, 0.54, 0.68]
}

data_sub03 = {
    'PSD-LDA': [0.62, 0.725, 0.72, 0.825],
    'FC-MDRM': [0.68, 0.6, 0.85, 0.875],
    'NS-LDA': [0.6, 0.7, 0.7, 0.6]
}

data_sub04 = {
    'PSD-LDA': [0.5, 0.6, 1, 0.575],
    'FC-MDRM': [0.57, 0.6, 0.53, 0.53],
    'NS-LDA': [0.58, 0.775, 0.47, 0.98]
}

data_sub05 = {
    'PSD-LDA': [0.98, 0.625, 0.65, 0.525],
    'FC-MDRM': [0.8, 1, 0.9, 0.95],
    'NS-LDA': [0.58, 0.625, 0.53, 0.75]
}

data_sub06 = {
    'PSD-LDA': [0.83, 0.575, 0.7, 0.4],
    'FC-MDRM': [0.67, 0.88, 0.71, 0.675],
    'NS-LDA': [0.65, 0.7, 0.625, 0.575]
}

data_sub07 = {
    'PSD-LDA': [0.5, 0.7, 0.52, 0.6],
    'FC-MDRM': [0.67, 0.625, 0.52, 0.45],
    'NS-LDA': [0.66, 0.5, 0.57, 0.55]
}

data_sub08 = {
    'PSD-LDA': [1, 1, 0.97, 1],
    'FC-MDRM': [0.77, 0.77, 0.83, 0.725],
    'NS-LDA': [0.7, 0.7, 0.42, 0.7]
}

data_sub09 = {
    'PSD-LDA': [0.68, 0.725, 0.7, 0.725],
    'FC-MDRM': [0.57, 0.675, 0.55, 0.625],
    'NS-LDA': [0.5, 0.675, 0.5, 0.6]
}


#Accuracies for each subject
positions = np.arange(len(list(data_sub14.keys())))

name_sub = ["Sub 01", "Sub02", "Sub 03", "Sub 04", "Sub 05", "Sub 06", "Sub 07", "Sub 08", "Sub 09"]
for i, sub in enumerate(
        [data_sub02, data_sub03, data_sub04, data_sub05, data_sub06, data_sub07, data_sub08, data_sub09]):
    j = 0
    plt.figure(figsize=(10, 6))
    for methode, acc in sub.items():
        acc_D1 = [acc[k] for k in range(len(acc)) if k % 2 == 0]
        print(acc_D1)
        acc_D2 = [acc[k] for k in range(len(acc)) if k % 2 != 0]
        plt.boxplot(acc_D1, positions=[positions[j] - 0.2], widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='dodgerblue', color='blue'))
        plt.boxplot(acc_D2, positions=[positions[j] + 0.2], widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='darkgoldenrod', color='red'))
        j += 1
    plt.title(f'Accuracies comparison for the different classification methods ({name_sub[i]})')
    plt.xlabel('classification methods')
    plt.xticks(range(3), list(data_sub02.keys()))
    plt.ylabel('Accuracy')
    plt.legend(loc="upper right")

    plt.show()


# global accuracies over every subject
PSD= []
FC = []
NS = []
plt.figure(figsize=(10, 6))
for sub in [data_sub2, data_sub14, data_sub02, data_sub03, data_sub04, data_sub05, data_sub06, data_sub07, data_sub08, data_sub09] :
    PSD += sub['PSD-LDA']
    FC += sub['FC-MDRM']
    NS += sub['NS-LDA']
for i, acc in enumerate([PSD, FC, NS]):
    plt.boxplot(acc, positions=[positions[i]])
plt.title(f'Accuracies comparison for the different classification methods')
plt.xlabel('classification methods')
plt.xticks(range (3), list(data_sub02.keys()))
plt.ylabel('Accuracy')
plt.legend(loc="upper right")
plt.show()

# global accuracy over every subject (separation between Drive 1 (D1) and Drive 2 (D2))
PSD_D1 = []
PSD_D2 = []
FC_D1 = []
FC_D2 = []
NS_D1 = []
NS_D2 = []
plt.figure(figsize=(10, 6))
for sub in [data_sub2, data_sub14, data_sub02, data_sub03, data_sub04, data_sub05, data_sub06, data_sub07, data_sub08, data_sub09] :
    PSD_D1 += [sub['PSD-LDA'][k] for k in range(len(sub['PSD-LDA'])) if k%2==0]
    PSD_D2 += [sub['PSD-LDA'][k] for k in range(len(sub['PSD-LDA'])) if k%2!=0]
    FC_D1 += [sub['FC-MDRM'][k] for k in range(len(sub['FC-MDRM'])) if k%2==0]
    FC_D2 += [sub['FC-MDRM'][k] for k in range(len(sub['FC-MDRM'])) if k%2!=0]
    NS_D1 += [sub['NS-LDA'][k] for k in range(len(sub['NS-LDA'])) if k%2==0]
    NS_D2 += [sub['NS-LDA'][k] for k in range(len(sub['NS-LDA'])) if k%2!=0]
for i, acc in enumerate([PSD_D1, FC_D1, NS_D1]):
    box_A = plt.boxplot(acc, positions=[positions[i] - 0.2], widths=0.3, patch_artist=True, boxprops=dict(facecolor='dodgerblue', color='blue'))
for i, acc in enumerate([PSD_D2, FC_D2, NS_D2]) :
    box_B = plt.boxplot(acc, positions=[positions[i] + 0.2], widths=0.3, patch_artist=True, boxprops=dict(facecolor='darkgoldenrod', color='red'))
plt.title(f'Accuracies comparison for the different classification methods')
plt.xlabel('classification methods')
plt.xticks(range (3), list(data_sub02.keys()))
plt.ylabel('Accuracy')
plt.legend([box_A["boxes"][0], box_B["boxes"][0]], ['Drive 1', 'Drive 2'], loc='upper right')
plt.show()