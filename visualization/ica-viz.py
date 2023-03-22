import numpy as np                                      
import matplotlib.pyplot as plt                         
import matplotlib.patches as patches
import seaborn as sns
import scipy.signal as signal 
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import FastICA
import mne
from mne.preprocessing import (ICA, corrmap, create_ecg_epochs,
                               create_eog_epochs)


# get sample data
df = pd.read_csv('./d2.csv', header=None)
#df = pd.read_csv('./datasets/bi2014b/group_01_sujet_01.csv', header=None)
X_train = df.iloc[:, 1:33].values
print(X_train.shape)


chnames = ['Fp1',
            'Fp2',
            'AFz',
            'F7',
            'F3',
            'F4',
            'F8',
            'FC5',
            'FC1',
            'FC2',
            'FC6',
            'T7',
            'C3',
            'Cz',
            'C4',
            'T8',
            'CP5',
            'CP1',
            'CP2',
            'CP6',
            'P7',
            'P3',
            'Pz',
            'P4',
            'P8',
            'PO7',
            'O1',
            'Oz',
            'O2',
            'PO8',
            'PO9',
            'PO10']
            #'STI 014']

chtypes = ['eeg'] * 32
print(chtypes)
print(chnames)
info = mne.create_info(ch_names=chnames, sfreq=512,
                        ch_types=chtypes)
m = mne.channels.make_standard_montage('standard_1020')

#headerNames = pd.read_csv('../datasets/bi2015a/Header.csv', header=None)
#headerNames = list(headerNames.iloc[0])
X_train = mne.io.RawArray(X_train.T, info)
X_train.set_montage(m)
#X_train.filter(0.5, 30, fir_design='firwin')
#X_train.plot_sensors(ch_type='eeg', kind='topomap', show_names=True)
#X_train.ch_names
X_train = X_train.get_data(start=5000, stop=5000+512*30)
X_train = mne.io.RawArray(X_train, info)
X_train.set_montage(m)

mne.viz.plot_raw(X_train, scalings='auto', title='Data from arrays',)
plt.show()

X_train.filter(1, 24)

mne.viz.plot_raw(X_train, scalings='auto', title='Data from arrays',)
plt.show()

X_train_filted = X_train.copy()
#X_train_filted.filter(1, None, fir_design='firwin')
n_cmp = 12
ica = ICA(n_components=n_cmp, max_iter='auto', random_state=42)#, method='infomax')
ica.fit(X_train_filted)
ica.plot_components()
plt.show()

for i in range(n_cmp): 
    explained_var_ratio = ica.get_explained_variance_ratio(
        X_train_filted,
        components=[i],
        ch_type='eeg'
    )
    # This time, print as percentage.
    ratio_percent = round(100 * explained_var_ratio['eeg'])
    print(
        f'{i}  Fraction of variance in EEG signal explained by first component: '
        f'{ratio_percent}%'
    )

ica.plot_sources(X_train_filted)
plt.show()

ica.exclude = [0, 1, 2, 4, 6, 8]
#ica.exclude = range(0,10)
ica.apply(X_train_filted)
mne.viz.plot_raw(X_train_filted, scalings='auto', title='Data from arrays',)
plt.show()