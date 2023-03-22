import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
import pickle

# Define our filter variables
fs = 512                      # Hz; sampling rate
dt = 1000. / fs                 # ms; time between samples
sdt = dt#np.round(dt).astype(int); # rounded dt so that we can index samples
hp = 1                        # Hz; our low cut for our bandpass
lp = 24.                        # Hz; our high cut for our bandpass
num_taps = 31                   # Number of taps/coefficients of FIR filter

# Define ERP-related variables
epoch_start = 0    # ms
epoch_end = 800    # ms
baseline_start = 0 # ms
baseline_end = 100 # ms
erp_start = 200    # ms
erp_end = 600      # ms

# Let's translate these from time into index space to save time later
e_s = np.round(epoch_start / sdt).astype(int)     # epoch start
e_e = np.round(epoch_end / sdt).astype(int)       # epoch end
bl_s = np.round(baseline_start / sdt).astype(int) # baseline start
bl_e = np.round(baseline_end / sdt).astype(int)   # baseline end
erp_s = np.round(erp_start / sdt).astype(int)     # ERP component window start
erp_e = np.round(erp_end / sdt).astype(int)       # ERP component window end

def load_model(model_path):
    if model_path is None:
        class_weight = {-1: 0.5, 1: 2}
        clf = LinearDiscriminantAnalysis()
        return clf
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def train_and_test(
    X_train_pre_,
    y_train_pre_,
    X_train,
    y_train,
    X_test
):
    clf = load_model(None)
    
    X_train_pre_len = 0
    if X_train_pre_ is not None:
        X_train_pre = X_train_pre_.copy()
        y_train_pre = y_train_pre_.copy()
        X_train_pre_len = X_train_pre.shape[0]
    else :
        X_train_pre = None
        y_train_pre = None

    
    # preprocess
    def preprocess(x):
        for i in range(x.shape[0]):
            # correct DC offset of signal
            x[i] = x[i] - np.mean(x[i], axis=1).reshape(-1, 1)
            # devide sd from every channel
            sd_every_chan = np.std(x[i], axis=1).reshape(-1, 1)
            x[i] = x[i] / sd_every_chan
            # baseline correction
            x[i] = x[i] - np.mean(x[i][bl_s:bl_e], axis=0)
    preprocess(X_train)
    preprocess(X_test)
    if X_train_pre_len > 0:
        preprocess(X_train_pre)

    # downsample
    num_points = 6; # we will divide our window into num_points means
    # Define a simple windowed means function
    def wm(x, start, end, num_points):
        num_trials = x.shape[0] # assumes first dem is numb observations
        num_chans = x.shape[1] # assumes last dim is num channels
        len_time = x.shape[2] # assumes second dim is time
        w = np.round((end-start)/num_points).astype(int)
        y = np.zeros((num_trials, num_chans, num_points))
        for i in range(0, num_points):
            s = start + (w * i)
            e = s + w
            if e > len_time:
                e = len_time
            y[:,:,i] = np.mean(x[:,:,s:e], axis=2)
        return y

    X_train = wm(X_train, erp_s, erp_e, num_points)
    X_test = wm(X_test, erp_s, erp_e, num_points)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    if X_train_pre_len > 0:
        X_train_pre = wm(X_train_pre, erp_s, erp_e, num_points)
        X_train_pre = X_train_pre.reshape(X_train_pre.shape[0], -1)
    
    max_sample = 1500
    if X_train.shape[0] > max_sample:
        X_train, y_train = shuffle(X_train, y_train)
        X_train = X_train[:max_sample-1]
        y_train = y_train[:max_sample-1]
    if X_train_pre_len + X_train.shape[0] > max_sample and X_train_pre_len > 0:
        X_train_pre, y_train_pre = shuffle(X_train_pre, y_train_pre)
        X_train_pre = X_train_pre[:max_sample - X_train.shape[0]]
        y_train_pre = y_train_pre[:max_sample - X_train.shape[0]]
    if X_train_pre_len > 0:
        X_train = np.concatenate((X_train_pre, X_train), axis=0)
        y_train = np.concatenate((y_train_pre, y_train), axis=0)
    
    # predict
    clf.fit(X_train, y_train)
    return clf.predict(X_test)