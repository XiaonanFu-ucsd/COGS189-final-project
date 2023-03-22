import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
import pickle



def train(
    model_output_path: str,
    report_output_path: str,
    X_train_: np.ndarray,
    y_train_: np.ndarray
):
    X_train = X_train_.copy()
    y_train = y_train_.copy()
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
    erp_end = 800      # ms

    # Let's translate these from time into index space to save time later
    e_s = np.round(epoch_start / sdt).astype(int)     # epoch start
    e_e = np.round(epoch_end / sdt).astype(int)       # epoch end
    bl_s = np.round(baseline_start / sdt).astype(int) # baseline start
    bl_e = np.round(baseline_end / sdt).astype(int)   # baseline end
    erp_s = np.round(erp_start / sdt).astype(int)     # ERP component window start
    erp_e = np.round(erp_end / sdt).astype(int)       # ERP component window end
    
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
    
    # downsample
    num_points = 18; # we will divide our window into num_points means
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
    X_train = X_train.reshape(X_train.shape[0], -1)
    
    class_weight = {-1: 0.5, 1: 20}
    clf = RandomForestClassifier(n_estimators=75, max_depth=75, n_jobs=-1, class_weight=class_weight)
    clf.fit(X_train, y_train)
    
    # save model
    with open(model_output_path, 'wb') as f:
        pickle.dump(clf, f)
    
    # save report
    pred = clf.predict(X_train)
    report = "tn, fp, fn, tp: " + "  ".join([str(i) for i in confusion_matrix(y_train, pred).ravel().tolist()]) + "\n"
    report += "f1 score: " + f1_score(y_train, pred).astype(str) + "\n"
    report += "accuracy: " + clf.score(X_train, y_train).astype(str) + "\n"
    report += classification_report(y_train, pred)
    with open(report_output_path, 'w') as f:
        f.write(report)