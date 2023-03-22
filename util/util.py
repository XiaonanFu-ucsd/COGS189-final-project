default_sample_rate = 150.0

def getSamplingRate(timestamps):
    """Return the sampling rate of the timestamps array.
    """
    return round((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))

import scipy.signal as signal
import numpy as np

def to512hz(eeg_points, timestamps):
    """Return the 512Hz version of the EEG points.
    """
    target_sampling_rate = 512.0
    sampling_rate = getSamplingRate(timestamps)
    if sampling_rate == target_sampling_rate:
        return (eeg_points, timestamps)
    target_data_length = int(round(len(eeg_points) * target_sampling_rate / sampling_rate))
    target_timestamps = np.linspace(timestamps[0], timestamps[-1], target_data_length)
    return (signal.resample(eeg_points, target_data_length), target_timestamps)

def to512hz_label(labels, timestamps):
    """Return the 512Hz version of the labels.
    """
    target_sampling_rate = 512.0
    sampling_rate = getSamplingRate(timestamps)
    if sampling_rate == target_sampling_rate:
        return (labels, timestamps)
    # target_data_length = int(round(len(labels) * target_sampling_rate / sampling_rate))
    # target_timestamps = np.linspace(timestamps[0], timestamps[-1], target_data_length)
    # target_labels = [0] * target_data_length
    # for y, t in zip(labels, timestamps):
    #     if y != 0:
    
    return (target_labels, target_timestamps)

def resample_x(eegData, rate=None, timestamps=None, target_rate=default_sample_rate):
    if rate is None and timestamps is None:
        raise ValueError("Either rate or timestamps should be provided.")
    if target_rate != None and target_rate < 0:
        return eegData
    rate = rate if rate!=None else getSamplingRate(timestamps)
    
    if rate == target_rate:
        return eegData
    target_data_length = int(round(len(eegData) * target_rate / rate))
    return signal.resample(eegData, target_data_length)


def resample_y(y, rate=None, timestamps=None, target_rate=default_sample_rate, ignoredValue=0):
    if rate is None and timestamps is None:
        raise ValueError("Either rate or timestamps should be provided.")
    if target_rate != None and target_rate < 0:
        return y
    rate = rate if rate!=None else getSamplingRate(timestamps)
    if rate == target_rate:
        return y
    target_data_length = int(round(len(y) * target_rate / rate))
    ret = np.zeros(shape=(target_data_length,))
    y_index_hasValue = [i for i in range(len(y)) if y[i] != ignoredValue]
    for i in y_index_hasValue:
        ret[int(i * (target_rate / rate))] = y[i]
    return ret
    

def getWindow(randIndex, eeg_points, timestamps, label, length=default_sample_rate * 0.8):
    """Return a random window of the EEG points.
    Assume the sampling rate is 512Hz.
    Default window size is 1 second.
    """
    if randIndex + length >= len(eeg_points):
        return None
    return (eeg_points[randIndex:randIndex+length], 
            timestamps[randIndex:randIndex+length],
            isP300(randIndex, default_sample_rate, label)
    )

def isP300(startIndex, length, label, p300_label=1):
    """Return whether the label is P300 or not.
    if is a P300, return 1, otherwise return 0.
    Assume the sampling rate is 512Hz.
    Assume P300 starting is possible from 250ms to 750ms. 
    Assume P300 effective time period is 50ms to 800ms after the stimulus.
    No later than 300ms after the stimulus.
    The end should not earlier than 425 ms
    """
    s = startIndex + length - int(425 / 1000 * default_sample_rate)
    e = startIndex - int(400 / 1000 * default_sample_rate)
    print(s, e)
    
    for i in range(e,s):
        if label[i] == p300_label:
            return 1
    return 0



def epoch_wrt_event(x, y, epoch_start_i, epoch_end_i, y_ignoredValue=0):
    """if the epoch_start_i is 0, it means the epoch starts at the event.
        If one epoch is not completed, this epoch will be ignored.
    """
    onsets_i = np.where(y != y_ignoredValue)[0]
    ret_x = []
    ret_y = []
    for i in onsets_i:
        if i + epoch_end_i >= len(y):
            continue
        if i + epoch_start_i < 0:
            continue
        ret_x.append(x[i+epoch_start_i:i+epoch_end_i])
        ret_y.append(y[i])
    return np.array(ret_x), np.array(ret_y)

def epoch_wrt_event_chanFirst(x, y, epoch_start_i, epoch_end_i, y_ignoredValue=0):
    """if the epoch_start_i is 0, it means the epoch starts at the event.
        If one epoch is not completed, this epoch will be ignored.
    """
    onsets_i = np.where(y != y_ignoredValue)[0]
    ret_x = []
    ret_y = []
    for i in onsets_i:
        if i + epoch_end_i >= len(y):
            continue
        if i + epoch_start_i < 0:
            continue
        ret_x.append(x[:, i+epoch_start_i:i+epoch_end_i])
        ret_y.append(y[i])
    return np.array(ret_x), np.array(ret_y)

def countLine_hasNull(x):
    ret = 0
    for i in range(x.shape[0]):
        if np.isnan(x[i]).any():
            ret += 1
    return ret