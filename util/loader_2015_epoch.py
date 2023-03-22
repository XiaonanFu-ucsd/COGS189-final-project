import numpy as np
import pickle

def load(a, b):
    """
    return a list of epoch data, from subject a to subject b, excluding b
    """
    ret_x = None
    ret_y = np.array([])
    
    for i in range(a, b):
        x_path = f"./datasets/2015pre/x{i}.pkl"
        y_path = f"./datasets/2015pre/y{i}.pkl"
        tmp_x = pickle.load(open(x_path, "rb"))
        tmp_y = pickle.load(open(y_path, "rb"))
        ret_x = tmp_x if ret_x is None else np.concatenate((ret_x, tmp_x), axis=0)
        ret_y = np.concatenate((ret_y, tmp_y), axis=0)
    
    return ret_x, ret_y
        