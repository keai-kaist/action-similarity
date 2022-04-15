import os
import pickle

import numpy as np

def euclidean_distances(x: any, y: any) -> any:
    return np.linalg.norm(x - y, ord=2)
    #return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def manhattan_distances(x: any, y: any) -> any:
    return np.linalg.norm(x - y, ord=1)
    #return np.abs(np.subtract(x, y)).sum()

def cache_file(file_name: str, func, *args, **kargs):
    file_name = file_name.rstrip(".mp4") + ".pickle"
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)
    else:
        data = func(*args, **kargs)
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    return data
