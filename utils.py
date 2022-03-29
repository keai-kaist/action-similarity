import numpy as np

def euclidean_distances(x: any, y: any) -> any:
    return np.linalg.norm(x - y, ord=2)
    #return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def manhattan_distances(x: any, y: any) -> any:
    return np.linalg.norm(x - y, ord=1)
    #return np.abs(np.subtract(x, y)).sum()

