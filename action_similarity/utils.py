import os
import pickle
import re
from typing import List
from action_similarity.dtw import accelerated_dtw
import numpy as np

def parse_action_label(action_label):
    actions = {}
    with open(action_label) as f:
        lines = f.readlines()
        for line in lines:
            no, action = line.split(None, maxsplit=1)
            no = int(re.search(r'\d+', no).group())
            actions[no] = action.strip()
    return actions

def seq_feature_to_motion_embedding(
    seq_features: List[List[np.ndarray]]) -> List[np.ndarray]:
    # input value:  #windows x #body part x #features
    # output value: #body part x #windows x #features
    seq_features_np = np.array(seq_features, dtype=np.ndarray) # #windows x 5, dtype = ndarray
    seq_features_t = seq_features_np.transpose() # 5 x #windows, dtype = ndarray
    motion_embedding = []
    for i in range(len(seq_features_t)):
        motion_embedding.append(np.stack(seq_features_t[i], axis=0))
    return motion_embedding # #body part x #windows x #features

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

def time_align(seq1: List[np.ndarray], seq2: List[np.ndarray]):
    _, _, _, path = accelerated_dtw(seq1, seq2)
    path1, path2 = path[0], path[1]
    aligned_seq = []
    last_idx = -1
    for i, j in zip(path1, path2):
        if last_idx != i:
            aligned_seq.append(seq2[j])
            last_idx = i
    return seq1, aligned_seq

class Timer():
    def __init__(self):
        self.times: List = []
        #self.end_times: List = []
        self.tags: List = []
        self.index = 0

    def log(self, tag = None):
        self.times.append(time.time())
        if tag is None:
            self.tags.append(self.index)
            self.index += 1
        else:
            self.tags.append(tag)

    def pprint(self):
        for tag, start, end in zip(self.tags[:-1], self.times[:-1], self.times[1:]):
            print(f"[{tag}] time elasp: {(end-start):.3f}")
        print(f"[Total] time elasp: {(self.times[-1] - self.times[0]):.3f}")
    
    def info(self):
        times = []
        for tag, start, end in zip(self.tags[:-1], self.times[:-1], self.times[1:]):
            times.append((tag, end-start))
        return times