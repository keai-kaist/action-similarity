import os
import re
import pickle
import time
from typing import List, Dict
from pathlib import Path

import numpy as np
from glob import glob

from bpe import Config

from action_similarity.dtw import accelerated_dtw

def parse_action_label(action_label):
    actions = {}
    with open(action_label, encoding='utf-8') as f:
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

def cache_file(file_name: str, func, *args, **kwargs):
    base_name = Path(".cache") / (os.path.splitext(file_name)[0] + ".pickle")
    
    # 디렉토리 생성
    head, _ = os.path.split(base_name)
    Path(head).mkdir(parents=True, exist_ok=True)
    #breakpoint()
    
    # cache pickle 생성 또는 불러오기
    if os.path.exists(base_name):
        #print("load from", file_name)
        with open(base_name, "rb") as f:
            data = pickle.load(f)
    else:
        data = func(*args, **kwargs)
        with open(base_name, "wb") as f:
            pickle.dump(data, f)
    return data

def save_file(file_name: str, func, *args, **kwargs):
    file_name = os.path.splitext(file_name)[0] + ".pickle"
    data = func(*args, **kwargs)
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

def save_embeddings(db: Dict, config: Config, embeddings_dir = "embeddings"):
    k_clusters = config.k_clusters if not config is None and config.clustering else 0
    # embeddings_dir = Path(config.data_dir) / embeddings_dir
    embeddings_dir = Path(embeddings_dir)
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)

    for action_idx, seq_features in db.items():
        embeddings_filename = embeddings_dir / f'action_embeddings_{action_idx:03d}.pickle'
        # pickle 파일이 이미 있는 경우
        if os.path.exists(embeddings_filename):
            with open(embeddings_filename, 'rb') as f:
                seq_features_by_k = pickle.load(f) # dictionary
                seq_features_by_k[k_clusters] = seq_features
        # pickle 파일이 없는 경우 dict로 생성
        else:
            seq_features_by_k = {k_clusters: seq_features}
            
        with open(embeddings_filename, 'wb') as f:
            pickle.dump(seq_features_by_k, f)

    with open(embeddings_dir / "readme.md", 'a') as f:
        f.write(f"K clusters of saved embeddings: {config.k_clusters}\n")
            
def load_embeddings(config: Config, embeddings_dir = "embeddings", target_actions=None):
    k_clusters = config.k_clusters if not config.k_clusters is None and config.clustering else 0
    # embeddings_dir = Path(config.data_dir) / embeddings_dir
    embeddings_dir = Path(embeddings_dir)
    std_db = {}
    for embedding_file in glob(f'{embeddings_dir}/*'):
        if not embedding_file.endswith(".pickle"):
            continue
        with open(embedding_file, 'rb') as f:
            # seq_features.shape == (#videos, #windows, 5, 128[0:4] or 256[4])
            # seq_features: List[List[List[np.ndarray]]]
            # 64 * (T=16 / 8), 128 * (T=16 / 8)
            seq_features_per_k = pickle.load(f)
            file_name = os.path.basename(embedding_file).rstrip(".pickle")
            action_idx = int(file_name.split("_")[-1])
            if target_actions is None or action_idx in target_actions:
                std_db[action_idx] = seq_features_per_k[k_clusters]
    return std_db 

def exist_embeddings(config: Config = None, embeddings_dir = "embeddings"):
    k_clusters = config.k_clusters if not config is None and config.clustering else 0
    # embeddings_dir = Path(config.data_dir) / embeddings_dir
    embeddings_dir = Path(embeddings_dir)
    exist_flags = []
    for embedding_file in glob(f'{embeddings_dir}/*'):
        if not embedding_file.endswith(".pickle"):
            continue
        with open(embedding_file, 'rb') as f:
            # seq_features.shape == (#videos, #windows, 5, 128[0:4] or 256[4])
            # seq_features: List[List[List[np.ndarray]]]
            # 64 * (T=16 / 8), 128 * (T=16 / 8)
            seq_features_per_k = pickle.load(f)
            exist_flags.append(k_clusters in seq_features_per_k)
    return len(exist_flags) !=0 and all(exist_flags)

def take_best_id(keypoints_by_id: Dict[str, List[Dict]]):
    max_id = -1
    max_len = -1
    for id, annotations in keypoints_by_id.items():
        if len(annotations) > max_len:
            max_len = len(annotations)
            max_id = id
    return max_id


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