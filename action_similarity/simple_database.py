from typing import Dict, List

import os
import pickle

import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

from bpe import Config

from action_similarity.utils import exist_embeddings, parse_action_label, load_embeddings, save_embeddings, cache_file, take_best_id
from action_similarity.motion import compute_motion_embedding, extract_keypoints

class ActionDatabase():

    def __init__(
        self,
        action_label_path: str = None,
    ):
        self.db = {}
        self.action_label_path = action_label_path

    def compute_standard_action_database(
        self, 
        config: Config = None,
        height = 1080,
        width = 1920
    ):
        self.config = config
        self.actions = parse_action_label(self.action_label_path)
        print(f"[db] Load motion embedding...")
        # seq_features.shape == (#videos, #windows, 5, 128[0:4] or 256[4])
        # seq_features: List[List[List[np.ndarray]]]
        # 64 * (T=16 / 8), 128 * (T=16 / 8)
        assert exist_embeddings(config=config), f"The embeddings(k = {config.k_clusters}) not exist. "\
            f"You should run the main with --update or bin.postprocess with --k_clusters option"
        self.db = load_embeddings(config)
                   