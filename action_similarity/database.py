from typing import Dict, List

import os
import pickle

import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

from bpe import Config
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.utils import pad_to_height

from action_similarity.utils import exist_embeddings, parse_action_label, load_embeddings, save_embeddings, cache_file, take_best_id
from action_similarity.motion import compute_motion_embedding, extract_keypoints

class ActionDatabase():

    def __init__(
        self,
        config: Config = None,
        action_label_path: str = None,
    ):
        self.db = {}
        self.action_label_path = action_label_path

    def compute_standard_action_database(
        self, 
        data_path: str,
        model_path: str,
        config: Config = None,
        height = 1080,
        width = 1920,
    ):
        self.config = config
        self.similarity_analyzer = SimilarityAnalyzer(self.config, model_path)
        self.mean_pose_bpe = np.load(os.path.join(data_path, 'meanpose_rc_with_view_unit64.npy'))
        self.std_pose_bpe = np.load(os.path.join(data_path, 'stdpose_rc_with_view_unit64.npy'))

        self.actions = parse_action_label(self.action_label_path)
        h1, w1, self.scale = pad_to_height(self.config.img_size[0], height, width)
        print(f"[db] Load motion embedding...")
        # seq_features.shape == (#videos, #windows, 5, 128[0:4] or 256[4])
        # seq_features: List[List[List[np.ndarray]]]
        # 64 * (T=16 / 8), 128 * (T=16 / 8)
        assert exist_embeddings(config=config), f"The embeddings(k = {config.k_clusters}) not exist. "\
            f"You should run the main with --update or bin.postprocess with --k_clusters option"
        self.db = load_embeddings(config)
    
    def load_database(self, database_path: str, label_path: str):
        self.db = {}
        self.actions = parse_action_label(label_path)
        for db_filename in glob(database_path + '/*.pickle'):
            db_basename, _ = os.path.splitext(os.path.basename(db_filename))
            # 'action_embeddings_001' --> '001' --> 1
            action_idx = int(db_basename.split('_')[-1])
            with open(db_filename, 'rb') as f:
                embeddings = pickle.load(f)
                self.db[action_idx] = embeddings
